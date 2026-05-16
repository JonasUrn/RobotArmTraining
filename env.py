import os
import json
import math
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces


HERE = os.path.dirname(os.path.abspath(__file__))
PANDA_URDF = os.path.join(pybullet_data.getDataPath(), "franka_panda", "panda.urdf")
SCENE_PATH = os.path.join(HERE, "scene.json")

ARM_BASE = [-0.45, 0.0, 0.35]
PEDESTAL_HALF = [0.06, 0.06, 0.175]
WALL_T = 0.03
WALL_H = 0.04
BALL_R = 0.025
BALL_JITTER = 0.02
HOVER = 0.09

if os.path.exists(SCENE_PATH):
    _scene = json.load(open(SCENE_PATH))
    BOX_CENTER = [_scene["box_center"][0], _scene["box_center"][1], 0.0]
    BOX_HALF_X, BOX_HALF_Y = _scene["box_half"]
    BALL_SPAWN = _scene["ball_spawn"]
else:
    BOX_CENTER = [0.0, 0.0, 0.0]
    BOX_HALF_X = BOX_HALF_Y = 0.10
    BALL_SPAWN = [0.0, 0.0]

ACTUATED = ["panda_joint1", "panda_joint2", "panda_joint4", "panda_joint6"]

DELTA = 0.06
PANDA_FORCE = 120.0
PANDA_VMAX = 1.2
HOLD_FORCE = 160.0
FINGER_FORCE = 30.0
FINGER_VMAX = 0.10
FINGER_OPEN = 0.04
FINGER_CLOSE = 0.0
GRASP_NEAR = 0.07

MAX_STEPS = 90
SUBSTEPS = 8
DOWN_ORN = p.getQuaternionFromEuler([math.pi, 0.0, 0.0])


class PullBoxEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render=False):
        super().__init__()
        self.render_mode = "rgb_array"
        self.client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        n_act = len(ACTUATED)
        self.action_space = spaces.Box(-1.0, 1.0, (n_act + 1,), dtype=np.float32)
        obs_dim = n_act + 3 + 3 + 3 + 1 + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
        self.constraint = None
        self._build()

    def _build(self):
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.client)
        p.setPhysicsEngineParameter(numSolverIterations=80,
                                    contactBreakingThreshold=0.001,
                                    physicsClientId=self.client)
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.client)
        self._build_pedestal()
        self.arm = p.loadURDF(PANDA_URDF, basePosition=ARM_BASE,
                              useFixedBase=True, physicsClientId=self.client)

        self.name2joint = {}
        link2idx = {}
        self.movable = []
        for i in range(p.getNumJoints(self.arm, physicsClientId=self.client)):
            info = p.getJointInfo(self.arm, i, physicsClientId=self.client)
            jname = info[1].decode()
            lname = info[12].decode()
            self.name2joint[jname] = i
            link2idx[lname] = i
            if info[2] in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                self.movable.append(i)
        self.act_idx = [self.name2joint[n] for n in ACTUATED]
        self.finger_idx = [self.name2joint["panda_finger_joint1"],
                           self.name2joint["panda_finger_joint2"]]
        self.left_finger_link = self.name2joint["panda_finger_joint1"]
        self.right_finger_link = self.name2joint["panda_finger_joint2"]
        self.hand_idx = link2idx["panda_hand"]
        self.ee_link = link2idx["panda_grasptarget"]
        
        self.hold_idx = [j for j in self.movable
                         if j not in self.act_idx and j not in self.finger_idx]
        self.j_limits = []
        for j in self.act_idx:
            info = p.getJointInfo(self.arm, j, physicsClientId=self.client)
            self.j_limits.append((info[8], info[9]))

        for j in self.finger_idx:
            p.changeDynamics(self.arm, j, lateralFriction=1.2,
                             spinningFriction=0.1, rollingFriction=0.0,
                             contactStiffness=30000, contactDamping=1000,
                             frictionAnchor=1, physicsClientId=self.client)

        self._build_box()
        self.ball = self._make_ball([BALL_SPAWN[0], BALL_SPAWN[1], BALL_R + 0.005])
        self.hold_targets = {}

    def _build_pedestal(self):
        hs = PEDESTAL_HALF
        cs = p.createCollisionShape(p.GEOM_BOX, halfExtents=hs, physicsClientId=self.client)
        vs = p.createVisualShape(p.GEOM_BOX, halfExtents=hs,
                                 rgbaColor=[0.25, 0.25, 0.25, 1], physicsClientId=self.client)
        p.createMultiBody(0, cs, vs,
                          basePosition=[ARM_BASE[0], ARM_BASE[1], hs[2]],
                          physicsClientId=self.client)

    def _build_box(self):
        cx, cy, _ = BOX_CENTER
        hx, hy = BOX_HALF_X, BOX_HALF_Y
        h = WALL_H
        wt = WALL_T
        walls = [
            ([hx + wt / 2, 0, h / 2], [wt / 2, hy + wt, h / 2]),
            ([-hx - wt / 2, 0, h / 2], [wt / 2, hy + wt, h / 2]),
            ([0, hy + wt / 2, h / 2], [hx + wt, wt / 2, h / 2]),
            ([0, -hy - wt / 2, h / 2], [hx + wt, wt / 2, h / 2]),
        ]
        self.walls = []
        for pos, hs in walls:
            cs = p.createCollisionShape(p.GEOM_BOX, halfExtents=hs, physicsClientId=self.client)
            vs = p.createVisualShape(p.GEOM_BOX, halfExtents=hs, rgbaColor=[0.6, 0.4, 0.2, 1], physicsClientId=self.client)
            b = p.createMultiBody(0, cs, vs, basePosition=[cx + pos[0], cy + pos[1], pos[2]], physicsClientId=self.client)
            p.changeDynamics(b, -1, lateralFriction=0.6, restitution=0.0,
                             contactStiffness=30000, contactDamping=1000,
                             contactProcessingThreshold=0.0,
                             physicsClientId=self.client)
            self.walls.append(b)

    def _make_ball(self, pos):
        cs = p.createCollisionShape(p.GEOM_SPHERE, radius=BALL_R, physicsClientId=self.client)
        vs = p.createVisualShape(p.GEOM_SPHERE, radius=BALL_R, rgbaColor=[0.9, 0.1, 0.1, 1], physicsClientId=self.client)
        b = p.createMultiBody(0.05, cs, vs, basePosition=pos, physicsClientId=self.client)
        p.changeDynamics(b, -1, lateralFriction=1.0, rollingFriction=0.002,
                         spinningFriction=0.002, restitution=0.0,
                         contactStiffness=30000, contactDamping=1000,
                         physicsClientId=self.client)
        return b

    def _ik(self, target_pos):
        sol = p.calculateInverseKinematics(
            self.arm, self.ee_link, list(target_pos), DOWN_ORN,
            maxNumIterations=200, residualThreshold=1e-4,
            physicsClientId=self.client)
        return list(sol)

    def _apply_holds(self):
        for idx, val in self.hold_targets.items():
            p.setJointMotorControl2(self.arm, idx, p.POSITION_CONTROL,
                                    targetPosition=val, force=HOLD_FORCE,
                                    maxVelocity=PANDA_VMAX, physicsClientId=self.client)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.constraint is not None:
            try:
                p.removeConstraint(self.constraint, physicsClientId=self.client)
            except Exception:
                pass
            self.constraint = None

        jx = self.np_random.uniform(-BALL_JITTER, BALL_JITTER)
        jy = self.np_random.uniform(-BALL_JITTER, BALL_JITTER)
        bx = float(np.clip(BALL_SPAWN[0] + jx,
                           BOX_CENTER[0] - BOX_HALF_X + BALL_R + 0.005,
                           BOX_CENTER[0] + BOX_HALF_X - BALL_R - 0.005))
        by = float(np.clip(BALL_SPAWN[1] + jy,
                           BOX_CENTER[1] - BOX_HALF_Y + BALL_R + 0.005,
                           BOX_CENTER[1] + BOX_HALF_Y - BALL_R - 0.005))
        p.resetBasePositionAndOrientation(self.ball, [bx, by, BALL_R + 0.002], [0, 0, 0, 1], physicsClientId=self.client)
        p.resetBaseVelocity(self.ball, [0, 0, 0], [0, 0, 0], physicsClientId=self.client)

        
        target = [bx, by, BALL_R + HOVER]
        sol = self._ik(target)
        self.hold_targets = {}
        for k, j in enumerate(self.movable):
            if j in self.finger_idx:
                continue
            val = float(sol[k])
            p.resetJointState(self.arm, j, val, 0.0, physicsClientId=self.client)
            if j in self.hold_idx:
                self.hold_targets[j] = val
        for j in self.finger_idx:
            p.resetJointState(self.arm, j, FINGER_OPEN, 0.0, physicsClientId=self.client)

        self.steps = 0
        self.grasped = False
        self._apply_holds()
        for _ in range(5):
            p.stepSimulation(physicsClientId=self.client)
        return self._obs(), {}

    def _ee_pos(self):
        st = p.getLinkState(self.arm, self.ee_link, physicsClientId=self.client)
        return np.array(st[0])

    def _ball_pos(self):
        pos, _ = p.getBasePositionAndOrientation(self.ball, physicsClientId=self.client)
        return np.array(pos)

    def _obs(self):
        jpos = [p.getJointState(self.arm, j, physicsClientId=self.client)[0] for j in self.act_idx]
        ee = self._ee_pos()
        ball = self._ball_pos()
        rel = ball - ee
        return np.concatenate([jpos, ee, ball, rel, [ball[2]], [1.0 if self.grasped else 0.0]]).astype(np.float32)

    def _finger_contact(self, finger_link):
        cps = p.getContactPoints(bodyA=self.arm, bodyB=self.ball,
                                 linkIndexA=finger_link,
                                 physicsClientId=self.client)
        return any(c[9] > 0.5 for c in cps)

    def _update_grasp(self, grip_cmd):
        left = self._finger_contact(self.left_finger_link)
        right = self._finger_contact(self.right_finger_link)
        if grip_cmd > 0 and not self.grasped and left and right:
            hand_pos, hand_orn = p.getLinkState(self.arm, self.hand_idx,
                                                physicsClientId=self.client)[:2]
            ball_pos, _ = p.getBasePositionAndOrientation(self.ball, physicsClientId=self.client)
            inv_p, inv_o = p.invertTransform(hand_pos, hand_orn)
            rel_p, rel_o = p.multiplyTransforms(inv_p, inv_o, ball_pos, [0, 0, 0, 1])
            self.constraint = p.createConstraint(
                self.arm, self.hand_idx, self.ball, -1, p.JOINT_FIXED,
                [0, 0, 0], rel_p, [0, 0, 0], rel_o,
                physicsClientId=self.client,
            )
            p.changeConstraint(self.constraint, maxForce=50, physicsClientId=self.client)
            self.grasped = True
        elif grip_cmd <= 0 and self.grasped:
            try:
                p.removeConstraint(self.constraint, physicsClientId=self.client)
            except Exception:
                pass
            self.constraint = None
            self.grasped = False
        return left, right

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        for i, j in enumerate(self.act_idx):
            cur = p.getJointState(self.arm, j, physicsClientId=self.client)[0]
            lo, hi = self.j_limits[i]
            tgt = float(np.clip(cur + float(action[i]) * DELTA, lo, hi))
            p.setJointMotorControl2(self.arm, j, p.POSITION_CONTROL, targetPosition=tgt,
                                    force=PANDA_FORCE, maxVelocity=PANDA_VMAX,
                                    physicsClientId=self.client)
        self._apply_holds()
        grip_cmd = action[-1]
        finger_target = FINGER_CLOSE if grip_cmd > 0 else FINGER_OPEN
        for j in self.finger_idx:
            p.setJointMotorControl2(self.arm, j, p.POSITION_CONTROL,
                                    targetPosition=finger_target,
                                    force=FINGER_FORCE, maxVelocity=FINGER_VMAX,
                                    physicsClientId=self.client)
        for _ in range(SUBSTEPS):
            p.stepSimulation(physicsClientId=self.client)
        left, right = self._update_grasp(grip_cmd)

        ee = self._ee_pos()
        ball = self._ball_pos()
        d = np.linalg.norm(ee - ball)
        dx = max(0.0, abs(ball[0] - BOX_CENTER[0]) - BOX_HALF_X)
        dy = max(0.0, abs(ball[1] - BOX_CENTER[1]) - BOX_HALF_Y)
        outward = max(dx, dy)
        outside = outward > BALL_R + 0.01
        reward = -d
        if self.grasped:
            lift = min(max(0.0, ball[2]), WALL_H + 0.06)
            reward += 0.1 + 1.0 * lift + 3.0 * min(outward, 0.15)
        else:
            if d < GRASP_NEAR and grip_cmd > 0:
                reward += 0.05
            if left or right:
                reward += 0.02
        reward -= 0.001 * float(np.sum(np.square(action)))
        reward -= 0.05
        terminated = False
        info = {"is_success": False}
        if outside:
            reward += 50.0
            terminated = True
            info["is_success"] = True
        self.steps += 1
        truncated = self.steps >= MAX_STEPS
        return self._obs(), float(reward), terminated, truncated, info

    def render_frame(self, w=480, h=360):
        cx, cy = BOX_CENTER[0], BOX_CENTER[1]
        view = p.computeViewMatrix(cameraEyePosition=[cx + 0.7, cy - 0.9, 0.7],
                                   cameraTargetPosition=[cx, cy, 0.10],
                                   cameraUpVector=[0, 0, 1])
        proj = p.computeProjectionMatrixFOV(fov=60, aspect=w / h, nearVal=0.05, farVal=3.0)
        _, _, rgba, _, _ = p.getCameraImage(w, h, view, proj,
                                            renderer=p.ER_TINY_RENDERER,
                                            physicsClientId=self.client)
        arr = np.reshape(np.array(rgba, dtype=np.uint8), (h, w, 4))[:, :, :3]
        return arr

    def close(self):
        try:
            p.disconnect(physicsClientId=self.client)
        except Exception:
            pass
