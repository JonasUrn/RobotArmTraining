import os
import json
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces

HERE = os.path.dirname(os.path.abspath(__file__))
ARM_URDF = os.path.join(HERE, "arm.urdf")
SCENE_PATH = os.path.join(HERE, "scene.json")

ARM_BASE = [-0.45, 0.0, 0.0]
WALL_T = 0.01
WALL_H = 0.10
BALL_R = 0.022
BALL_JITTER = 0.02

if os.path.exists(SCENE_PATH):
    _scene = json.load(open(SCENE_PATH))
    BOX_CENTER = [_scene["box_center"][0], _scene["box_center"][1], 0.0]
    BOX_HALF_X, BOX_HALF_Y = _scene["box_half"]
    BALL_SPAWN = _scene["ball_spawn"]
else:
    BOX_CENTER = [0.0, 0.0, 0.0]
    BOX_HALF_X = BOX_HALF_Y = 0.075
    BALL_SPAWN = [0.0, 0.0]

J_LIMITS = [(-3.14, 3.14), (-1.57, 1.57), (-1.57, 1.57)]
J_INIT = [0.0, -0.5, 1.0]
DELTA = 0.08
MAX_STEPS = 80
SUBSTEPS = 5
GRASP_DIST = 0.05


class PullBoxEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, render=False):
        super().__init__()
        self.render_mode = "rgb_array"
        self.client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        self.action_space = spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)
        obs_dim = 3 + 3 + 3 + 3 + 1 + 1
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
        self.constraint = None
        self._build()

    def _build(self):
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.client)
        self.plane = p.loadURDF("plane.urdf", physicsClientId=self.client)
        self.arm = p.loadURDF(ARM_URDF, basePosition=ARM_BASE, useFixedBase=True, physicsClientId=self.client)
        self.j_idx = []
        self.finger_idx = []
        for i in range(p.getNumJoints(self.arm, physicsClientId=self.client)):
            info = p.getJointInfo(self.arm, i, physicsClientId=self.client)
            jname = info[1].decode()
            if info[2] == p.JOINT_REVOLUTE:
                self.j_idx.append(i)
            if info[2] == p.JOINT_PRISMATIC and jname in ("fl", "fr"):
                self.finger_idx.append(i)
                p.changeDynamics(self.arm, i, lateralFriction=0.05,
                                 spinningFriction=0.0, rollingFriction=0.0,
                                 contactStiffness=2000, contactDamping=50,
                                 physicsClientId=self.client)
            if info[12].decode() == "ee":
                self.ee_link = i
        self._build_box()
        self.ball = self._make_ball([0.0, 0.0, BALL_R + 0.005])

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
            p.changeDynamics(b, -1, lateralFriction=0.05,
                             spinningFriction=0.0, rollingFriction=0.0,
                             physicsClientId=self.client)
            self.walls.append(b)

    def _make_ball(self, pos):
        cs = p.createCollisionShape(p.GEOM_SPHERE, radius=BALL_R, physicsClientId=self.client)
        vs = p.createVisualShape(p.GEOM_SPHERE, radius=BALL_R, rgbaColor=[0.9, 0.1, 0.1, 1], physicsClientId=self.client)
        b = p.createMultiBody(0.05, cs, vs, basePosition=pos, physicsClientId=self.client)
        p.changeDynamics(b, -1, lateralFriction=1.0, rollingFriction=0.005, physicsClientId=self.client)
        return b

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.constraint is not None:
            try:
                p.removeConstraint(self.constraint, physicsClientId=self.client)
            except Exception:
                pass
            self.constraint = None
        for i, j in enumerate(self.j_idx):
            p.resetJointState(self.arm, j, J_INIT[i], physicsClientId=self.client)
        for j in self.finger_idx:
            p.resetJointState(self.arm, j, 0.04, physicsClientId=self.client)
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
        self.steps = 0
        self.grasped = False
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
        jpos = [p.getJointState(self.arm, j, physicsClientId=self.client)[0] for j in self.j_idx]
        ee = self._ee_pos()
        ball = self._ball_pos()
        rel = ball - ee
        return np.concatenate([jpos, ee, ball, rel, [ball[2]], [1.0 if self.grasped else 0.0]]).astype(np.float32)

    def _try_grasp(self, cmd):
        ee = self._ee_pos()
        ball = self._ball_pos()
        d = np.linalg.norm(ee - ball)
        if cmd > 0 and not self.grasped and d < GRASP_DIST:
            offset = ball - ee
            self.constraint = p.createConstraint(
                self.arm, self.ee_link, self.ball, -1, p.JOINT_FIXED,
                [0, 0, 0], [offset[0], offset[1], offset[2]], [0, 0, 0],
                physicsClientId=self.client,
            )
            self.grasped = True
        elif cmd <= 0 and self.grasped:
            try:
                p.removeConstraint(self.constraint, physicsClientId=self.client)
            except Exception:
                pass
            self.constraint = None
            self.grasped = False

    def step(self, action):
        action = np.clip(action, -1.0, 1.0)
        for i, j in enumerate(self.j_idx):
            cur = p.getJointState(self.arm, j, physicsClientId=self.client)[0]
            tgt = cur + float(action[i]) * DELTA
            lo, hi = J_LIMITS[i]
            tgt = float(np.clip(tgt, lo, hi))
            p.setJointMotorControl2(self.arm, j, p.POSITION_CONTROL, targetPosition=tgt,
                                    force=40, maxVelocity=3, physicsClientId=self.client)
        finger_target = 0.012 if action[3] > 0 else 0.04
        for j in self.finger_idx:
            p.setJointMotorControl2(self.arm, j, p.POSITION_CONTROL,
                                    targetPosition=finger_target,
                                    force=20, maxVelocity=0.5,
                                    physicsClientId=self.client)
        self._try_grasp(action[3])
        for _ in range(SUBSTEPS):
            p.stepSimulation(physicsClientId=self.client)

        ee = self._ee_pos()
        ball = self._ball_pos()
        d = np.linalg.norm(ee - ball)
        dx = max(0.0, abs(ball[0] - BOX_CENTER[0]) - BOX_HALF_X)
        dy = max(0.0, abs(ball[1] - BOX_CENTER[1]) - BOX_HALF_Y)
        outward = max(dx, dy)
        outside = outward > BALL_R + 0.01
        reward = -d
        if self.grasped:
            lift = min(max(0.0, ball[2]), WALL_H + 0.02)
            reward += 0.1 + 1.0 * lift + 3.0 * min(outward, 0.15)
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
        view = p.computeViewMatrix(cameraEyePosition=[0.6, -0.7, 0.6],
                                   cameraTargetPosition=[0.0, 0.0, 0.1],
                                   cameraUpVector=[0, 0, 1])
        proj = p.computeProjectionMatrixFOV(fov=55, aspect=w / h, nearVal=0.05, farVal=3.0)
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
