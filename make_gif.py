import numpy as np
import imageio
from stable_baselines3 import SAC
from env import PullBoxEnv

MAX_TRIES = 20
RELEASE_STEPS = 50
HOLD_FRAMES = 20


def run_episode(env, model, seed):
    obs, _ = env.reset(seed=seed)
    frames = [env.render_frame()]
    done = False
    trunc = False
    success = False
    while not (done or trunc):
        action, _ = model.predict(obs, deterministic=True)
        obs, r, done, trunc, info = env.step(action)
        frames.append(env.render_frame())
        if info.get("is_success"):
            success = True
    if success:
        release = np.zeros(env.action_space.shape, dtype=np.float32)
        release[-1] = -1.0
        for _ in range(RELEASE_STEPS):
            env.step(release)
            frames.append(env.render_frame())
    return frames, success


def main():
    env = PullBoxEnv(render=False)
    model = SAC.load("sac_pullbox", device="cpu")
    chosen = None
    for i in range(MAX_TRIES):
        frames, success = run_episode(env, model, seed=1000 + i)
        print(f"try {i}: success={success}, frames={len(frames)}")
        if success:
            chosen = frames
            break
    if chosen is None:
        print("No successful episode found; using last attempt.")
        chosen = frames
    chosen.extend([chosen[-1]] * HOLD_FRAMES)
    env.close()
    imageio.mimsave("rollout.gif", chosen, fps=30, loop=0)
    print(f"Saved rollout.gif with {len(chosen)} frames")


if __name__ == "__main__":
    main()
