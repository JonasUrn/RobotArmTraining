import imageio
import numpy as np
from stable_baselines3 import SAC
from env import PullBoxEnv

N_EPISODES = 3
MAX_FRAMES = 400


def main():
    env = PullBoxEnv(render=False)
    model = SAC.load("sac_pullbox", device="cpu")
    frames = []
    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=100 + ep)
        frames.append(env.render_frame())
        done = False
        trunc = False
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, info = env.step(action)
            frames.append(env.render_frame())
            if len(frames) >= MAX_FRAMES:
                break
        if len(frames) >= MAX_FRAMES:
            break
    env.close()
    imageio.mimsave("rollout.gif", frames, fps=30, loop=0)
    print(f"Saved rollout.gif with {len(frames)} frames")


if __name__ == "__main__":
    main()
