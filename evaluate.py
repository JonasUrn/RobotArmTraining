import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from env import PullBoxEnv

CKPT_DIR = "checkpoints"
N_EPISODES = 10


def main():
    paths = sorted(glob.glob(os.path.join(CKPT_DIR, "ckpt_*.zip")))
    if not paths:
        print("No checkpoints found.")
        return
    env = PullBoxEnv(render=False)

    steps = []
    rates = []
    for path in paths:
        m = re.search(r"ckpt_(\d+)\.zip$", path)
        step = int(m.group(1)) if m else 0
        model = SAC.load(path, device="cpu")
        successes = 0
        for i in range(N_EPISODES):
            obs, _ = env.reset(seed=1000 + i)
            done = False
            trunc = False
            success = False
            while not (done or trunc):
                action, _ = model.predict(obs, deterministic=True)
                obs, r, done, trunc, info = env.step(action)
                if info.get("is_success"):
                    success = True
            successes += int(success)
        rate = successes / N_EPISODES
        steps.append(step)
        rates.append(rate)
        print(f"step {step}: success {successes}/{N_EPISODES} = {rate:.2f}")

    env.close()

    plt.figure(figsize=(8, 4))
    plt.plot(steps, rates, marker="o")
    plt.xlabel("steps")
    plt.ylabel("success rate")
    plt.title(f"Success rate")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("evaluation.png", dpi=120)
    print("Saved evaluation.png")


if __name__ == "__main__":
    main()
