import os
import time
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from env import PullBoxEnv

# Contact-gated grasping on the Panda is a harder task than the old distance
# magnet, so we let training use most of the hour. The time-budget callback
# still saves a usable checkpoint if it stops before TOTAL_STEPS.
TOTAL_STEPS = 150_000
TIME_BUDGET_SEC = 50 * 60
CKPT_EVERY = 5_000
CKPT_DIR = "checkpoints"


class LogCallback(BaseCallback):
    def __init__(self, time_budget, ckpt_every, ckpt_dir):
        super().__init__()
        self.time_budget = time_budget
        self.ckpt_every = ckpt_every
        self.ckpt_dir = ckpt_dir
        self.next_ckpt = ckpt_every
        self.start = None
        self.ep_rewards = []
        self.ep_success = []
        self.ep_steps = []
        self.last_logged = 0
        os.makedirs(ckpt_dir, exist_ok=True)

    def _on_training_start(self):
        self.start = time.time()
        self.model.save(os.path.join(self.ckpt_dir, "ckpt_000000"))

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                self.ep_rewards.append(ep["r"])
                self.ep_success.append(1.0 if info.get("is_success", False) else 0.0)
                self.ep_steps.append(self.num_timesteps)
        if self.num_timesteps >= self.next_ckpt:
            path = os.path.join(self.ckpt_dir, f"ckpt_{self.num_timesteps:06d}")
            self.model.save(path)
            self.next_ckpt += self.ckpt_every
        if time.time() - self.start > self.time_budget:
            print("Time budget reached, stopping.")
            self.model.save(os.path.join(self.ckpt_dir, f"ckpt_{self.num_timesteps:06d}"))
            return False
        n_eps = len(self.ep_rewards)
        if n_eps >= self.last_logged + 25:
            self.last_logged = n_eps
            mr = np.mean(self.ep_rewards[-25:])
            sr = np.mean(self.ep_success[-25:])
            print(f"steps={self.num_timesteps} eps={n_eps} mean_r={mr:.2f} succ={sr:.2f}")
        return True


def main():
    env = Monitor(PullBoxEnv(render=False), info_keywords=("is_success",))
    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=1500,
        batch_size=256,
        tau=0.005,
        gamma=0.98,
        train_freq=1,
        gradient_steps=2,
        policy_kwargs=dict(net_arch=[128, 128]),
        verbose=0,
        device="cpu",
    )
    cb = LogCallback(TIME_BUDGET_SEC, CKPT_EVERY, CKPT_DIR)
    t0 = time.time()
    model.learn(total_timesteps=TOTAL_STEPS, callback=cb)
    print(f"Training wall time: {time.time() - t0:.1f}s")
    model.save("sac_pullbox")

    rewards = np.array(cb.ep_rewards)
    success = np.array(cb.ep_success)
    steps = np.array(cb.ep_steps)
    np.savez("training_log.npz", rewards=rewards, success=success, steps=steps)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(steps, rewards, alpha=0.3, label="ep reward")
    if len(rewards) >= 20:
        w = 20
        ma = np.convolve(rewards, np.ones(w) / w, mode="valid")
        axes[0].plot(steps[w - 1:], ma, label=f"MA({w})")
    axes[0].set_xlabel("env steps")
    axes[0].set_ylabel("episode reward")
    axes[0].set_title("Training reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if len(success) >= 20:
        w = 20
        sma = np.convolve(success, np.ones(w) / w, mode="valid")
        axes[1].plot(steps[w - 1:], sma)
    else:
        axes[1].plot(steps, success)
    axes[1].set_xlabel("env steps")
    axes[1].set_ylabel("success rate (rolling)")
    axes[1].set_title("Success rate")
    axes[1].set_ylim(-0.05, 1.05)
    axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("training.png", dpi=120)
    print("Saved training.png and sac_pullbox.zip")
    env.close()


if __name__ == "__main__":
    main()
