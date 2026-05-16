import numpy as np
import matplotlib.pyplot as plt
import env

env.BALL_JITTER = 0.0
env.BOX_CENTER = [0.06623376623376631, -0.209090909090909, 0.0]
env.BOX_HALF_X = 0.10584415584415585
env.BOX_HALF_Y = 0.08701298701298699
env.BALL_SPAWN = [0.06948051948051959, -0.2012987012987012]

from train import main_train

EXP_STEPS = 40_000
EXP_TIME_BUDGET = 60 * 60
SEED = 0
CONFIGS = ["shaped", "sparse", "no_bonus"]
W = 20


def rolling(steps, success):
    if len(success) >= W:
        sma = np.convolve(success, np.ones(W) / W, mode="valid")
        return steps[W - 1:], sma
    return steps, success


def main():
    results = {}
    for mode in CONFIGS:
        print(f"\n=== Training config: reward={mode} ===")
        env.REWARD_MODE = mode
        results[mode] = main_train(
            total_steps=EXP_STEPS,
            time_budget_sec=EXP_TIME_BUDGET,
            save_checkpoints=False,
            save_outputs=False,
            seed=SEED,
        )

    np.savez("experiment.npz", **{
        f"{m}_{k}": v for m, r in results.items() for k, v in r.items()
    })

    plt.figure(figsize=(9, 5))
    print("\n=== Summary ===")
    print(f"{'config':<10} {'final_succ':>10} {'first_succ_step':>16} {'steps_to_0.5':>14}")
    for mode in CONFIGS:
        r = results[mode]
        steps, success = r["steps"], r["success"]
        if len(steps) == 0:
            print(f"{mode:<10} {'no episodes':>10}")
            continue
        x, y = rolling(steps, success)
        plt.plot(x, y, label=mode)

        final_succ = float(np.mean(success[-25:])) if len(success) else 0.0
        succ_idx = np.where(success > 0.5)[0]
        first_succ = int(steps[succ_idx[0]]) if len(succ_idx) else -1
        hit = np.where(y >= 0.5)[0]
        steps_to_half = int(x[hit[0]]) if len(hit) else -1
        fs = first_succ if first_succ >= 0 else "never"
        sh = steps_to_half if steps_to_half >= 0 else "never"
        print(f"{mode:<10} {final_succ:>10.2f} {str(fs):>16} {str(sh):>14}")

    plt.xlabel("env steps")
    plt.ylabel("success rate (rolling)")
    plt.title(f"Reward function comparison (fixed box, {EXP_STEPS} steps)")
    plt.ylim(-0.05, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("experiment.png", dpi=120)
    print("\nSaved experiment.png and experiment.npz")


if __name__ == "__main__":
    main()
