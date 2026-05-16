import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

# Reachable annulus for the Panda (env.py mounts it on a 0.35 m pedestal and
# uses IK each reset, so it adapts to any box drawn within this ring). The box
# must also be big enough that the real Panda gripper physically fits inside
# (MIN_HALF) and the walls are a shallow 0.04 m tray (set in env.py).
ARM_BASE_XY = (-0.45, 0.0)
REACH_MIN = 0.32
REACH_MAX = 0.72
MIN_HALF = 0.09          # min box half-extent so the real gripper fits inside
PLOT_LIM = 0.8
BALL_R = 0.025


def in_reach(x, y):
    d = np.hypot(x - ARM_BASE_XY[0], y - ARM_BASE_XY[1])
    return REACH_MIN <= d <= REACH_MAX


def main():
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-PLOT_LIM, PLOT_LIM)
    ax.set_ylim(-PLOT_LIM, PLOT_LIM)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    annulus_outer = Circle(ARM_BASE_XY, REACH_MAX, color="green", alpha=0.08)
    annulus_inner = Circle(ARM_BASE_XY, REACH_MIN, color="white", alpha=1.0, zorder=2)
    ax.add_patch(annulus_outer)
    ax.add_patch(annulus_inner)
    ax.plot(*ARM_BASE_XY, "rs", markersize=12, label="arm base")
    ax.set_title("Click 2 corners for the BOX")
    ax.legend(loc="upper right")

    print("Click 2 corners of the BOX (must be inside the green ring).")
    while True:
        pts = plt.ginput(2, timeout=0)
        if len(pts) < 2:
            print("Need 2 clicks. Retry.")
            continue
        (x1, y1), (x2, y2) = pts
        if not (in_reach(x1, y1) and in_reach(x2, y2)):
            print("Both corners must be inside the green reachable ring. Retry.")
            continue
        break

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    hx = abs(x1 - x2) / 2
    hy = abs(y1 - y2) / 2
    if hx < MIN_HALF or hy < MIN_HALF:
        print(f"Box too small for the Panda gripper "
              f"(need each half-extent >= {MIN_HALF} m). Aborting.")
        return

    rect = Rectangle((cx - hx, cy - hy), 2 * hx, 2 * hy,
                     linewidth=2, edgecolor="brown", facecolor="none")
    ax.add_patch(rect)
    ax.set_title("Click 1 point for the BALL spawn center (inside the box)")
    fig.canvas.draw()

    print("Click 1 point for BALL spawn center (must be inside the box).")
    while True:
        pts = plt.ginput(1, timeout=0)
        if len(pts) < 1:
            continue
        bx, by = pts[0]
        if abs(bx - cx) > hx - BALL_R - 0.005 or abs(by - cy) > hy - BALL_R - 0.005:
            print("Ball must be inside the box. Retry.")
            continue
        if not in_reach(bx, by):
            print("Ball must be inside the green reachable ring. Retry.")
            continue
        break

    ax.plot(bx, by, "ro", markersize=10)
    ax.set_title("Saved scene.json — close the window")
    fig.canvas.draw()

    scene = {
        "box_center": [cx, cy],
        "box_half": [hx, hy],
        "ball_spawn": [bx, by],
    }
    with open("scene.json", "w") as f:
        json.dump(scene, f, indent=2)
    print(f"Saved scene.json: box_center=({cx:.3f},{cy:.3f}) half=({hx:.3f},{hy:.3f}) ball=({bx:.3f},{by:.3f})")
    plt.show()


if __name__ == "__main__":
    main()
