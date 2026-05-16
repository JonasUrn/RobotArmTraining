import os
import glob
import re
import numpy as np
import imageio
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import SAC
from env import PullBoxEnv

CKPT_DIR = "checkpoints"
EVAL_SEED = 1234
MAX_FRAMES_PER_CKPT = 90
HOLD_FRAMES = 10
RELEASE_STEPS = 30


def load_font(size=18):
    for path in [
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]:
        if os.path.exists(path):
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def annotate(frame, step, success_text):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    font = load_font(18)
    txt = f"step {step:>6}    {success_text}"
    draw.rectangle([(0, 0), (img.width, 28)], fill=(0, 0, 0))
    draw.text((8, 4), txt, fill=(255, 255, 255), font=font)
    return np.array(img)


def run_episode(env, model):
    obs, _ = env.reset(seed=EVAL_SEED)
    frames = [env.render_frame()]
    done = trunc = False
    success = False
    while not (done or trunc) and len(frames) < MAX_FRAMES_PER_CKPT:
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
    paths = sorted(glob.glob(os.path.join(CKPT_DIR, "ckpt_*.zip")))
    if not paths:
        print("No checkpoints found.")
        return
    env = PullBoxEnv(render=False)
    all_frames = []
    for path in paths:
        m = re.search(r"ckpt_(\d+)\.zip$", path)
        step = int(m.group(1)) if m else 0
        model = SAC.load(path, device="cpu")
        frames, success = run_episode(env, model)
        tag = "SUCCESS" if success else "..."
        annotated = [annotate(f, step, tag) for f in frames]
        annotated.extend([annotated[-1]] * HOLD_FRAMES)
        all_frames.extend(annotated)
        print(f"{path}: {len(frames)} frames, success={success}")
    env.close()
    imageio.mimsave("progress.gif", all_frames, fps=30, loop=0)
    print(f"Saved progress.gif with {len(all_frames)} frames from {len(paths)} checkpoints")


if __name__ == "__main__":
    main()
