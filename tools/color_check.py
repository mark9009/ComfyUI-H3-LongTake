"""Colour/style consistency between clips: Lab statistics (mean, std) of every clip,
dE distance between the last frame of clip N and the first of N+1, and between clip means.
Usage: color_check.py <project folder>"""
import os, subprocess, sys
import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe

ff = get_ffmpeg_exe()
W = 96


def frames(path):
    probe = subprocess.run([ff, "-v", "error", "-i", path, "-vf", f"scale={W}:-2", "-frames:v", "1",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], capture_output=True, check=True).stdout
    out = subprocess.run([ff, "-v", "error", "-i", path, "-an", "-vf", f"scale={W}:-2:flags=area",
                          "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], capture_output=True, check=True).stdout
    return np.frombuffer(out, np.uint8).reshape(-1, len(probe) // 3, 3).astype(np.float32) / 255.0


def lab_stats(px):
    px = px.reshape(-1, 3)
    m = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    lin = np.where(px <= 0.04045, px / 12.92, ((px + 0.055) / 1.055) ** 2.4)
    xyz = lin @ m.T / np.array([0.9505, 1.0, 1.089])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    L = 116 * f[:, 1] - 16
    a = 500 * (f[:, 0] - f[:, 1])
    b = 200 * (f[:, 1] - f[:, 2])
    lab = np.stack([L, a, b], 1)
    return lab.mean(0), lab.std(0)


proj = sys.argv[1]
clips = sorted(f for f in os.listdir(proj) if f.startswith("clip_") and f.endswith(".mp4") and "_ref" not in f)
stats = []
for c in clips:
    a = frames(os.path.join(proj, c))
    mean, std = lab_stats(a)
    stats.append((c, a[0], a[-1], mean))
    print(f"{c}: Lab mean L={mean[0]:5.1f} a={mean[1]:+5.1f} b={mean[2]:+5.1f}  std L={std[0]:4.1f} a={std[1]:4.1f} b={std[2]:4.1f}")
for (c0, _, last, m0), (c1, first, _, m1) in zip(stats, stats[1:]):
    dj = float(np.linalg.norm(lab_stats(last)[0] - lab_stats(first)[0]))
    dc = float(np.linalg.norm(m0 - m1))
    print(f"seam {c0} -> {c1}: dE last/first frame {dj:5.1f}   dE clip means {dc:5.1f}   (<5 imperceptible, >15 visible jump)")
