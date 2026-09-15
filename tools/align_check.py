"""Measures how closely the generated video follows the source, clip by clip.
- lag: delay (frames) that maximises the motion-energy correlation
- sim: mean frame-to-frame correlation (lag 0) between 48x80 greyscale frames
Usage: align_check.py <project folder> <source video>
"""
import json, os, subprocess, sys, tempfile
import numpy as np
from imageio_ffmpeg import get_ffmpeg_exe

ff = get_ffmpeg_exe()
W, H = 48, 80


def decode(path, n, start_seconds=0.0):
    cmd = [ff, "-v", "error", "-i", path, "-an", "-vf",
           f"fps=24,select=gte(n\\,{int(round(start_seconds * 24))}),setpts=N/(24*TB),scale={W}:{H}:flags=area,format=gray",
           "-vsync", "0", "-frames:v", str(n), "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
    out = subprocess.run(cmd, capture_output=True, check=True).stdout
    a = np.frombuffer(out, np.uint8).astype(np.float32) / 255.0
    return a.reshape(-1, H, W)


proj, src = sys.argv[1], sys.argv[2]
meta = json.load(open(os.path.join(proj, "plan.json")))
plan = meta["plan"]
offset = float(meta.get("source_offset_seconds", 0.0) or 0.0)
avail = [c for c in plan["clips"] if os.path.isfile(os.path.join(proj, "clip_%03d.mp4" % c["index"]))]
total = sum(c["new"] for c in avail)

lst = os.path.join(tempfile.gettempdir(), "lt_concat.txt")
prev = os.path.join(tempfile.gettempdir(), "lt_preview.mp4")
paths = [os.path.abspath(os.path.join(proj, "clip_%03d.mp4" % c["index"])).replace("\\", "/") for c in avail]
with open(lst, "w") as fh:
    for q in paths:
        fh.write("file '%s'\n" % q)
subprocess.run([ff, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", prev], check=True)

gen = decode(prev, total)
ref = decode(src, total, offset)
n = min(len(gen), len(ref)); gen, ref = gen[:n], ref[:n]
print(f"clips available: {len(avail)}  frames compared: {n}")


def energy(v):
    e = np.abs(np.diff(v, axis=0)).mean(axis=(1, 2))
    return (e - e.mean()) / (e.std() + 1e-8)


def best_lag(a, b, maxlag=36):
    best = (0, -9)
    for lag in range(-maxlag, maxlag + 1):
        if lag >= 0:
            x, y = a[lag:], b[:len(b) - lag]
        else:
            x, y = a[:len(a) + lag], b[-lag:]
        m = min(len(x), len(y))
        if m < 20:
            continue
        c = float(np.corrcoef(x[:m], y[:m])[0, 1])
        if c > best[1]:
            best = (lag, c)
    return best


def frame_sim(a, b):
    a = a.reshape(len(a), -1); b = b.reshape(len(b), -1)
    a = a - a.mean(1, keepdims=True); b = b - b.mean(1, keepdims=True)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-8)


sim_all = frame_sim(gen, ref)
cursor = 0
print(f"{'clip':>4} {'frame':>10} {'lag':>5} {'corr_mov':>8} {'sim_frame':>9}  (sim per quarter)")
for c in avail:
    a, b = cursor, min(cursor + c["new"], n)
    if b - a < 24:
        break
    lag, corr = best_lag(energy(gen[a:b]), energy(ref[a:b]))
    s = sim_all[a:b]
    q = " ".join(f"{x.mean():.2f}" for x in np.array_split(s, 4))
    print(f"{c['index']:>4} {a:>4}-{b - 1:<5} {lag:>+5} {corr:>8.2f} {s.mean():>9.2f}  {q}")
    cursor = b

print("\nframe-to-frame similarity over 48-frame windows (2 s):")
for a in range(0, n, 48):
    s = sim_all[a:a + 48].mean()
    print(f"  {a:>4}-{min(a + 47, n - 1):<4} {s:5.2f} {'#' * int(max(0, s) * 40)}")
