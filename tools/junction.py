import subprocess, numpy as np, sys, os
from imageio_ffmpeg import get_ffmpeg_exe
ff = get_ffmpeg_exe(); W, H = 48, 80
def decode(path):
    out = subprocess.run([ff, "-v", "error", "-i", path, "-an", "-vf", f"scale={W}:{H}:flags=area,format=gray",
                          "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"], capture_output=True, check=True).stdout
    return np.frombuffer(out, np.uint8).astype(np.float32).reshape(-1, H, W) / 255.0
def sim(a, b):
    a = a.reshape(-1) - a.mean(); b = b.reshape(-1) - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
base = os.path.join(os.environ.get("COMFY_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))), "output", "h3_longtake")
print(f"{'project':16} {'sim last0/first1':>18} {'|diff| jump at seam':>20} {'mean inner diff':>20}")
for proj in sys.argv[1:]:
    g0 = decode(os.path.join(base, proj, "clip_000.mp4")); g1 = decode(os.path.join(base, proj, "clip_001.mp4"))
    jump = float(np.abs(g1[0] - g0[-1]).mean())
    inner = float(np.abs(np.diff(np.concatenate([g0[-12:], g1[:12]]), axis=0)).mean(axis=(1, 2))[[i for i in range(23) if i != 11]].mean())
    print(f"{proj:16} {sim(g0[-1], g1[0]):18.2f} {jump:20.4f} {inner:20.4f}")
