import os, sys, tempfile, subprocess
COMFY = os.environ.get("COMFY_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, COMFY)
os.chdir(COMFY)
import torch, importlib
m = importlib.import_module("custom_nodes.ComfyUI-H3-LongTake.longtake_nodes")
ff = m._find_ffmpeg()
tmp = tempfile.mkdtemp()

def make_video(path, n, fps, w=128, h=96):
    cmd = [ff, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "pipe:0", "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", path]
    data = b"".join(bytes([min(255, i)]) * (w * h) for i in range(n))
    subprocess.run(cmd, input=data, check=True)

def frame_values(frames):  # mean luminance per frame, in index units
    return [round(float(f.mean()) * 255) for f in frames]

# 30 fps, 150 frames = 5 s -> 120 frames at 24 fps
p30 = os.path.join(tmp, "src30.mp4"); make_video(p30, 150, 30)
s = m._FileSource(p30)
print("probe:", s.label, "n24 =", s.n24, "size", s.width, s.height)
assert s.n24 == 120
L, C = 56, 22
a = s.frames_at(0, L, 96, 64)
b = s.frames_at(L - C, L, 96, 64)
va, vb = frame_values(a), frame_values(b)
print("slice 0 tail   :", va[-C:])
print("slice 1 head   :", vb[:C])
assert va[-C:] == vb[:C], "slice 1 does not start on the last C frames of slice 0"
# frozen tail past the end of the file
c = s.frames_at(100, 39, 96, 64)
vc = frame_values(c)
print("tail:", vc)
assert len(vc) == 39 and len(set(vc[19:])) <= 2, vc   # 100..119 real, then frozen
# audio (the file has no audio -> None, not an error)
assert s.audio_at(0, 56) is None

# 24 fps: exact index mapping
p24 = os.path.join(tmp, "src24.mp4"); make_video(p24, 130, 24)
s24 = m._FileSource(p24)
assert s24.n24 == 130
v = frame_values(s24.frames_at(102, 22, 128, 96))
print("24fps slice from 102:", v)
assert all(abs(x - (102 + j)) <= 2 for j, x in enumerate(v)), v   # +-2: yuv420p round trip

# comparison with the tensor source at 24 fps: same slice
ten = torch.tensor([[i / 255.0] for i in range(130)]).view(130, 1, 1, 1).expand(130, 96, 128, 3)
t24 = m._TensorSource(ten, 24.0)
assert frame_values(t24.frames_at(102, 22, 128, 96)) == list(range(102, 124))
print("ALL OK")

# --- start/end range ---------------------------------------------------
full = m._FileSource(p24)                       # 130 frames at 24 fps
part = m._FileSource(p24, 1.0, 4.0)             # frames 24..95
assert part.offset == 24 and part.n24 == 72, (part.offset, part.n24)
assert frame_values(part.frames_at(10, 22, 128, 96)) == frame_values(full.frames_at(34, 22, 128, 96))
tp = m._TensorSource(ten, 24.0, None, 1.0, 4.0)
assert tp.offset == 24 and tp.n24 == 72
assert frame_values(tp.frames_at(10, 22, 128, 96)) == list(range(34, 56))
tail = m._FileSource(p24, 0.0, 3.0); assert tail.n24 == 72 and tail.offset == 0
try:
    m._FileSource(p24, 5.3, 5.35); raise SystemExit("doveva fallire")
except ValueError as e:
    print("ok error:", e)
print("range ok")
