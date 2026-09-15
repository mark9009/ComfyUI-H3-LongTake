import os, sys, tempfile, json
COMFY = os.environ.get("COMFY_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, COMFY)
os.chdir(COMFY)
import torch
import importlib
m = importlib.import_module("custom_nodes.ComfyUI-H3-LongTake.longtake_nodes")
print("nodes:", list(m.NODE_CLASS_MAPPINGS))

# --- plan ---------------------------------------------------------------
def check_plan(n24, L, C, max_clips=0):
    p = m.build_plan(n24, L, C, max_clips)
    clips = p["clips"]
    total_new = sum(c["new"] for c in clips)
    # invariants
    for k, c in enumerate(clips):
        assert c["length"] % 17 == 5, c
        assert c["length"] > c["ctx"], c
        if k > 0:
            prev = clips[k - 1]
            # the first C frames of this clip = the last C frames of the previous one (in the source)
            assert c["src_start"] == prev["src_start"] + prev["length"] - C, (prev, c)
            assert c["ctx"] == C
        else:
            assert c["src_start"] == 0 and c["ctx"] == 0
        assert c["src_start"] + c["ctx"] + c["new"] <= n24
    if not max_clips:
        assert total_new == n24, (total_new, n24)
    # output continuity: the new frames are contiguous
    cursor = 0
    for c in clips:
        assert c["src_start"] + c["ctx"] == cursor, (c, cursor)
        cursor += c["new"]
    return p

for n24, L, C in [(360, 124, 22), (2880, 243, 22), (124, 124, 22), (130, 124, 22), (48, 124, 22),
                  (1000, 124, 5), (1000, 124, 39), (1000, 124, 56), (500, 22, 5), (23, 124, 22)]:
    p = check_plan(n24, L, C)
    print(f"n24={n24:5d} L={L:3d} C={C:2d} -> {len(p['clips']):3d} clips, "
          f"last {p['clips'][-1]}")
print(m.plan_report(check_plan(360, 124, 22), 450, 30.0))
p = check_plan(2880, 243, 22, max_clips=3); assert len(p["clips"]) == 3
try:
    m.build_plan(100, 22, 22); raise SystemExit("doveva fallire")
except ValueError as e:
    print("ok error:", e)

# --- motion context: phase and padding --------------------------------------
for frames, C in [(124, 22), (243, 22), (124, 5), (124, 39), (124, 56), (39, 22)]:
    T = m.video_latent_t(frames)
    prev_v = torch.randn(1, 24, T, 36, 56)
    prev_a = torch.randn(1, 32, 2, m.audio_t_for_frames(frames))
    tgt = torch.zeros(1, 24, T, 36, 56)
    kfs = m.motion_context_keyframes(prev_v, prev_a, C, tgt, True)
    steps = m.latent_steps_for_frames(C)
    assert kfs[0]["latent"].shape[2] == steps and m.pixel_frames(steps) == C
    assert torch.equal(kfs[0]["latent"], prev_v[:, :, -steps:])
    rt = kfs[1]["audio_latent"].shape[-1]
    assert rt == round(C / 24 * 40)
    end = kfs[1]["resolved_frame_index"] + rt / m.FRAME_RESCALE
    assert abs(end - C) < 1e-6
    print(f"frames={frames} C={C}: T={T} steps={steps} rt={rt} audio_start={kfs[1]['resolved_frame_index']:.3f}")
# odd H/W -> pad
tgt_odd = torch.zeros(1, 24, 7, 33, 49)
kfs = m.motion_context_keyframes(torch.randn(1, 24, 37, 33, 49), None, 22, tgt_odd, False)
assert kfs[0]["latent"].shape[3:] == (34, 50)
print("pad ok")

# --- source slice / resampling ------------------------------------
src = torch.arange(90).float().view(90, 1, 1, 1).expand(90, 4, 4, 3)
sl = m._take_source_frames(src, 30.0, 0, 22)     # 30 fps -> 24 fps
assert sl.shape[0] == 22 and sl[0, 0, 0, 0] == 0 and sl[-1, 0, 0, 0] == round(21 * 30 / 24)
sl = m._take_source_frames(src, 24.0, 80, 39)    # past the end: frozen
assert sl[-1, 0, 0, 0] == 89 and sl[9, 0, 0, 0] == 89
print("slice ok, n24(450@30)=", m.source_frames_at_24fps(450, 30.0))

# --- mp4 + stitch --------------------------------------------------------
import folder_paths
tmp = tempfile.mkdtemp()
folder_paths.set_output_directory(tmp)
pdir = m._project_dir("test proj")
for i in range(3):
    imgs = torch.rand(10, 64, 96, 3)
    _, mp4 = m._clip_paths(pdir, i)
    m._write_mp4(mp4, imgs, 10)
    assert os.path.getsize(mp4) > 0
m._save_plan(pdir, {"plan": {"clips": [{}, {}, {}]}})
audio = {"waveform": torch.zeros(1, 2, 48000 * 2), "sample_rate": 48000}
res = m.H3LongTakeStitch().stitch("test proj", "final", source_audio=audio)
print(res["result"][1])
assert os.path.isfile(res["result"][0])
# frame count check with ffmpeg
import subprocess
ff = m._find_ffmpeg()
out = subprocess.run([ff, "-i", res["result"][0]], capture_output=True).stderr.decode(errors="replace")
print([l.strip() for l in out.splitlines() if "Stream" in l or "Duration" in l])
print("delete from 1:", m._delete_clips_from(pdir, 1))
print("ALL OK")
