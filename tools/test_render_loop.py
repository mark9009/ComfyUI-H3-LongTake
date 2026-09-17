import os, sys, tempfile, subprocess, json
COMFY = os.environ.get("COMFY_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
sys.path.insert(0, COMFY)
os.chdir(COMFY)
import torch, importlib
import folder_paths, comfy.nested_tensor
# load the pack this test ships with (not whatever copy is installed under custom_nodes)
import importlib.util
_spec = importlib.util.spec_from_file_location("longtake_nodes", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "longtake_nodes.py"))
m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(m)

tmp = tempfile.mkdtemp()
folder_paths.set_output_directory(tmp)
W, H = 128, 96

class FakeVAE:
    audio_sample_rate = 48000
    def encode(self, x):
        if x.ndim == 3:  # audio [1, L, C]
            return torch.zeros(1, 32, 2, max(1, round(x.shape[1] / 48000 * 40)))
        t = x.shape[0]
        return torch.zeros(1, 24, m.video_latent_t(t) if t >= 5 else 1, x.shape[1] // 16, x.shape[2] // 16)
    def decode(self, z):
        t = m.pixel_frames(z.shape[2])
        # every decoded frame has value = "global" latent index stored in channel 0
        base = float(z[0, 0, -1, 0, 0])
        vals = torch.arange(t).float().add(base).div(1000.0)
        return vals.view(t, 1, 1, 1).expand(t, z.shape[3] * 16, z.shape[4] * 16, 3).clone()

class FakeClip:
    def tokenize(self, prompt, minimax_ref_items=None):
        return {"items": minimax_ref_items, "prompt": prompt}
    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1, 4, 8), {"tokens": tokens}]]

calls = []
def fake_sample(model, positive, latent, seed, sampler_name, scheduler, steps, noise_mask=None):
    video, audio = m._split_av(latent["samples"])
    extra = positive[0][1]
    calls.append({
        "seed": seed, "T": video.shape[2], "noise_mask": noise_mask, "latent_video": video.clone(),
        "keyframes": extra.get("minimax_keyframes"),
        "refs": [r["kind"] for r in extra["minimax_refs"]],
        "items": [it["type"] for it in extra["tokens"]["items"]] if "tokens" in extra else None,
        "cross": positive[0][0], "tags": extra.get("minimax_token_tags"),
    })
    v = torch.zeros_like(video); v[0, 0, -1, 0, 0] = seed  # marks the last token with the seed
    return comfy.nested_tensor.NestedTensor((v, torch.zeros_like(audio)))
m._sample = fake_sample

src = torch.rand(300, H, W, 3)  # 300 frames at 24 fps -> L=124, C=22 -> 3 clips
ref = torch.rand(1, 200, 160, 3)
node = m.H3LongTakeRender()
common = dict(model=object(), clip=FakeClip(), vae=FakeVAE(), audio_vae=FakeVAE(), source_file=m.NO_FILE,
              prompt="<Video 1> <Picture 1>", project_name="loop", width=W, height=H, clip_frames=124,
              context_frames="22", seed=100, steps=4, sampler_name="euler", scheduler="simple",
              redo_from_clip=0, max_clips=0, dry_run=False, ref_image_1=ref, source_video=src, source_fps=24.0,
              aspect="manual")

# dry run
r = node.render(mode="continue", **{**common, "dry_run": True})
print(r["result"][2]); assert not calls

# full run
r = node.render(mode="restart", **common)
print(r["result"][2])
assert [c["seed"] for c in calls] == [100, 101, 102]
assert calls[0]["keyframes"] is None and calls[1]["keyframes"] is not None
assert calls[1]["keyframes"][0]["latent"].shape[2] == 7 and len(calls[1]["keyframes"]) == 2
assert calls[0]["refs"] == ["image", "video"] and calls[0]["items"] == ["image", "video"]
assert calls[2]["T"] == m.video_latent_t(m.align_up(300 - 204))  # last clip: 96 -> 107
pdir = r["result"][1]
files = sorted(os.listdir(pdir)); print(files)
assert all(f"clip_{i:03d}.mp4" in files and f"clip_{i:03d}.latent.pt" in files for i in range(3))
# last clip: last_clip has 'new' frames, and the first 22 were trimmed (value starts at 22/1000)
last = r["result"][0]
plan = json.load(open(os.path.join(pdir, "plan.json")))["plan"]["clips"]
assert last.shape[0] == plan[2]["new"] == 300 - 226, last.shape
assert abs(float(last[0, 0, 0, 0]) - (22 + 102) / 1000.0) < 1e-6   # seed 102 + trim 22

# total mp4 duration = 300 frames
def nframes(path):
    out = subprocess.run([m._find_ffmpeg(), "-i", path, "-map", "0:v:0", "-c", "copy", "-f", "null", "-"],
                         capture_output=True).stderr.decode(errors="replace")
    import re
    return int(re.findall(r"frame=\s*(\d+)", out)[-1])
assert sum(nframes(os.path.join(pdir, f"clip_{i:03d}.mp4")) for i in range(3)) == 300

# continue: nothing to do
calls.clear()
r = node.render(mode="continue", **common)
assert not calls and "already cached: [0, 1, 2]" in r["result"][2]

# redo_from 1: regenerates 1 and 2 using clip 0's latent loaded from disk
r = node.render(mode="redo_from", **{**common, "redo_from_clip": 1})
assert [c["seed"] for c in calls] == [101, 102]
assert float(calls[0]["keyframes"][0]["latent"][0, 0, -1, 0, 0]) == 100.0  # tail of clip 0 (seed 100) read from disk
print("redo ok")

# redo_one 1: only clip 1, anchored at head (clip 0) and tail (clip 2)
calls.clear()
m0 = os.path.getmtime(os.path.join(pdir, "clip_000.mp4")); m2 = os.path.getmtime(os.path.join(pdir, "clip_002.mp4"))
r = node.render(mode="redo_one", **{**common, "redo_from_clip": 1, "seed": 500})
assert [c["seed"] for c in calls] == [501], calls
kfs = calls[0]["keyframes"]
assert len(kfs) == 4, kfs
assert kfs[0]["resolved_frame_index"] == 0 and kfs[0]["latent"].shape[2] == 7
assert kfs[2]["resolved_frame_index"] == 124 - 22 and kfs[2]["latent"].shape[2] == 7
assert kfs[3]["resolved_frame_index"] == 102.0 and kfs[3]["audio_latent"].shape[-1] == 37
assert os.path.getmtime(os.path.join(pdir, "clip_000.mp4")) == m0
assert os.path.getmtime(os.path.join(pdir, "clip_002.mp4")) == m2
assert nframes(os.path.join(pdir, "clip_001.mp4")) == 102
# redo_one on the last clip: head anchor only
calls.clear()
r = node.render(mode="redo_one", **{**common, "redo_from_clip": 2})
assert [c["seed"] for c in calls] == [102] and len(calls[0]["keyframes"]) == 2
print("redo_one ok")

# different plan -> clear error
try:
    node.render(mode="continue", **{**common, "width": W + 32}); raise SystemExit("should have failed")
except ValueError as e:
    print("ok plan error:", str(e).splitlines()[0])

# max_clips
calls.clear()
r = node.render(mode="restart", **{**common, "max_clips": 1, "project_name": "loop2"})
assert [c["seed"] for c in calls] == [100]

# source audio on -> audio item before the video, kind video_audio
calls.clear()
audio = {"waveform": torch.zeros(1, 2, 48000 * 13), "sample_rate": 48000}
r = node.render(mode="restart", **{**common, "max_clips": 1, "project_name": "loop3",
                                    "source_audio": audio, "use_source_audio": True})
assert calls[0]["items"] == ["image", "audio", "video"] and calls[0]["refs"] == ["image", "video_audio"]

# stitch
st = m.H3LongTakeStitch().stitch("loop", "final_loop")
print(st["result"][1]); assert nframes(st["result"][0]) == 300
print("ALL OK")

# stitch: project_dir from the socket + auto audio from the source saved in the project (with offset)
ff = m._find_ffmpeg()
srcav = os.path.join(tmp, "src_av.mp4")
subprocess.run([ff, "-y", "-v", "error", "-f", "lavfi", "-i", "testsrc=size=64x64:rate=24:duration=13",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=13", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest", srcav], check=True)
pj = json.load(open(os.path.join(pdir, "plan.json"))); pj["source_path"] = srcav; pj["source_offset_seconds"] = 1.0
json.dump(pj, open(os.path.join(pdir, "plan.json"), "w"))
st = m.H3LongTakeStitch().stitch("nome_ignorato", "final_auto", audio_file=m.AUDIO_AUTO, project_dir=pdir)
assert "audio from src_av.mp4 from 1.00s" in st["result"][1], st["result"][1]
info = subprocess.run([ff, "-i", st["result"][0]], capture_output=True).stderr.decode(errors="replace")
assert "Audio:" in info and nframes(st["result"][0]) == 300
st = m.H3LongTakeStitch().stitch("loop", "final_none", audio_file=m.AUDIO_NONE)
info = subprocess.run([ff, "-i", st["result"][0]], capture_output=True).stderr.decode(errors="replace")
assert "Audio:" not in info
st = m.H3LongTakeStitch().stitch("loop", "final_legacy", audio_file=m.NO_FILE)   # old workflows -> auto
assert "audio from src_av.mp4" in st["result"][1]
print("stitch auto ok")

# anchor_mode=inpaint: tail copied into the latent + mask, no keyframe
calls.clear()
r = node.render(mode="restart", **{**common, "project_name": "loop_inp", "anchor_mode": "inpaint"})
assert [c["seed"] for c in calls] == [100, 101, 102]
assert calls[0]["noise_mask"] is None and calls[0]["keyframes"] is None
c1 = calls[1]; vm, am = c1["noise_mask"].unbind()
assert c1["keyframes"] is None
assert vm.shape[2] == c1["T"] and float(vm[:, :, :7].max()) == 0.0 and float(vm[:, :, 7:].min()) == 1.0
assert float(am[..., :37].max()) == 0.0 and float(am[..., 37:].min()) == 1.0
assert float(c1["latent_video"][0, 0, 6, 0, 0]) == 100.0     # last token of clip 0's tail (seed 100) copied to token 6
# redo_one inpaint: mask at zero on head and tail
calls.clear()
pdir2 = os.path.join(tmp, "h3_longtake", "loop_inp")
r = node.render(mode="redo_one", **{**common, "project_name": "loop_inp", "anchor_mode": "inpaint", "redo_from_clip": 1, "seed": 700})
vm, am = calls[0]["noise_mask"].unbind()
assert float(vm[:, :, :7].max()) == 0.0 and float(vm[:, :, -7:].max()) == 0.0 and float(vm[:, :, 7:-7].min()) == 1.0
assert float(calls[0]["latent_video"][0, 0, -1, 0, 0]) == 0.0  # head of clip 2: token 6 of its head = 0 (the marker is in the tail)
print("inpaint ok")

# anchor_mode=keyframe_before: no overlap, keyframes at negative indices, no trim
calls.clear()
r = node.render(mode="restart", **{**common, "project_name": "loop_before", "anchor_mode": "keyframe_before"})
pb = json.load(open(os.path.join(tmp, "h3_longtake", "loop_before", "plan.json")))["plan"]
assert [c["src_start"] for c in pb["clips"]] == [0, 124, 248] and all(c["ctx"] == 0 for c in pb["clips"]), pb["clips"]
assert sum(c["new"] for c in pb["clips"]) == 300
k = calls[1]["keyframes"]
assert k[0]["resolved_frame_index"] == -22 and k[0]["latent"].shape[2] == 7
assert abs(k[1]["resolved_frame_index"] - (0 - 37 / m.FRAME_RESCALE)) < 1e-6
assert nframes(os.path.join(tmp, "h3_longtake", "loop_before", "clip_001.mp4")) == 124
# redo_one in before: tail anchored at target_frames
calls.clear()
r = node.render(mode="redo_one", **{**common, "project_name": "loop_before", "anchor_mode": "keyframe_before", "redo_from_clip": 1, "seed": 900})
k = calls[0]["keyframes"]; assert len(k) == 4 and k[2]["resolved_frame_index"] == 124 and k[3]["resolved_frame_index"] == 124.0
print("before ok")

# anchor_mode=none: no keyframe, no mask, no overlap
calls.clear()
r = node.render(mode="restart", **{**common, "project_name": "loop_none", "anchor_mode": "none"})
assert all(c["keyframes"] is None and c["noise_mask"] is None for c in calls)
pn = json.load(open(os.path.join(tmp, "h3_longtake", "loop_none", "plan.json")))["plan"]
assert [c["src_start"] for c in pn["clips"]] == [0, 124, 248]
print("none ok")

# source_role=guide: no <Video 1> among the references, guide latent at frame 0 along
# the whole clip, empty prompt -> template, keyframe context appended after the guide
calls.clear()
r = node.render(mode="restart", **{**common, "project_name": "loop_guide", "source_role": "guide",
                                   "prompt": "", "context_frames": "5", "anchor_mode": "keyframe"})
assert "the guide template" in r["result"][2], r["result"][2]
assert calls[0]["items"] == ["image"] and calls[0]["refs"] == ["image"]
k0 = calls[0]["keyframes"]; assert len(k0) == 1 and k0[0]["resolved_frame_index"] == 0
assert k0[0]["latent"].shape[2] == calls[0]["T"] == m.video_latent_t(124)
k1 = calls[1]["keyframes"]; assert len(k1) == 3 and k1[0]["latent"].shape[2] == calls[1]["T"]
assert k1[1]["resolved_frame_index"] == 0 and k1[1]["latent"].shape[2] == 2      # 5-frame tail = 2 tokens
assert k1[0]["latent"].shape[3] % 2 == 0 and k1[0]["latent"].shape[4] % 2 == 0    # pad 2x2
assert calls[2]["keyframes"][0]["latent"].shape[2] == calls[2]["T"]              # last clip extended: guide extended the same way
pg = json.load(open(os.path.join(tmp, "h3_longtake", "loop_guide", "plan.json")))
assert pg["signature"]["source_role"] == "guide"
assert os.path.isfile(os.path.join(tmp, "h3_longtake", "loop_guide", "clip_001_ref.mp4"))
# guide + inpaint: mask present, a single keyframe (the guide)
calls.clear()
r = node.render(mode="restart", **{**common, "project_name": "loop_guide_inp", "source_role": "guide",
                                   "prompt": "style_transfer: Re-render this video with the style of <Picture 1>: neon.",
                                   "context_frames": "5", "anchor_mode": "inpaint"})
assert calls[1]["noise_mask"] is not None and len(calls[1]["keyframes"]) == 1
# guide+reference: both
calls.clear()
r = node.render(mode="restart", **{**common, "project_name": "loop_guide_ref", "source_role": "guide+reference",
                                   "prompt": "style_transfer: <Video 1> <Picture 1>", "context_frames": "5"})
assert calls[0]["items"] == ["image", "video"] and len(calls[0]["keyframes"]) == 1
# <Video 1> in the prompt with pure guide -> clear error
try:
    node.render(mode="restart", **{**common, "project_name": "loop_guide_err", "source_role": "guide",
                                   "prompt": "<Video 1> style"})
    raise AssertionError("error expected")
except ValueError as exc:
    assert "<Video 1>" in str(exc)
print("guide ok")

# seam_match: unit test of the function (exposure + hue jump at the seam)
imgs = torch.full((40, 16, 16, 3), 0.5)
imgs[5:] = torch.tensor([0.7, 0.6, 0.4])   # after the context: brighter and warmer
out = m.seam_match_fn(imgs, 5, "color")
assert torch.allclose(out[:5], imgs[:5])                                  # context untouched
assert abs(float(m._luminance(out[5]).mean()) - 0.5) < 0.02, float(m._luminance(out[5]).mean())
assert float((out[5] - out[5].mean()).abs().max()) < 0.06                 # nearly neutral hue (0.08 per-channel limit)
assert torch.allclose(out[29], imgs[29])                                  # end of the fade: original
mid = float(m._luminance(out[17]).mean()); assert 0.52 < mid < 0.6, mid   # halfway: partial correction
outl = m.seam_match_fn(imgs, 5, "luminance")
assert abs(float(m._luminance(outl[5]).mean()) - 0.5) < 0.02 and float((outl[5] - outl[5].mean()).abs().max()) > 0.05
assert m.seam_match_fn(imgs, 0, "color") is imgs and m.seam_match_fn(imgs, 5, "off") is imgs
# external reference (real tail of the previous clip): the head is pulled towards it
ref = torch.full((4, 16, 16, 3), 0.55)   # within the 0.25 EV limit
out2 = m.seam_match_fn(imgs, 5, "color", ref=ref)
assert abs(float(m._luminance(out2[5]).mean()) - 0.55) < 0.03 and torch.allclose(out2[:5], imgs[:5])
# reading the tail from a cached chunk's mp4
tail = m._tail_frames_from_mp4(os.path.join(pdir, "clip_000.mp4"), 124, 4)
assert tail.shape == (4, H, W, 3)
# in the node: runs without errors (also resuming from the cache, where the tail comes from the mp4)
calls.clear()
r = node.render(mode="restart", **{**common, "project_name": "loop_seam", "context_frames": "5", "seam_match": "color", "max_clips": 1})
r = node.render(mode="continue", **{**common, "project_name": "loop_seam", "context_frames": "5", "seam_match": "color"})
assert "seam_match=color" in r["result"][2] and [c["seed"] for c in calls] == [100, 101, 102]
print("seam ok")


# workflow saved into the project and embedded in the mp4s (workflow/prompt tags like SaveVideo)
fake_wf = {"nodes": [{"id": 7, "type": "H3LongTakeRender"}], "note": "a=b;c#d\\e"}
fake_api = {"7": {"class_type": "H3LongTakeRender", "inputs": {"prompt": "x", "model": ["2", 0], "project_name": "loop_meta"}},
            "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "style.safetensors", "strength_model": 1.0, "model": ["1", 0]}},
            "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "h3.safetensors"}}}
calls.clear()
r = node.render(mode="restart", **{**common, "project_name": "loop_meta", "max_clips": 1,
                                   "api_prompt": fake_api, "extra_pnginfo": {"workflow": fake_wf}})
pm = r["result"][1]
assert json.load(open(os.path.join(pm, "workflow.json"), encoding="utf-8")) == fake_wf
st_ = json.load(open(os.path.join(pm, "settings.json"), encoding="utf-8"))
assert st_["7"]["model"]["node"] == "LoraLoaderModelOnly" and st_["7"]["model"]["model"]["unet_name"] == "h3.safetensors"
def mp4_tags(path):
    out = subprocess.run([m._find_ffmpeg(), "-i", path, "-f", "ffmetadata", "-"], capture_output=True).stdout.decode("utf-8", "replace")
    tags = {}
    for line in out.splitlines():
        if "=" in line and not line.startswith(";"):
            k, v = line.split("=", 1); tags[k] = v
    return tags
t = mp4_tags(os.path.join(pm, "preview.mp4"))
assert "workflow" in t and "prompt" in t, t.keys()
# ffmpeg re-exports with the same escapes: strip them and compare the JSON
import re
unesc = lambda v: re.sub(r"\\(.)", r"\1", v)
assert json.loads(unesc(t["workflow"])) == fake_wf, unesc(t["workflow"])[:200]
assert json.loads(unesc(t["prompt"]))["7"]["inputs"]["project_name"] == "loop_meta"
# stitch without hidden inputs: takes the files saved in the project
st = m.H3LongTakeStitch().stitch("loop_meta", "loop_meta_final", audio_file=m.AUDIO_NONE)
assert "workflow embedded" in st["result"][1]
assert json.loads(unesc(mp4_tags(st["result"][0])["workflow"])) == fake_wf
# with audio (output options -map/-t): the metadata must stay among the inputs
pj = json.load(open(os.path.join(pm, "plan.json"))); pj["source_path"] = srcav; pj["source_offset_seconds"] = 0.0
json.dump(pj, open(os.path.join(pm, "plan.json"), "w"))
st = m.H3LongTakeStitch().stitch("loop_meta", "loop_meta_final_audio", audio_file=m.AUDIO_AUTO)
info = subprocess.run([ff, "-i", st["result"][0]], capture_output=True).stderr.decode(errors="replace")
assert "Audio:" in info and json.loads(unesc(mp4_tags(st["result"][0])["workflow"])) == fake_wf
# immutable copy of the first run: a continue with another graph updates workflow.json but not workflow_first.json
fake_wf2 = dict(fake_wf, note="secondo run")
r = node.render(mode="continue", **{**common, "project_name": "loop_meta", "max_clips": 1,
                                    "api_prompt": fake_api, "extra_pnginfo": {"workflow": fake_wf2}})
assert json.load(open(os.path.join(pm, "workflow.json"), encoding="utf-8")) == fake_wf2
assert json.load(open(os.path.join(pm, "workflow_first.json"), encoding="utf-8")) == fake_wf
assert os.path.isfile(os.path.join(pm, "settings_first.json"))
print("meta ok")

# --- Stitch before any render (dry run left unmuted): report, no exception -------------------------
st = m.H3LongTakeStitch().stitch("never_rendered", "never_final", audio_file=m.AUDIO_NONE)
assert st["result"][0] == "" and "nothing to stitch" in st["result"][1], st
print("stitch-empty ok")

# --- Viggle-Animate: text_cond instead of clip, refs video->image, prompt ignored ---
calls.clear()
tc = {"prompt_embeds": torch.zeros(1, 362, 8), "text_token_tags": torch.arange(362)}
vig = {**common, "project_name": "loop_viggle", "max_clips": 2, "context_frames": "5", "prompt": "any text",
       "ref_image_2": ref, "use_source_audio": True, "source_audio": {"waveform": torch.zeros(1, 2, 48000 * 13), "sample_rate": 48000},
       "text_cond": tc, "steps": 3}
del vig["clip"]
r = node.render(mode="restart", **vig)
rep = r["result"][2]
assert "Viggle-Animate" in rep and "ref_image_2/3 ignored" in rep and "use_source_audio ignored" in rep, rep
assert len(calls) == 2 and calls[0]["refs"] == ["video", "image"] and calls[0]["items"] is None
assert calls[0]["cross"] is tc["prompt_embeds"] and calls[0]["tags"] is tc["text_token_tags"]
assert calls[1]["keyframes"] is not None and calls[1]["keyframes"][0]["latent"].shape[2] == 2  # 5-frame anchor
plan = json.load(open(os.path.join(r["result"][1], "plan.json")))
assert plan["signature"]["engine"] == "viggle" and plan["prompt_hash"] != ""
for bad, msg in (({**vig, "source_role": "guide"}, "source_role=reference"),
                 ({**vig, "ref_image_1": None}, "ref_image_1"),
                 ({**common, "clip": None}, "connect clip")):
    try:
        node.render(mode="restart", **bad); raise AssertionError("error expected")
    except ValueError as e:
        assert msg in str(e), str(e)
# without text_cond everything as before (clip used, image->video order)
calls.clear()
node.render(mode="restart", **{**common, "project_name": "loop_viggle2", "max_clips": 1})
assert calls[0]["refs"] == ["image", "video"] and calls[0]["items"] == ["image", "video"]
print("viggle ok")
