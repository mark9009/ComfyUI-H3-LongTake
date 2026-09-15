"""Driver: queues H3 LongTake tests through the ComfyUI API, waits, measures, appends to results.md.
Set COMFY_DIR (ComfyUI folder), COMFY_PYTHON, LT_SOURCE and LT_IMAGE for your install."""
import json, os, subprocess, sys, time, urllib.request, uuid

API = "http://127.0.0.1:8188"
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results.md")
COMFY = os.environ.get("COMFY_DIR", os.path.abspath(os.path.join(HERE, "..", "..", "..")))
PY = os.environ.get("COMFY_PYTHON", sys.executable)
CLIP_NAME = os.environ.get("LT_CLIP", "qwen3vl_32b_minimax_h3.safetensors")
SOURCE = os.environ.get("LT_SOURCE", "source.mp4")
IMAGE = os.environ.get("LT_IMAGE", "identity.png")

wf = json.load(open(os.path.join(HERE, "..", "workflow", "H3_LongTake_example.json"), encoding="utf-8"))
PROMPT = [n for n in wf["nodes"] if n["type"] == "PrimitiveStringMultiline"][0]["widgets_values"][0]

TESTS = [
    # name, anchor_mode, context, audio_context, clip_frames
    ("t3_none",        "none",            "22", True,  124),
    ("t0_keyframe",    "keyframe",        "22", True,  124),
    ("t1_inpaint",     "inpaint",         "22", True,  124),
    ("t2_before",      "keyframe_before", "22", True,  124),
    ("t4_keyframe_c5", "keyframe",        "5",  True,  124),
    ("t5_inpaint_c5",  "inpaint",         "5",  True,  124),
]


def api(path, data=None):
    req = urllib.request.Request(API + path, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def wait_server():
    for _ in range(120):
        try:
            api("/object_info/H3LongTakeRender")
            return True
        except Exception:
            time.sleep(5)
    return False


def build(name, anchor, ctx, audio_ctx, clip_frames):
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_ref2va_pruned_fp8_scaled.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", "strength_model": 0.95}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP_NAME, "type": "minimax", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "6": {"class_type": "LoadImage", "inputs": {"image": IMAGE, "upload": "image"}},
        "7": {"class_type": "H3LongTakeRender", "inputs": {
            "model": ["2", 0], "clip": ["3", 0], "vae": ["4", 0], "audio_vae": ["5", 0], "ref_image_1": ["6", 0],
            "source_file": SOURCE, "prompt": PROMPT, "project_name": name,
            "width": 896, "height": 576, "clip_frames": clip_frames, "context_frames": ctx,
            "seed": 0, "steps": 4, "sampler_name": "euler", "scheduler": "simple",
            "mode": "restart", "redo_from_clip": 0, "max_clips": 2, "dry_run": False,
            "source_fps": 24.0, "use_source_audio": False, "ref_image_size": "match", "ref_video_size": "match",
            "audio_context": audio_ctx, "chunk_crf": 10, "aspect": "source", "megapixels": 0.5,
            "start_seconds": 0.0, "end_seconds": 0.0, "anchor_mode": anchor,
        }},
    }


def run_one(name, anchor, ctx, audio_ctx, clip_frames):
    pid = api("/prompt", {"prompt": build(name, anchor, ctx, audio_ctx, clip_frames), "client_id": uuid.uuid4().hex})["prompt_id"]
    t0 = time.time()
    while True:
        time.sleep(20)
        h = api(f"/history/{pid}")
        if pid in h:
            st = h[pid].get("status", {})
            ok = st.get("status_str") == "success"
            err = ""
            if not ok:
                for m in st.get("messages", []):
                    if m[0] == "execution_error":
                        err = m[1].get("exception_message", "")[:300]
            return ok, err, time.time() - t0
        if time.time() - t0 > 3600:
            return False, "timeout", time.time() - t0


def measure(name):
    proj = os.path.join(COMFY, "output", "h3_longtake", name)
    src = os.path.join(COMFY, "input", SOURCE)
    r = subprocess.run([PY, os.path.join(HERE, "align_check.py"), proj, src], capture_output=True, text=True)
    return r.stdout + r.stderr[-500:]


def log(text):
    with open(RESULTS, "a", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text, flush=True)


if __name__ == "__main__":
    only = sys.argv[1:]  # optional: names of the tests to run
    log(f"\n## Session {time.strftime('%Y-%m-%d %H:%M')}  source={SOURCE}  seed=0  2 clips\n")
    if not wait_server():
        log("server unreachable"); sys.exit(1)
    for name, anchor, ctx, audio_ctx, cf in TESTS:
        if only and name not in only:
            continue
        log(f"### {name}: anchor={anchor} context={ctx} audio_context={audio_ctx} clip_frames={cf}")
        ok, err, dt = run_one(name, anchor, ctx, audio_ctx, cf)
        log(f"result: {'ok' if ok else 'ERROR ' + err}  time {dt / 60:.1f} min")
        if ok:
            log("```\n" + measure(name).strip() + "\n```")
    log("END")
