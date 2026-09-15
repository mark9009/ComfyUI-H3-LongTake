"""Driver: source_role=guide tests with the StyleTransfer LoRA (NRDX) through the ComfyUI API.
Same source/seed as the Ref2VA battery (run_tests.py) so the measurements can be compared.
Set COMFY_DIR (ComfyUI folder), COMFY_PYTHON, LT_SOURCE and LT_STYLE_IMAGE for your install."""
import json, os, subprocess, sys, time, urllib.request, uuid

API = "http://127.0.0.1:8188"
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results.md")
COMFY = os.environ.get("COMFY_DIR", os.path.abspath(os.path.join(HERE, "..", "..", "..")))
PY = os.environ.get("COMFY_PYTHON", sys.executable)
CLIP_NAME = os.environ.get("LT_CLIP", "qwen3vl_32b_minimax_h3.safetensors")
SOURCE = os.environ.get("LT_SOURCE", "source.mp4")
STYLE_IMAGE = os.environ.get("LT_STYLE_IMAGE", "style_popart_halftone.png")
STYLE_LORA = "minimax-h3\\minimax_h3_style_transfer_v1.0_r64.safetensors"  # Windows separator, as in the combo
PROMPT_STYLE = ("style_transfer: Re-render this video with the style of <Picture 1>: "
                "bold graphic style, vibrant flat colours, clean black outlines, flat shading, halftone dots.")
PROMPT_RETEX = "retexture: change the outfit to red glossy latex, keeping face, hair, background and motion unchanged."

PROMPT_GTA = ("style_transfer: Re-render this video in the style of GTA V loading screen artwork: "
              "cel-shaded comic illustration, thick dark outlines, flat saturated colours, hard-edged shadows, glossy highlights.")

TESTS = [
    # name, source_role, anchor_mode, context, steps, prompt, lora_strength
    ("s1_guide_none",     "guide",           "none",     "5", 4, PROMPT_STYLE, 1.0),
    ("s2_guide_inpaint",  "guide",           "inpaint",  "5", 4, PROMPT_STYLE, 1.0),
    ("s3_guide_keyframe", "guide",           "keyframe", "5", 4, PROMPT_STYLE, 1.0),
    ("s4_guide_ref",      "guide+reference", "inpaint",  "5", 4, PROMPT_STYLE, 1.0),
    ("s5_guide_8step",    "guide",           "inpaint",  "5", 8, PROMPT_STYLE, 1.0),
    ("s6_retexture",      "guide_retexture", "inpaint",  "5", 4, PROMPT_RETEX, 1.0),
    ("s7_gta",            "guide",           "keyframe", "5", 4, PROMPT_GTA,   1.0),   # text only, no <Picture 1>
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


def build(name, role, anchor, ctx, steps, prompt, strength):
    inputs = {
        "model": ["8", 0], "clip": ["3", 0], "vae": ["4", 0], "audio_vae": ["5", 0],
        "source_file": SOURCE, "prompt": prompt, "project_name": name,
        "width": 896, "height": 576, "clip_frames": 124, "context_frames": ctx,
        "seed": 0, "steps": steps, "sampler_name": "euler", "scheduler": "simple",
        "mode": "restart", "redo_from_clip": 0, "max_clips": 2, "dry_run": False,
        "source_fps": 24.0, "use_source_audio": False, "ref_image_size": "match", "ref_video_size": "match",
        "audio_context": True, "chunk_crf": 10, "aspect": "source", "megapixels": 0.5,
        "start_seconds": 0.0, "end_seconds": 0.0, "anchor_mode": anchor, "source_role": role, "seam_match": "color",
    }
    if "<Picture 1>" in prompt:
        inputs["ref_image_1"] = ["6", 0]
    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_ref2va_pruned_fp8_scaled.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": "minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", "strength_model": 0.95}},
        "8": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["2", 0], "lora_name": STYLE_LORA, "strength_model": strength}},
        "3": {"class_type": "CLIPLoader", "inputs": {"clip_name": CLIP_NAME, "type": "minimax", "device": "default"}},
        "4": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_fp16.safetensors"}},
        "5": {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}},
        "6": {"class_type": "LoadImage", "inputs": {"image": STYLE_IMAGE, "upload": "image"}},
        "7": {"class_type": "H3LongTakeRender", "inputs": inputs},
    }


def run_one(*args):
    pid = api("/prompt", {"prompt": build(*args), "client_id": uuid.uuid4().hex})["prompt_id"]
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
    out = ""
    for script, args in (("align_check.py", [proj, src]), ("junction.py", [name]), ("color_check.py", [proj])):
        r = subprocess.run([PY, os.path.join(HERE, script)] + args, capture_output=True, text=True)
        out += f"-- {script}\n" + r.stdout + r.stderr[-300:]
    return out


def log(text):
    with open(RESULTS, "a", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(text, flush=True)


if __name__ == "__main__":
    only = sys.argv[1:]
    log(f"\n## STYLE session {time.strftime('%Y-%m-%d %H:%M')}  source={SOURCE}  style={STYLE_IMAGE}  seed=0  2 clips x 124\n")
    if not wait_server():
        log("server unreachable"); sys.exit(1)
    for t in TESTS:
        name = t[0]
        if only and name not in only:
            continue
        log(f"### {name}: role={t[1]} anchor={t[2]} context={t[3]} steps={t[4]} lora={t[6]}")
        ok, err, dt = run_one(*t)
        log(f"result: {'ok' if ok else 'ERROR ' + err}  time {dt / 60:.1f} min")
        if ok:
            log("```\n" + measure(name).strip() + "\n```")
    log("END")
