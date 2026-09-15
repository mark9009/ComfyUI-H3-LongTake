"""Re-queues via API the last H3LongTakeRender run in ComfyUI's history with changes.
Usage: rerun_history.py project_name=my_project mode=restart image=style_swatch.png prompt="..." """
import json, sys, time, urllib.request, uuid

API = "http://127.0.0.1:8188"


def api(path, data=None):
    req = urllib.request.Request(API + path, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


changes = dict(a.split("=", 1) for a in sys.argv[1:])
hist = api("/history?max_items=20")
entry = None
for pid, e in hist.items():
    pr = e["prompt"][2]
    if any(n["class_type"] == "H3LongTakeRender" for n in pr.values()):
        entry = pr  # the last one in insertion order
prompt = json.loads(json.dumps(entry))
for nid, n in prompt.items():
    if n["class_type"] == "H3LongTakeRender":
        i = n["inputs"]
        for k in ("project_name", "mode", "max_clips", "seed", "megapixels", "seam_match", "anchor_mode", "context_frames"):
            if k in changes:
                i[k] = type(i[k])(changes[k]) if not isinstance(i[k], bool) else changes[k].lower() == "true"
        if "prompt" in changes:
            src = i.get("prompt_text")
            if isinstance(src, list):
                prompt[src[0]]["inputs"]["value"] = changes["prompt"]
            else:
                i["prompt"] = changes["prompt"]
        if changes.get("image") == "none":
            i.pop("ref_image_1", None)          # text-only style: no <Picture 1>
        elif "image" in changes:
            src = i.get("ref_image_1")
            prompt[src[0]]["inputs"]["image"] = changes["image"]
        print("inputs:", {k: v for k, v in i.items() if not isinstance(v, list)})
pid = api("/prompt", {"prompt": prompt, "client_id": uuid.uuid4().hex})["prompt_id"]
print("prompt_id", pid)
t0 = time.time()
while True:
    time.sleep(20)
    h = api(f"/history/{pid}")
    if pid in h:
        st = h[pid]["status"]
        print("result:", st["status_str"], f"{(time.time() - t0) / 60:.1f} min")
        for m in st.get("messages", []):
            if m[0] == "execution_error":
                print("ERR", m[1].get("exception_message", "")[:400])
        break
