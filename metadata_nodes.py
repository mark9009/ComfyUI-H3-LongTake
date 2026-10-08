"""Two small helpers for Civitai metadata on the finished video, used by the author's workflows with
Image Saver (Image Saver Metadata / Image Saver Video Metadata):

  * H3 Video File: the path of a saved mp4 (e.g. the Stitch's `path`) as VHS_FILENAMES, plus its real size;
  * H3 LoRA Tags: the <lora:name:strength> tags of the LoRAs applied to the connected MODEL and the base model
    name, read from the graph instead of typed by hand.
"""

import os

import av
from comfy_execution.graph_utils import ExecutionBlocker


class H3VideoFile:
    """The path of an already saved mp4 (e.g. H3 LongTake Stitch's 'path') as VHS_FILENAMES, to connect it to
    Image Saver Video Metadata; plus its real resolution for the metadata."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"path": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ("VHS_FILENAMES", "INT", "INT")
    RETURN_NAMES = ("filenames", "width", "height")
    FUNCTION = "wrap"
    CATEGORY = "H3 LongTake"
    DESCRIPTION = "Path of a saved video -> VHS_FILENAMES (for Image Saver Video Metadata), width and height."

    def wrap(self, path):
        path = (path or "").strip()
        if not path:
            # Stitch in dry run: nothing to tag, the metadata branch stops without an error
            return (ExecutionBlocker(None),) * 3
        if not path.lower().endswith(".mp4") or not os.path.isfile(path):
            raise RuntimeError("No mp4 video to tag: %r" % path)
        with av.open(path) as c:
            stream = c.streams.video[0]
            width, height = stream.width, stream.height
        return ((True, [path]), width, height)


class H3LoraTags:
    """<lora:name:strength> tags of the LoRAs applied to the connected MODEL and the base model name, read from the
    graph: for Image Saver Metadata, instead of typing them by hand."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL", {"lazy": True, "tooltip": "The MODEL that goes to the Render (after the "
                                                 "LoRAs). It is not loaded: it only serves to find the chain in the graph."})},
                "hidden": {"prompt": "PROMPT", "unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("lora_tags", "model_name")
    FUNCTION = "read"
    CATEGORY = "H3 LongTake"
    DESCRIPTION = "LoRAs applied to the connected MODEL as <lora:name:strength> tags, and the base model name (for Image Saver Metadata)."

    def check_lazy_status(self, **kwargs):
        return []

    def read(self, prompt, unique_id, model=None):
        tags, model_name, nid, seen = [], "", prompt[str(unique_id)]["inputs"]["model"][0], set()
        while isinstance(nid, str) and nid in prompt and nid not in seen:
            seen.add(nid)
            inputs = prompt[nid]["inputs"]
            found = []
            if isinstance(inputs.get("lora_name"), str):
                found.append((inputs["lora_name"], inputs.get("strength_model", inputs.get("strength", 1.0))))
            for v in inputs.values():  # Power Lora Loader (rgthree): lora_1 = {on, lora, strength}
                if isinstance(v, dict) and v.get("lora") not in (None, "", "None") and v.get("on", True):
                    found.append((v["lora"], v.get("strength", 1.0)))
            tags = ["<lora:%s:%g>" % (os.path.splitext(os.path.basename(n))[0], s) for n, s in found if s] + tags
            for key in ("unet_name", "ckpt_name"):
                if isinstance(inputs.get(key), str):
                    model_name = os.path.splitext(os.path.basename(inputs[key]))[0]
            link = inputs.get("model")
            nid = str(link[0]) if isinstance(link, list) else None
        return (" ".join(tags), model_name)


NODE_CLASS_MAPPINGS = {"H3VideoFile": H3VideoFile, "H3LoraTags": H3LoraTags}
NODE_DISPLAY_NAME_MAPPINGS = {"H3VideoFile": "H3 Video File (path -> VHS filenames)",
                              "H3LoraTags": "H3 LoRA Tags (from the graph, for Image Saver)"}
