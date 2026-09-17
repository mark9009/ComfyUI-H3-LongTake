"""
H3 LongTake - long videos with MiniMax H3 Ref2VA, rendered clip by clip as if
they were a single generation.

Idea:
  * the source video is sliced into windows of L frames at 24 fps with a stride
    of L-C, where C = context_frames;
  * every clip receives ITS OWN slice of the source as <Video 1> (or as a guide
    latent) and, as an anchor (H3 keyframe at frame 0), the last C latent frames
    of the previous clip. Anchor and reference point to the same instant of the
    source, so sync is never lost;
  * every clip lands on disk (latent + mp4 already trimmed of the C overlapping
    frames), so VRAM only ever holds one clip and any clip can be resumed or
    redone.

Depends only on ComfyUI core >= 0.34 (internal keyframes + refs coexist natively
in comfy/model_base.py and comfy/ldm/minimax/model.py).
"""

import gc
import hashlib
import json
import logging
import math
import os
import shutil
import subprocess

import torch

import comfy.model_management
import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import comfy.utils
import folder_paths
import latent_preview
import node_helpers
from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE

_LOG = logging.getLogger("h3_longtake")

FPS = 24
AUDIO_HZ = 40.0
CANVAS = 32
REF_IMAGE_SHORT_EDGE = 2048
REF_VIDEO_SHORT_EDGE = 768
REF_VIDEO_MAX_PIXELS = 768 * 1344
MIN_REF_AUDIO_SECONDS = 2.0
CONTEXT_CHOICES = ["5", "22", "39", "56"]

# Role of the source slice in every clip.
#   reference        : <Video 1> among the Ref2VA references (motion suggestion)
#   guide            : guide latent = the slice encoded and anchored at frame 0 for the
#                      whole clip (frame-accurate control, like MiniMaxH3AddGuide with a
#                      video); meant for NRDX's StyleTransfer/retexture LoRA, which
#                      re-renders the guide in the style of <Picture 1> or of the text
#   guide_retexture  : like guide, with the text-only "retexture:" template (material/colour change)
#   guide+reference  : both
SOURCE_ROLES = ["reference", "guide", "guide_retexture", "guide+reference"]
SEAM_MODES = ["off", "color", "luminance"]
SEAM_FADE_FRAMES = 24
SEAM_ANALYSIS_FRAMES = 4
SEAM_MAX_EV = 0.25
SEAM_MAX_CHROMA = 0.08
PROMPT_TEMPLATES = {
    "reference": "<Video 1> provides the full performance, motion, timing and camera. "
                 "<Picture 1> defines the identity and appearance of the subject.",
    "guide": "style_transfer: Re-render this video with the style of <Picture 1>: "
             "bold graphic style, vibrant flat colours, clean outlines, flat shading. "
             "Keep the original scene, background and framing exactly as in the video; do not add any scenery from <Picture 1>.",
    "guide_retexture": "retexture: change the dress to red silk, glossy finish, keeping everything else unchanged.",
    "guide+reference": "style_transfer: Re-render this video with the style of <Picture 1>: "
                       "painterly brushwork, warm palette, soft shading, visible canvas texture.",
}
CACHE_SUBFOLDER = "h3_longtake"
PLAN_FILE = "plan.json"


# ----------------------------------------------------------------------------
# H3 temporal grid
# ----------------------------------------------------------------------------

def align_up(n):
    n = max(5, int(n))
    while n % 17 != 5:
        n += 1
    return n


def align_down(n):
    n = int(n)
    while n >= 5 and n % 17 != 5:
        n -= 1
    return n


def video_latent_t(frames):
    return 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2


def pixel_frames(latent_t):
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(int(latent_t)))


def latent_steps_for_frames(frames):
    """How many latent tokens cover exactly `frames` frames (1/4/4/4/4 cycle)."""
    steps, covered = 0, 0
    while covered < int(frames):
        covered += FRAME_PER_TOKEN[steps % 5]
        steps += 1
    return steps if covered == int(frames) else None


def audio_t_for_frames(frames):
    return int(round(float(frames) / FPS * AUDIO_HZ))


# ----------------------------------------------------------------------------
# slicing plan
# ----------------------------------------------------------------------------

def source_frames_at_24fps(n_source, source_fps):
    if abs(float(source_fps) - FPS) < 1e-6 or n_source <= 1:
        return int(n_source)
    return max(1, int(round(n_source / float(source_fps) * FPS)))


def build_plan(n24, clip_frames, context_frames, max_clips=0, overlap=None):
    """Source windows for every clip.

    clip i: source frames [src_start, src_start+length) at 24 fps; the first
    `ctx` frames are reconstructed from the previous clip's anchor and get
    trimmed; `new` fresh frames remain. The last clip is extended to the next
    17k+5 length by freezing the last source frame, then trimmed to `new` on
    output: the source is covered in full.

    overlap: source frames shared by consecutive clips (default =
    context_frames, anchor "at the head"). With overlap=0 the anchor sits
    before frame 0 and every clip produces only new frames.
    """
    L = align_up(clip_frames)
    C = int(context_frames)
    O = C if overlap is None else int(overlap)
    if C >= L:
        raise ValueError(f"H3 LongTake: context_frames ({C}) must be smaller than clip_frames ({L}).")
    if n24 < 5:
        raise ValueError("H3 LongTake: the source video is too short (fewer than 5 frames at 24 fps).")

    clips = []
    start = 0
    i = 0
    while True:
        ctx = O if i > 0 else 0
        if start + ctx >= n24:
            break
        needed = n24 - start
        length = L if needed >= L else align_up(needed)
        new = min(length - ctx, n24 - start - ctx)
        clips.append({
            "index": i,
            "src_start": start,
            "length": length,
            "ctx": ctx,
            "new": new,
            "padded": max(0, start + length - n24),
        })
        start += length - O
        i += 1
        if max_clips and i >= max_clips:
            break
    return {"clip_frames": L, "context_frames": C, "overlap": O, "n24": n24, "clips": clips}


def plan_report(plan, n_source, source_fps):
    clips = plan["clips"]
    total_new = sum(c["new"] for c in clips)
    lines = [
        f"source: {n_source} frames at {source_fps:g} fps -> {plan['n24']} frames at 24 fps "
        f"({plan['n24'] / FPS:.2f}s)",
        f"clips of {plan['clip_frames']} frames, context {plan['context_frames']} frames, "
        f"overlap {plan.get('overlap', plan['context_frames'])}, "
        f"stride {plan['clip_frames'] - plan.get('overlap', plan['context_frames'])} -> {len(clips)} clips",
        f"output frames: {total_new} ({total_new / FPS:.2f}s)",
    ]
    for c in clips:
        extra = f", +{c['padded']} frozen" if c["padded"] else ""
        lines.append(
            f"  clip {c['index']:03d}: source {c['src_start']}..{c['src_start'] + c['length'] - 1}"
            f" ({c['length']} frames{extra}), trim {c['ctx']}, new {c['new']}"
        )
    return "\n".join(lines)


# ----------------------------------------------------------------------------
# image / audio helpers
# ----------------------------------------------------------------------------

def _resize(image, width, height):
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, int(width), int(height), "lanczos", "disabled")
    return samples.movedim(1, -1)


def _snap32(w, h):
    return (max(CANVAS, round(w / CANVAS) * CANVAS), max(CANVAS, round(h / CANVAS) * CANVAS))


def _native_video_canvas(w, h):
    """Same canvas as the core node: short edge 768, max area 768x1344."""
    ratio = w / float(h)
    if ratio >= 1.0:
        nw, nh = REF_VIDEO_SHORT_EDGE * ratio, REF_VIDEO_SHORT_EDGE
    else:
        nw, nh = REF_VIDEO_SHORT_EDGE, REF_VIDEO_SHORT_EDGE / ratio
    if nw * nh > REF_VIDEO_MAX_PIXELS:
        s = math.sqrt(REF_VIDEO_MAX_PIXELS / (nw * nh))
        nw, nh = nw * s, nh * s
    cw, ch = _snap32(nw, nh)
    if w * h < cw * ch:
        cw, ch = _snap32(w, h)
    return cw, ch


def _ref_video_canvas(w, h, mode, out_w, out_h):
    if mode == "native":
        return _native_video_canvas(w, h)
    scale = min(1.0, math.sqrt((out_w * out_h) / float(w * h)))
    return _snap32(w * scale, h * scale)


ASPECTS = {"16:9": 16 / 9, "9:16": 9 / 16, "1:1": 1.0, "4:3": 4 / 3, "3:4": 3 / 4, "21:9": 21 / 9}
ASPECT_CHOICES = ["source"] + list(ASPECTS) + ["manual"]


def resolve_output_size(aspect, megapixels, width, height, src_w, src_h):
    """Output canvas: source aspect ratio (or a preset) at `megapixels`, or
    manual width/height. Always multiples of 32."""
    if aspect == "manual":
        return _snap32(width, height)
    ratio = (src_w / float(src_h)) if aspect == "source" else ASPECTS[aspect]
    px = max(0.05, float(megapixels)) * 1_000_000
    w = math.sqrt(px * ratio)
    # two candidates (snapped side -> derived other side): keep the one with the
    # most faithful aspect ratio
    a = _snap32(w, 0); a = (a[0], max(CANVAS, round(a[0] / ratio / CANVAS) * CANVAS))
    b = _snap32(0, w / ratio); b = (max(CANVAS, round(b[1] * ratio / CANVAS) * CANVAS), b[1])
    return min((a, b), key=lambda s: abs(s[0] / s[1] - ratio))


def _ref_image_canvas(w, h, mode, out_w, out_h):
    if mode == "match":
        scale = min(1.0, math.sqrt((out_w * out_h) / float(w * h)))
    else:
        scale = min(1.0, REF_IMAGE_SHORT_EDGE / float(min(w, h)))
    return _snap32(w * scale, h * scale)


def _take_source_frames(frames, source_fps, start, length):
    """Source slice resampled to 24 fps by index; past the end it repeats the
    last frame (frozen tail of the last clip)."""
    n = int(frames.shape[0])
    positions = torch.arange(start, start + length, dtype=torch.float32)
    if abs(float(source_fps) - FPS) < 1e-6 or n <= 1:
        idx = positions.to(torch.long)
    else:
        idx = torch.round(positions * (float(source_fps) / FPS)).to(torch.long)
    idx = torch.clamp(idx, 0, n - 1).to(frames.device)
    return frames.index_select(0, idx)


def _slice_audio(audio, start_frame, length):
    wave = audio["waveform"]
    sr = int(audio["sample_rate"])
    a = int(round(start_frame / FPS * sr))
    b = int(round((start_frame + length) / FPS * sr))
    piece = wave[..., a:b]
    return {"waveform": piece, "sample_rate": sr}


# ----------------------------------------------------------------------------
# source: from an IMAGE batch (small tests) or from a file (long videos, RAM of
# a single clip: ffmpeg decodes only the requested slice)
# ----------------------------------------------------------------------------

VIDEO_EXTS = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")
NO_FILE = "(use source_video)"
AUDIO_AUTO = "(auto from project)"
AUDIO_NONE = "(no audio)"


def _list_input_videos():
    input_dir = folder_paths.get_input_directory()
    try:
        files = sorted(f for f in os.listdir(input_dir)
                       if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(VIDEO_EXTS))
    except OSError:
        files = []
    return [NO_FILE] + files


def _probe_video_file(path):
    """(fps, source frames, duration, width, height)."""
    fps = frames = w = h = 0
    duration = 0.0
    try:
        import cv2
        cap = cv2.VideoCapture(path)
        try:
            if cap.isOpened():
                fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        finally:
            cap.release()
    except Exception:
        pass
    if fps > 0 and frames > 0:
        duration = frames / fps
    if duration <= 0 or w <= 0 or h <= 0:
        import re
        res = subprocess.run([_find_ffmpeg(), "-hide_banner", "-i", path], capture_output=True)
        text = res.stderr.decode("utf-8", "replace")
        m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", text)
        if not m:
            raise RuntimeError(f"H3 LongTake: cannot read the duration of {path}.")
        duration = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        f = re.search(r"(\d+(?:\.\d+)?)\s*fps", text)
        fps = float(f.group(1)) if f else 0.0
        frames = int(duration * fps) if fps > 0 else 0
        vline = next((l for l in text.splitlines() if " Video: " in l), "")
        d = re.search(r"(?<!\d)(\d{2,5})x(\d{2,5})(?!\d)", vline)
        if d:
            w, h = int(d.group(1)), int(d.group(2))
    if duration <= 0 or w <= 0 or h <= 0:
        raise RuntimeError(f"H3 LongTake: {path} is not a readable video.")
    return fps, frames, duration, w, h


def _apply_range(total24, start_seconds, end_seconds):
    """[start, end) window in 24 fps frames inside the source; end<=0 = to the end."""
    start = max(0, int(round(float(start_seconds or 0) * FPS)))
    end = total24 if float(end_seconds or 0) <= 0 else min(total24, int(round(float(end_seconds) * FPS)))
    if end - start < 5:
        raise ValueError(f"H3 LongTake: empty source range ({start_seconds}s..{end_seconds}s).")
    return start, end


class _TensorSource:
    def __init__(self, frames, fps, audio=None, start_seconds=0.0, end_seconds=0.0):
        self.frames = frames
        self.fps = float(fps)
        self.audio = audio
        self.n_source = int(frames.shape[0])
        self.width, self.height = int(frames.shape[2]), int(frames.shape[1])
        total24 = source_frames_at_24fps(self.n_source, self.fps)
        self.offset, end = _apply_range(total24, start_seconds, end_seconds)
        self.n24 = end - self.offset
        self.label = f"source_video ({self.n_source} frames at {self.fps:g} fps, frame24 {self.offset}..{end - 1})"

    def frames_at(self, start, length, cw, ch):
        return _resize(_take_source_frames(self.frames, self.fps, self.offset + start, length), cw, ch)

    def audio_at(self, start, length):
        return _slice_audio(self.audio, self.offset + start, length) if self.audio is not None else None


class _FileSource:
    def __init__(self, path, start_seconds=0.0, end_seconds=0.0):
        self.path = path
        self.fps, self.n_source, duration, self.width, self.height = _probe_video_file(path)
        # same formula as VHS: truncated, not rounded
        total24 = max(1, int(duration * FPS))
        self.offset, end = _apply_range(total24, start_seconds, end_seconds)
        self.n24 = end - self.offset
        self.label = f"{os.path.basename(path)} ({duration:.2f}s at {self.fps:g} fps, frame24 {self.offset}..{end - 1})"

    def frames_at(self, start, length, cw, ch):
        # no -ss: the fps=24 conversion is identical for every clip, so slice i+1
        # starts exactly on the last C frames of slice i.
        # setpts renumbers the timestamps after select: without it ffmpeg fills the
        # gap by duplicating the first selected frame (FROZEN slice on files with
        # real timestamps, e.g. TikTok). -vsync 0 = never duplicate/drop.
        vf = (f"fps={FPS},select=gte(n\\,{int(self.offset + start)}),setpts=N/({FPS}*TB),"
              f"scale={cw}:{ch}:flags=lanczos")
        cmd = [_find_ffmpeg(), "-v", "error", "-i", self.path, "-an", "-vf", vf, "-vsync", "0",
               "-frames:v", str(int(length)), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError("H3 LongTake: ffmpeg failed to decode the slice:\n"
                               + proc.stderr.decode("utf-8", "replace"))
        frame_bytes = cw * ch * 3
        count = len(proc.stdout) // frame_bytes
        if count <= 0:
            raise RuntimeError(f"H3 LongTake: no frames decoded from {self.path} starting at {start}.")
        arr = torch.frombuffer(bytearray(proc.stdout[:count * frame_bytes]), dtype=torch.uint8)
        frames = arr.view(count, ch, cw, 3).to(torch.float32).div_(255.0)
        if count < length:  # frozen tail
            frames = torch.cat([frames, frames[-1:].expand(length - count, -1, -1, -1)], dim=0)
        return frames

    def audio_at(self, start, length):
        cmd = [_find_ffmpeg(), "-v", "error", "-i", self.path, "-vn",
               "-ss", f"{(self.offset + start) / FPS:.6f}", "-t", f"{length / FPS:.6f}",
               "-ac", "2", "-ar", "48000", "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0 or len(proc.stdout) < 8:
            return None
        arr = torch.frombuffer(bytearray(proc.stdout), dtype=torch.float32)
        arr = arr[: (arr.numel() // 2) * 2].view(-1, 2).T.contiguous()
        return {"waveform": arr.unsqueeze(0), "sample_rate": 48000}


def _encode_ref_audio(audio_vae, audio):
    import torchaudio
    waveform = audio["waveform"]
    sr = audio["sample_rate"]
    vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
    if sr != vae_sr:
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    z = audio_vae.encode(waveform[:1].movedim(1, -1))
    return z, z.shape[-1]


def _empty_av_latent(width, height, frames):
    lt = video_latent_t(frames)
    at = audio_t_for_frames(frames)
    dev = comfy.model_management.intermediate_device()
    video = torch.zeros([1, 24, lt, height // 16, width // 16], device=dev)
    audio = torch.zeros([1, 32, 2, at], device=dev)
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}


def _split_av(samples):
    if getattr(samples, "is_nested", False):
        parts = list(samples.unbind())
    else:
        parts = list(samples)
    video, audio = parts[0], parts[1]
    if video.ndim == 4:
        video = video.unsqueeze(0)
    if audio.ndim == 3:
        audio = audio.unsqueeze(0)
    return video, audio


# ----------------------------------------------------------------------------
# motion context: latent tail of the previous clip -> keyframe at frame 0
# ----------------------------------------------------------------------------

def _luminance(x):
    return x[..., 0] * 0.2126 + x[..., 1] * 0.7152 + x[..., 2] * 0.0722


def seam_match_fn(images, boundary, mode="color", fade=SEAM_FADE_FRAMES, ref=None):
    """Removes the exposure/colour jump at the seam: the frames after `boundary`
    are corrected towards `ref` (the REAL last frames of the previous clip) with a
    luminance gain (in EV, clamped) and, in color mode, a per-channel chroma
    offset; the correction fades out with a cosine over `fade` frames. Without
    `ref` it uses the frames before `boundary` (regenerated context): fine only
    when they are copied (inpaint); with the keyframe anchor they are generated
    and already carry the new clip's colour cast.
    Idea from nikaskeba's Seam Exposure Match (luminance only), extended to colour."""
    if mode == "off" or boundary >= int(images.shape[0]):
        return images
    if ref is None:
        if boundary <= 0:
            return images
        ref = images[max(0, boundary - SEAM_ANALYSIS_FRAMES):boundary]
    ref = ref.float()
    if ref.shape[0] == 0:
        return images
    n = min(int(fade), int(images.shape[0]) - boundary)
    if n <= 0:
        return images
    head = images[boundary:boundary + min(SEAM_ANALYSIS_FRAMES, n)].float()
    rl, tl = float(_luminance(ref).mean()), float(_luminance(head).mean())
    if rl < 1e-4 or tl < 1e-4:
        return images
    ev = max(-SEAM_MAX_EV, min(SEAM_MAX_EV, math.log2(rl / tl)))
    w = 0.5 * (1.0 + torch.cos(math.pi * torch.arange(n, dtype=torch.float32) / max(1, n - 1)))
    out = images.clone()
    seg = out[boundary:boundary + n].float() * torch.exp2(ev * w)[:, None, None, None]
    if mode == "color":
        rm = ref.mean(dim=(0, 1, 2))
        tm = seg[:min(SEAM_ANALYSIS_FRAMES, n)].mean(dim=(0, 1, 2))
        delta = ((rm - rm.mean()) - (tm - tm.mean())).clamp(-SEAM_MAX_CHROMA, SEAM_MAX_CHROMA)
        delta = delta - _luminance(delta)   # hue only: luminance stays what the gain set
        seg = seg + delta[None, None, None, :] * w[:, None, None, None]
    out[boundary:boundary + n] = seg.clamp(0.0, 1.0).to(out.dtype)
    return out


def _tail_frames_from_mp4(path, total, n):
    """Last n frames of a chunk (for the seam match when the previous clip is cached)."""
    _, _, _, w, h = _probe_video_file(path)
    start = max(0, int(total) - int(n))
    cmd = [_find_ffmpeg(), "-v", "error", "-i", path, "-an", "-vf", f"select=gte(n\\,{start})", "-vsync", "0",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    frames = torch.frombuffer(bytearray(raw), dtype=torch.uint8).reshape(-1, h, w, 3)
    return frames.float() / 255.0


def _pad_even_hw(block, target_video):
    """The DiT patchifies 2x2: conditioning blocks need even latent H/W."""
    ph = ((int(target_video.shape[3]) + 1) // 2) * 2
    pw = ((int(target_video.shape[4]) + 1) // 2) * 2
    pad_h, pad_w = ph - int(block.shape[3]), pw - int(block.shape[4])
    if pad_h or pad_w:
        block = torch.nn.functional.pad(block, (0, pad_w, 0, pad_h))
    return block


def guide_keyframe(z_guide, target_video):
    """Source slice encoded as a guide latent: a keyframe at frame 0 spanning the
    whole clip (what MiniMaxH3AddGuide does with a batch of frames)."""
    if int(z_guide.shape[2]) != int(target_video.shape[2]):
        raise ValueError(f"H3 LongTake: guide latent of {int(z_guide.shape[2])} tokens, clip of {int(target_video.shape[2])}.")
    return {"resolved_frame_index": 0, "latent": _pad_even_hw(z_guide[:1].clone(), target_video)}


def motion_context_keyframes(prev_video, prev_audio, context_frames, target_video, with_audio=True, position="head"):
    """Tail of the previous clip as an H3 keyframe. position="head": occupies
    frames 0..C-1 of the target (trimmed afterwards). position="before": sits at
    frames -C..-1, before the target, which is therefore all new (kat3ri / Niko "before")."""
    steps = latent_steps_for_frames(context_frames)
    if steps is None:
        raise ValueError(f"H3 LongTake: context_frames={context_frames} is not a whole number of latent steps.")
    total_t = int(prev_video.shape[2])
    if steps > total_t:
        raise ValueError("H3 LongTake: the previous clip is shorter than the requested context.")
    start_t = total_t - steps
    if start_t % 5 != 0:
        raise RuntimeError(f"H3 LongTake: wrong latent phase (the tail starts at cycle {start_t % 5}).")

    block = prev_video[:1, :, start_t:].clone()
    # the DiT patchifies 2x2: conditioning blocks need even latent H/W
    ph = ((int(target_video.shape[3]) + 1) // 2) * 2
    pw = ((int(target_video.shape[4]) + 1) // 2) * 2
    pad_h, pad_w = ph - int(block.shape[3]), pw - int(block.shape[4])
    if pad_h or pad_w:
        block = torch.nn.functional.pad(block, (0, pad_w, 0, pad_h))

    index0 = 0 if position == "head" else -int(context_frames)
    keyframes = [{"resolved_frame_index": index0, "latent": block}]

    if with_audio and prev_audio is not None:
        rt = min(audio_t_for_frames(context_frames), int(prev_audio.shape[-1]))
        if rt >= 1:
            tail = prev_audio[:1, ..., -rt:].clone()
            # the audio ends where the C context frames end
            start_frame = float(index0 + context_frames) - float(rt) / FRAME_RESCALE
            keyframes.append({"resolved_frame_index": start_frame, "audio_latent": tail})
    return keyframes


def tail_context_keyframes(next_video, next_audio, context_frames, target_video, with_audio=True, position="head"):
    """Anchors the TAIL of the clip to the head of the already existing next clip
    (redo_one): the first C latent frames of next coincide with the last C of the
    current clip, so they go to frame target_frames - C."""
    steps = latent_steps_for_frames(context_frames)
    if steps is None or steps > int(next_video.shape[2]):
        raise ValueError("H3 LongTake: the next clip is too short to anchor the tail.")
    target_frames = pixel_frames(int(target_video.shape[2]))
    index = target_frames - int(context_frames) if position == "head" else target_frames
    if index <= 0 or (position == "head" and (int(target_video.shape[2]) - steps) % 5 != 0):
        raise RuntimeError("H3 LongTake: wrong latent phase for the tail anchor.")

    block = next_video[:1, :, :steps].clone()
    ph = ((int(target_video.shape[3]) + 1) // 2) * 2
    pw = ((int(target_video.shape[4]) + 1) // 2) * 2
    pad_h, pad_w = ph - int(block.shape[3]), pw - int(block.shape[4])
    if pad_h or pad_w:
        block = torch.nn.functional.pad(block, (0, pad_w, 0, pad_h))
    keyframes = [{"resolved_frame_index": index, "latent": block}]

    if with_audio and next_audio is not None:
        rt = min(audio_t_for_frames(context_frames), int(next_audio.shape[-1]))
        if rt >= 1:
            keyframes.append({"resolved_frame_index": float(index), "audio_latent": next_audio[:1, ..., :rt].clone()})
    return keyframes


# ----------------------------------------------------------------------------
# sampling (positive only, like the H3 workflows with the Turbo LoRA)
# ----------------------------------------------------------------------------

class _PositiveGuider(comfy.samplers.CFGGuider):
    def set_conds(self, positive):
        self.inner_set_conds({"positive": positive})


def _sample(model, positive, latent, seed, sampler_name, scheduler, steps, noise_mask=None):
    guider = _PositiveGuider(model)
    guider.set_conds(positive)
    sampler = comfy.samplers.sampler_object(sampler_name)
    sigmas = comfy.samplers.calculate_sigmas(
        model.get_model_object("model_sampling"), scheduler, int(steps)).cpu()

    latent_image = comfy.sample.fix_empty_latent_channels(
        model, latent["samples"],
        latent.get("downscale_ratio_spacial", None), latent.get("downscale_ratio_temporal", None))
    noise = comfy.sample.prepare_noise(latent_image, int(seed), None)
    x0_output = {}
    callback = latent_preview.prepare_callback(model, sigmas.shape[-1] - 1, x0_output)
    samples = guider.sample(
        noise, latent_image, sampler, sigmas,
        denoise_mask=noise_mask, callback=callback,
        disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED, seed=int(seed))
    return samples.to(comfy.model_management.intermediate_device())


# ----------------------------------------------------------------------------
# "inpaint" anchor: the previous clip's tail is copied INTO the target latent
# and protected with the denoise mask (native masked AV preservation) instead
# of travelling as a separate keyframe
# ----------------------------------------------------------------------------

class _InpaintAnchor:
    def __init__(self, latent):
        video, audio = _split_av(latent["samples"])
        self.video = video.clone()
        self.audio = audio.clone()
        self.vmask = torch.ones(1, 1, *self.video.shape[2:], dtype=torch.float32)
        self.amask = torch.ones(1, 1, *self.audio.shape[2:], dtype=torch.float32)
        self.used = False

    def _check(self, src_video, context_frames):
        steps = latent_steps_for_frames(context_frames)
        if steps is None or steps > int(src_video.shape[2]) or steps >= int(self.video.shape[2]):
            raise ValueError("H3 LongTake: context not compatible with the inpaint anchor.")
        if tuple(src_video.shape[3:]) != tuple(self.video.shape[3:]):
            raise ValueError("H3 LongTake: latent resolution differs between clips, cannot anchor.")
        return steps, min(audio_t_for_frames(context_frames), int(self.audio.shape[-1]))

    def head_from_tail(self, prev_video, prev_audio, context_frames, with_audio=True):
        steps, rt = self._check(prev_video, context_frames)
        if (int(prev_video.shape[2]) - steps) % 5 != 0:
            raise RuntimeError("H3 LongTake: wrong latent phase (head inpaint anchor).")
        self.video[:, :, :steps] = prev_video[:1, :, -steps:].to(self.video)
        self.vmask[:, :, :steps] = 0.0
        if with_audio and prev_audio is not None:
            rt = min(rt, int(prev_audio.shape[-1]))
            self.audio[..., :rt] = prev_audio[:1, ..., -rt:].to(self.audio)
            self.amask[..., :rt] = 0.0
        self.used = True

    def tail_from_head(self, next_video, next_audio, context_frames, with_audio=True):
        steps, rt = self._check(next_video, context_frames)
        if (int(self.video.shape[2]) - steps) % 5 != 0:
            raise RuntimeError("H3 LongTake: wrong latent phase (tail inpaint anchor).")
        self.video[:, :, -steps:] = next_video[:1, :, :steps].to(self.video)
        self.vmask[:, :, -steps:] = 0.0
        if with_audio and next_audio is not None:
            rt = min(rt, int(next_audio.shape[-1]))
            self.audio[..., -rt:] = next_audio[:1, ..., :rt].to(self.audio)
            self.amask[..., -rt:] = 0.0
        self.used = True

    def latent(self):
        return {"samples": comfy.nested_tensor.NestedTensor((self.video, self.audio))}

    def mask(self):
        return comfy.nested_tensor.NestedTensor((self.vmask, self.amask)) if self.used else None


# ----------------------------------------------------------------------------
# disk: project cache, chunk mp4s, stitch
# ----------------------------------------------------------------------------

def _project_dir(project_name):
    name = "".join(ch for ch in str(project_name).strip() if ch.isalnum() or ch in "-_ .").strip()
    if not name:
        raise ValueError("H3 LongTake: project_name is empty or invalid.")
    path = os.path.join(folder_paths.get_output_directory(), CACHE_SUBFOLDER, name)
    os.makedirs(path, exist_ok=True)
    return path


def _clip_paths(project_dir, index):
    base = os.path.join(project_dir, f"clip_{index:03d}")
    return base + ".latent.pt", base + ".mp4"


def _find_ffmpeg():
    try:
        from videohelpersuite.utils import ffmpeg_path
        if ffmpeg_path:
            return ffmpeg_path
    except Exception:
        pass
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        return get_ffmpeg_exe()
    except Exception:
        pass
    found = shutil.which("ffmpeg")
    if not found:
        raise RuntimeError("H3 LongTake: ffmpeg not found (install imageio-ffmpeg or VideoHelperSuite).")
    return found


def _write_mp4(path, images, crf):
    """images: [T,H,W,3] float 0..1 -> h264 yuv420p mp4 at 24 fps through a pipe."""
    ffmpeg = _find_ffmpeg()
    t, h, w = int(images.shape[0]), int(images.shape[1]), int(images.shape[2])
    cmd = [
        ffmpeg, "-y", "-v", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(FPS), "-i", "pipe:0",
        "-c:v", "libx264", "-preset", "medium", "-crf", str(int(crf)), "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", path,
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for start in range(0, t, 16):
            chunk = images[start:start + 16].clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).cpu()
            proc.stdin.write(chunk.numpy().tobytes())
    finally:
        proc.stdin.close()
        err = proc.stderr.read().decode("utf-8", "replace")
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"H3 LongTake: ffmpeg failed writing {path}:\n{err}")


def _ui_video(path):
    """Entry for the video player embedded in the node (same format as core SaveVideo)."""
    out_dir = folder_paths.get_output_directory()
    rel = os.path.relpath(path, out_dir).replace("\\", "/")
    subfolder, filename = os.path.split(rel)
    return {"images": [{"filename": filename, "subfolder": subfolder, "type": "output"}], "animated": (True,)}


def _concat_mp4(files, out_path, audio_args=None, stdin_bytes=None, metadata=None):
    """Concatenates homogeneous mp4s without re-encoding (ffmpeg concat list).
    `metadata` (workflow/prompt) goes into the file tags like core SaveVideo does:
    dropping the mp4 onto ComfyUI reopens the graph."""
    ffmpeg = _find_ffmpeg()
    list_path = out_path + ".txt"
    meta_path = out_path + ".ffmeta"
    with open(list_path, "w", encoding="utf-8") as fh:
        for path in files:
            fh.write("file '" + path.replace("\\", "/").replace("'", r"'\''") + "'\n")
    args = [ffmpeg, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", list_path]
    audio_args = list(audio_args or [])
    # inputs (-i ...) come before the output options (-map, -t, -c:a ...)
    n_inputs = sum(1 for a in audio_args if a == "-i")
    split = (max(i for i, a in enumerate(audio_args) if a == "-i") + 2) if n_inputs else 0
    args += audio_args[:split]
    movflags = "+faststart"
    if metadata:
        _write_ffmetadata(meta_path, metadata)
        args += ["-i", meta_path]
        audio_args = audio_args[split:] + ["-map_metadata", str(1 + n_inputs)]
        movflags = "use_metadata_tags+faststart"
    else:
        audio_args = audio_args[split:]
    args += audio_args
    args += ["-c:v", "copy", "-movflags", movflags, out_path]
    try:
        proc = subprocess.run(args, input=stdin_bytes, capture_output=True)
    finally:
        for tmp in (list_path, meta_path):
            try:
                os.remove(tmp)
            except OSError:
                pass
    if proc.returncode != 0:
        raise RuntimeError("H3 LongTake: ffmpeg failed during concatenation:\n"
                           + proc.stderr.decode("utf-8", "replace"))


WORKFLOW_FILE = "workflow.json"
API_PROMPT_FILE = "api_prompt.json"
SETTINGS_FILE = "settings.json"
_SKIP_INPUT_KEYS = ("prompt", "extra_pnginfo", "api_prompt")


def _describe_node(api_prompt, node_id, depth=0):
    """Readable summary of an API-graph node: widget values and, for linked
    inputs, the source node (recursive, to show the LoRA chain)."""
    node = api_prompt.get(str(node_id)) or {}
    out = {"node": node.get("class_type")}
    for k, v in (node.get("inputs") or {}).items():
        if isinstance(v, list) and len(v) == 2 and str(v[0]) in api_prompt:
            out[k] = _describe_node(api_prompt, v[0], depth + 1) if depth < 6 else {"node": api_prompt[str(v[0])].get("class_type")}
        else:
            out[k] = v
    return out


def _save_workflow_copy(project_dir, api_prompt, extra_pnginfo, node_class="H3LongTakeRender", fresh=False):
    """workflow.json (UI graph: drop it onto ComfyUI to get back to the project),
    api_prompt.json (API graph) and settings.json (readable summary of the node).
    Written at the start of every run, before clip 0. The `*_first.json` copies
    belong to the project's first run and are never overwritten (except by restart)."""
    def dump(name, data):
        with open(os.path.join(project_dir, name), "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=1)
        first = name.replace(".json", "_first.json")
        if fresh or not os.path.isfile(os.path.join(project_dir, first)):
            shutil.copyfile(os.path.join(project_dir, name), os.path.join(project_dir, first))
    try:
        workflow = (extra_pnginfo or {}).get("workflow") if isinstance(extra_pnginfo, dict) else None
        if workflow:
            dump(WORKFLOW_FILE, workflow)
        if isinstance(api_prompt, dict) and api_prompt:
            dump(API_PROMPT_FILE, api_prompt)
            dump(SETTINGS_FILE, {nid: _describe_node(api_prompt, nid) for nid, n in api_prompt.items()
                                 if n.get("class_type") == node_class})
    except Exception as exc:  # saving the graph must never stop a run
        _LOG.warning("H3 LongTake: workflow not saved into the project: %s", exc)


def _project_metadata(project_dir, api_prompt=None, extra_pnginfo=None):
    """'workflow' and 'prompt' tags for the mp4 (like core SaveVideo): prefers the
    ones the Render saved into the project, else those of the current run."""
    meta = {}
    for key, fname in (("workflow", WORKFLOW_FILE), ("prompt", API_PROMPT_FILE)):
        path = os.path.join(project_dir, fname)
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as fh:
                    meta[key] = json.load(fh)
                continue
            except Exception:
                pass
        if key == "workflow" and isinstance(extra_pnginfo, dict) and extra_pnginfo.get("workflow"):
            meta[key] = extra_pnginfo["workflow"]
        elif key == "prompt" and isinstance(api_prompt, dict) and api_prompt:
            meta[key] = api_prompt
    return meta


def _ffmetadata_escape(text):
    return "".join("\\" + ch if ch in "=;#\\\n" else ch for ch in text)


def _write_ffmetadata(path, metadata):
    """FFMETADATA file with the JSON tags (avoids the Windows command-line length limit)."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(";FFMETADATA1\n")
        for key, value in metadata.items():
            fh.write(f"{key}={_ffmetadata_escape(json.dumps(value, ensure_ascii=False, separators=(',', ':')))}\n")


def _existing_chunks(project_dir):
    """mp4s of the chunks present, in order, up to the first missing one."""
    files = []
    i = 0
    while True:
        _, mp4_path = _clip_paths(project_dir, i)
        if not os.path.isfile(mp4_path):
            return files
        files.append(mp4_path)
        i += 1


def _load_plan(project_dir):
    path = os.path.join(project_dir, PLAN_FILE)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _save_plan(project_dir, payload):
    with open(os.path.join(project_dir, PLAN_FILE), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _delete_clips_from(project_dir, first_index):
    removed = []
    for entry in os.listdir(project_dir):
        if not entry.startswith("clip_"):
            continue
        try:
            idx = int(entry[5:8])
        except ValueError:
            continue
        if idx >= first_index:
            os.remove(os.path.join(project_dir, entry))
            removed.append(entry)
    return removed


# ----------------------------------------------------------------------------
# nodes
# ----------------------------------------------------------------------------

class H3LongTakeRender:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "source_file": (_list_input_videos(), {"video_upload": True,
                                 "tooltip": "Source video in input/: ffmpeg decodes only each clip's slice (minimal RAM). "
                                            "With '(use source_video)' the connected IMAGE batch is used instead."}),
                "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True,
                                      "default": PROMPT_TEMPLATES["reference"],
                                      "tooltip": "Changing source_role fills this field with the role's template "
                                                 "(only if empty or still equal to a template). Empty = the role's template."}),
                "project_name": ("STRING", {"default": "longtake"}),
                "width": ("INT", {"default": 896, "min": 32, "max": 4096, "step": 32,
                                  "tooltip": "Only used with aspect=manual."}),
                "height": ("INT", {"default": 576, "min": 32, "max": 4096, "step": 32,
                                   "tooltip": "Only used with aspect=manual."}),
                "clip_frames": ("INT", {"default": 124, "min": 22, "max": 362, "step": 17,
                                        "tooltip": "Frames generated per clip (17k+5). 124 = 5.2s, 243 = 10.1s."}),
                "context_frames": (CONTEXT_CHOICES, {"default": "5",
                                                     "tooltip": "Tail frames of the previous clip used as anchor and overlap. "
                                                                "5: clean seam and best adherence to <Video 1> (measured). 22: longer anchor, steals attention from the reference."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 100}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
                "mode": (["continue", "restart", "redo_from", "redo_one"], {"default": "continue",
                          "tooltip": "continue: skips the clips already on disk. restart: deletes everything. "
                                     "redo_from: redoes from redo_from_clip onwards. "
                                     "redo_one: redoes ONLY redo_from_clip, anchoring head and tail to the neighbouring clips (change the seed)."}),
                "redo_from_clip": ("INT", {"default": 0, "min": 0, "max": 9999}),
                "max_clips": ("INT", {"default": 0, "min": 0, "max": 9999,
                                      "tooltip": "0 = all. Handy to try only the first N clips."}),
                "dry_run": ("BOOLEAN", {"default": False, "tooltip": "Only shows the slicing plan, does not generate."}),
            },
            "optional": {
                "ref_image_1": ("IMAGE",),
                "ref_image_2": ("IMAGE",),
                "ref_image_3": ("IMAGE",),
                "source_video": ("IMAGE", {"tooltip": "Alternative to source_file for short tests: IMAGE batch (float32, ~6 MB/frame at 0.5 MP)."}),
                "source_fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001,
                                         "tooltip": "Real FPS of source_video (ignored with source_file)."}),
                "source_audio": ("AUDIO", {"tooltip": "Audio of source_video (ignored with source_file)."}),
                "use_source_audio": ("BOOLEAN", {"default": False,
                                     "tooltip": "Passes the source audio as the soundtrack of <Video 1> (lip-sync); costs tokens."}),
                "ref_image_size": (["match", "max"], {"default": "match"}),
                "ref_video_size": (["match", "native"], {"default": "match",
                                    "tooltip": "match: the reference video is scaled to the output area (fewer tokens). native: the core node's 768 canvas."}),
                "audio_context": ("BOOLEAN", {"default": True, "tooltip": "Also carries the audio tail of the previous clip."}),
                "chunk_crf": ("INT", {"default": 10, "min": 0, "max": 30}),
                # deliberately at the end of the optionals: previously saved
                # workflows map widgets by position
                "aspect": (ASPECT_CHOICES, {"default": "source",
                           "tooltip": "Aspect ratio of the output canvas. source = the source video's (recommended: "
                                      "with a different aspect ratio H3 tends to copy <Video 1> and ignore <Picture 1>)."}),
                "megapixels": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 2.0, "step": 0.05,
                               "tooltip": "Area of the output canvas (ignored with aspect=manual). 0.5 MP ~ 544x960 in 9:16."}),
                "prompt_text": ("STRING", {"forceInput": True,
                                "tooltip": "Prompt from an external text node: when connected and not empty it replaces the prompt field."}),
                "start_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.1,
                                  "tooltip": "Start of the source range to use."}),
                "end_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.1,
                                "tooltip": "End of the source range (0 = to the end). Handy to cut tails such as the TikTok end card."}),
                "anchor_mode": (["keyframe", "inpaint", "none"], {"default": "keyframe",
                                "tooltip": "How the previous clip is hooked. keyframe: latent tail as an H3 guide on frames 0..C-1, then trimmed (Motion Context style). "
                                           "inpaint: tail copied into the latent and protected with the mask (native masked AV preservation; fine with context 5, a cut with 22). "
                                           "none: no anchor, independent clips (diagnostic: upper bound of reference adherence)."}),
                "source_role": (SOURCE_ROLES, {"default": "reference",
                                "tooltip": "reference: the slice is <Video 1> (Ref2VA, motion suggestion). "
                                           "guide: the slice is a guide latent anchored at frame 0 for the whole clip "
                                           "(frame-accurate motion and framing; for the StyleTransfer LoRA, with <Picture 1> = style). "
                                           "guide_retexture: like guide, text-only 'retexture:' template. "
                                           "guide+reference: both."}),
                "seam_match": (SEAM_MODES, {"default": "off",
                               "tooltip": "At the seam with the previous clip, corrects the first 24 frames towards the previous clip's "
                                          "last frames: color = luminance + hue (measured: dE 5.8 -> 0.9), "
                                          "luminance = exposure only. Pixels only, after decoding; the latent is untouched."}),
            },
            "hidden": {"api_prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("last_clip", "project_dir", "report")
    FUNCTION = "render"
    CATEGORY = "H3 LongTake"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Generates a long video with MiniMax H3 Ref2VA clip by clip: every clip sees its own "
        "slice of the source as <Video 1> (or as a guide latent) and the previous clip's tail as anchor. "
        "Chunks land in output/h3_longtake/<project_name>/; assemble them with H3 LongTake Stitch."
    )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # depends on what is on disk

    @classmethod
    def VALIDATE_INPUTS(cls, source_file=None):
        # files uploaded after startup are not in the combo list yet
        if source_file and source_file != NO_FILE and not folder_paths.exists_annotated_filepath(source_file):
            return f"Video not found in input/: {source_file}"
        return True

    # ------------------------------------------------------------------

    def render(self, model, clip, vae, audio_vae, source_file, prompt, project_name,
               width, height, clip_frames, context_frames, seed, steps, sampler_name, scheduler,
               mode, redo_from_clip, max_clips, dry_run,
               ref_image_1=None, ref_image_2=None, ref_image_3=None,
               source_video=None, source_fps=24.0, source_audio=None, use_source_audio=False,
               ref_image_size="match", ref_video_size="match", audio_context=True, chunk_crf=10,
               aspect="source", megapixels=0.5, prompt_text=None, start_seconds=0.0, end_seconds=0.0,
               anchor_mode="keyframe", source_role="reference", seam_match="off",
               api_prompt=None, extra_pnginfo=None):

        if source_role not in SOURCE_ROLES:
            raise ValueError(f"H3 LongTake: unknown source_role: {source_role}")
        use_guide = source_role.startswith("guide")
        use_ref_video = source_role in ("reference", "guide+reference")
        if isinstance(prompt_text, str) and prompt_text.strip():
            prompt = prompt_text
        template_used = False
        if not str(prompt).strip():
            prompt, template_used = PROMPT_TEMPLATES[source_role], True
        if use_guide and not use_ref_video and "<Video 1>" in prompt:
            raise ValueError("H3 LongTake: with source_role=guide the video is not a reference: remove <Video 1> from the prompt "
                             "(use the template: style_transfer: ... <Picture 1> ...).")
        if source_file and source_file != NO_FILE:
            source = _FileSource(folder_paths.get_annotated_filepath(source_file), start_seconds, end_seconds)
        elif torch.is_tensor(source_video) and source_video.ndim == 4:
            source = _TensorSource(source_video, source_fps, source_audio, start_seconds, end_seconds)
        else:
            raise ValueError("H3 LongTake: pick a source_file or connect source_video.")
        n24 = source.n24
        before = anchor_mode in ("keyframe_before", "none")
        plan = build_plan(n24, clip_frames, int(context_frames), int(max_clips), overlap=0 if before else None)
        clips = plan["clips"]
        C = plan["context_frames"]
        width, height = resolve_output_size(aspect, megapixels, width, height, source.width, source.height)
        n_refs = sum(img is not None for img in (ref_image_1, ref_image_2, ref_image_3))
        report_lines = [
            plan_report(plan, source.n_source, source.fps),
            f"output canvas {width}x{height} (aspect={aspect}, source {source.width}x{source.height}); "
            f"{n_refs} reference images; anchor {anchor_mode}, audio_context={bool(audio_context)}; "
            f"source as {source_role}; seam_match={seam_match}",
        ]
        if template_used:
            report_lines.append(f"empty prompt: using the {source_role} template: {prompt}")
        if n_refs == 0:
            report_lines.append("WARNING: no reference image connected, <Picture N> in the prompt has no effect.")

        placeholder = torch.zeros(1, 64, 64, 3)
        if dry_run:
            report = "\n".join(report_lines)
            print("[H3 LongTake] dry run\n" + report)
            return {"ui": {"text": [report]}, "result": (placeholder, "", report)}

        project_dir = _project_dir(project_name)
        signature = {
            "width": int(width), "height": int(height),
            "clip_frames": plan["clip_frames"], "context_frames": C, "n24": n24,
            "source": source.label, "overlap": 0 if before else C,
            "source_role": source_role,
        }
        previous = _load_plan(project_dir)
        if mode == "restart":
            removed = _delete_clips_from(project_dir, 0)
            if removed:
                report_lines.append(f"restart: removed {len(removed)} files")
            previous = None
        elif mode == "redo_from":
            removed = _delete_clips_from(project_dir, int(redo_from_clip))
            report_lines.append(f"redo_from {int(redo_from_clip)}: removed {len(removed)} files")
        elif mode == "redo_one":
            target = int(redo_from_clip)
            if target >= len(clips):
                raise ValueError(f"H3 LongTake: redo_one: clip {target} does not exist in the plan (0..{len(clips) - 1}).")
            for path in _clip_paths(project_dir, target):
                if os.path.isfile(path):
                    os.remove(path)
            report_lines.append(f"redo_one {target}: regenerating only this clip, tail anchored to clip {target + 1}"
                                if target + 1 < len(clips) else f"redo_one {target}: last clip, head anchor only")
        if previous is not None and previous.get("signature") != signature:
            raise ValueError(
                "H3 LongTake: the project on disk has a different plan (resolution, clip_frames, "
                "context, source role or source). Use mode=restart or change project_name.\n"
                f"on disk: {previous.get('signature')}\nnow: {signature}"
            )
        prompt_hash = hashlib.sha1(prompt.encode("utf-8")).hexdigest()[:12]
        if previous is not None and previous.get("prompt_hash") != prompt_hash:
            report_lines.append("WARNING: prompt differs from the one used for the cached clips.")
        # the full graph stays in the project: to get back to it, drop
        # workflow.json (or the final mp4) onto ComfyUI
        _save_workflow_copy(project_dir, api_prompt, extra_pnginfo, fresh=(mode == "restart" or previous is None))
        _save_plan(project_dir, {
            "signature": signature, "prompt_hash": prompt_hash, "seed": int(seed),
            "steps": int(steps), "sampler": sampler_name, "scheduler": scheduler, "plan": plan,
            "source_offset_seconds": source.offset / float(FPS),
            "source_path": getattr(source, "path", None),
            "anchor_mode": anchor_mode, "audio_context": bool(audio_context),
        })

        # --- image references: encoded once ------------------------------------
        image_items, image_blocks = [], []
        for img in (ref_image_1, ref_image_2, ref_image_3):
            if img is None:
                continue
            h, w = int(img.shape[1]), int(img.shape[2])
            tw, th = _ref_image_canvas(w, h, ref_image_size, width, height)
            resized = _resize(img[:1], tw, th)
            z = vae.encode(resized)
            image_items.append({"type": "image", "data": resized})
            image_blocks.append({"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": z})

        cw, ch = _ref_video_canvas(source.width, source.height, ref_video_size, width, height)
        if use_ref_video:
            report_lines.append(f"source {source.label}; <Video 1> at {cw}x{ch}")
        if use_guide:
            # the guide must have exactly the latent grid of the generated clip
            report_lines.append(f"source {source.label}; guide latent at {int(width)}x{int(height)}")
        print("[H3 LongTake] start\n" + "\n".join(report_lines))

        # --- clip loop -----------------------------------------------------------
        pbar = comfy.utils.ProgressBar(len(clips))
        prev_video = prev_audio = None
        prev_tail = None          # last decoded frames of the previous clip (seam match)
        prev_tail_src = None      # (mp4, frames) to read them from when the previous clip is cached
        last_images = None
        rendered, skipped = [], []

        for c in clips:
            comfy.model_management.throw_exception_if_processing_interrupted()
            i = c["index"]
            latent_path, mp4_path = _clip_paths(project_dir, i)
            if os.path.isfile(latent_path) and os.path.isfile(mp4_path):
                saved = torch.load(latent_path, map_location="cpu")
                prev_video, prev_audio = saved["video"], saved["audio"]
                prev_tail, prev_tail_src = None, (mp4_path, int(saved.get("new", c["new"])))
                skipped.append(i)
                pbar.update(1)
                continue
            if i > 0 and prev_video is None:
                raise RuntimeError(f"H3 LongTake: clip {i - 1} is missing from the cache, cannot continue from {i}.")

            length, start, ctx, new = c["length"], c["src_start"], c["ctx"], c["new"]
            print(f"[H3 LongTake] clip {i + 1}/{len(clips)}: source {start}..{start + length - 1}, "
                  f"{length} frames, new {new}")

            # video reference = source slice; also written to disk to verify what
            # the model actually saw (clip_NNN_ref.mp4)
            ref_items = list(image_items)
            ref_blocks = list(image_blocks)
            audio_latent, ref_audio_t = None, 0
            frames_count, guide_frames = 0, 0
            z_video = z_guide = None
            frames = None
            if use_guide:
                # guide latent: the slice at the output canvas, keyframe at frame 0 for the whole clip
                gframes = source.frames_at(start, length, int(width), int(height))
                guide_frames = int(gframes.shape[0])
                try:
                    _write_mp4(os.path.join(project_dir, f"clip_{i:03d}_ref.mp4"), gframes, 18)
                except Exception as exc:
                    _LOG.warning("H3 LongTake: guide dump not written: %s", exc)
                z_guide = vae.encode(gframes)
                if use_ref_video and (cw, ch) == (int(width), int(height)):
                    frames = gframes
                else:
                    del gframes
            if use_ref_video:
                if frames is None:
                    frames = source.frames_at(start, length, cw, ch)
                if not use_guide:
                    try:
                        _write_mp4(os.path.join(project_dir, f"clip_{i:03d}_ref.mp4"), frames, 18)
                    except Exception as exc:
                        _LOG.warning("H3 LongTake: reference dump not written: %s", exc)
                z_video = vae.encode(frames)
                piece = source.audio_at(start, length) if use_source_audio else None
                if piece is not None:
                    dur = piece["waveform"].shape[-1] / float(piece["sample_rate"])
                    if dur + 1e-6 >= MIN_REF_AUDIO_SECONDS:
                        audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, piece)
                        ref_items.append({"type": "audio"})
                    else:
                        _LOG.warning("H3 LongTake: clip %d, source audio too short (%.2fs), ignored.", i, dur)
                sample_idx = list(range(0, int(frames.shape[0]), FPS // 2))
                frames_count = int(frames.shape[0])
                ref_items.append({"type": "video", "data": frames[sample_idx],
                                  "timestamps": [k / 2.0 for k in range(len(sample_idx))]})
                ref_blocks.append({
                    "kind": "video_audio" if ref_audio_t else "video",
                    "latent_t": int(z_video.shape[2]), "latent_h": ch // 16, "latent_w": cw // 16,
                    "ref_audio_t": ref_audio_t, "latent": z_video, "audio_latent": audio_latent,
                })
            del frames

            # With the DiT resident in VRAM the 25 GB Qwen can only stream from
            # RAM: the encode goes from ~1 min to ~10 min (measured). Evicting
            # costs nothing, the weights stay staged in RAM and are back in a second.
            if i > 0:
                comfy.model_management.unload_all_models()
                comfy.model_management.soft_empty_cache()
            tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
            positive = clip.encode_from_tokens_scheduled(tokens)
            positive = node_helpers.conditioning_set_values(positive, {"minimax_refs": ref_blocks})

            latent = _empty_av_latent(int(width), int(height), length)
            target_video, _ = _split_av(latent["samples"])
            nxt = None
            if mode == "redo_one":
                next_latent_path, _ = _clip_paths(project_dir, i + 1)
                if os.path.isfile(next_latent_path):
                    nxt = torch.load(next_latent_path, map_location="cpu")
                    print(f"[H3 LongTake] clip_{i:03d}: tail anchored to the head of clip_{i + 1:03d}")

            noise_mask = None
            if z_guide is not None:
                positive = node_helpers.conditioning_set_values(
                    positive, {"minimax_keyframes": [guide_keyframe(z_guide, target_video)]})
            if anchor_mode == "none":
                pass  # independent clips
            elif anchor_mode == "inpaint":
                anchor = _InpaintAnchor(latent)
                if i > 0:
                    anchor.head_from_tail(prev_video, prev_audio, C, bool(audio_context))
                if nxt is not None:
                    anchor.tail_from_head(nxt["video"], nxt["audio"], C, bool(audio_context))
                latent, noise_mask = anchor.latent(), anchor.mask()
            else:
                pos = "before" if before else "head"
                keyframes = []
                if i > 0:
                    keyframes += motion_context_keyframes(prev_video, prev_audio, C, target_video, bool(audio_context), pos)
                if nxt is not None:
                    keyframes += tail_context_keyframes(nxt["video"], nxt["audio"], C, target_video, bool(audio_context), pos)
                if keyframes:
                    keyframes = (positive[0][1].get("minimax_keyframes") or []) + keyframes
                    positive = node_helpers.conditioning_set_values(positive, {"minimax_keyframes": keyframes})
            del nxt

            kf = positive[0][1].get("minimax_keyframes") or []
            n_guide = 1 if z_guide is not None else 0
            print(f"[H3 LongTake] clip_{i:03d} conditioning: prompt {len(prompt)} chars, "
                  f"{len(image_blocks)} images, video {int(frames_count)} frames"
                  f"{' + audio' if ref_audio_t else ''}"
                  f"{f', guide {guide_frames} frames' if n_guide else ''}"
                  f", video anchors {sum(1 for k in kf if k.get('latent') is not None) - n_guide}"
                  f" / audio {sum(1 for k in kf if k.get('audio_latent') is not None)}"
                  f"{', inpaint mask' if noise_mask is not None else ''}, seed {int(seed) + i}")
            samples = _sample(model, positive, latent, int(seed) + i, sampler_name, scheduler, steps,
                              noise_mask=noise_mask)
            video_lat, audio_lat = _split_av(samples)
            video_lat, audio_lat = video_lat.cpu(), audio_lat.cpu()
            del positive, latent, samples, ref_blocks, z_video, z_guide

            # decode (with the VRAM freed from the DiT), trim the overlap, write the chunk
            comfy.model_management.unload_all_models()
            comfy.model_management.soft_empty_cache()
            images = vae.decode(video_lat)
            if images.ndim == 5:
                images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
            images = images.cpu()
            if i > 0 and seam_match != "off":
                if prev_tail is None and prev_tail_src is not None:
                    try:
                        prev_tail = _tail_frames_from_mp4(prev_tail_src[0], prev_tail_src[1], SEAM_ANALYSIS_FRAMES)
                    except Exception as exc:
                        _LOG.warning("H3 LongTake: previous clip tail not read, seam match skipped: %s", exc)
                if prev_tail is not None and tuple(prev_tail.shape[1:]) == tuple(images.shape[1:]):
                    images = seam_match_fn(images, ctx, seam_match, ref=prev_tail)
            images = images[ctx: ctx + new].contiguous()
            _write_mp4(mp4_path + ".tmp.mp4", images, chunk_crf)
            torch.save({"video": video_lat, "audio": audio_lat, "length": length, "ctx": ctx, "new": new},
                       latent_path)
            os.replace(mp4_path + ".tmp.mp4", mp4_path)

            prev_video, prev_audio = video_lat, audio_lat
            prev_tail, prev_tail_src = images[-SEAM_ANALYSIS_FRAMES:].clone(), None
            last_images = images
            rendered.append(i)
            pbar.update(1)

            # the models stay in ComfyUI's cache (reloading them would cost minutes
            # per clip); only the clip tensors and the blocks reserved by the CUDA
            # allocator are released here
            del images, video_lat, audio_lat
            gc.collect()
            comfy.model_management.soft_empty_cache()

        report_lines.append(f"rendered: {rendered or '-'} | already cached: {skipped or '-'}")
        report_lines.append(f"folder: {project_dir}")

        # preview in the node: every chunk present, concatenated without re-encoding
        ui = {}
        chunks = _existing_chunks(project_dir)
        if chunks:
            preview_path = os.path.join(project_dir, "preview.mp4")
            try:
                _concat_mp4(chunks, preview_path, metadata=_project_metadata(project_dir, api_prompt, extra_pnginfo))
                ui = _ui_video(preview_path)
                report_lines.append(f"preview: {len(chunks)} clips in preview.mp4")
            except Exception as exc:  # the preview must never fail the run
                _LOG.warning("H3 LongTake: preview not created: %s", exc)

        report = "\n".join(report_lines)
        print("[H3 LongTake]\n" + report)
        out = last_images if last_images is not None else placeholder
        ui["text"] = [report]
        return {"ui": ui, "result": (out, project_dir, report)}


class H3LongTakeStitch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "project_name": ("STRING", {"default": "longtake"}),
                "output_name": ("STRING", {"default": "longtake_final"}),
            },
            "optional": {
                "audio_file": ([AUDIO_AUTO, AUDIO_NONE] + _list_input_videos()[1:], {"video_upload": True,
                               "tooltip": "auto: audio of the source used by the Render (saved in the project). "
                                          "Or pick/upload a video to take the audio from."}),
                "source_audio": ("AUDIO", {"tooltip": "Alternatively: the original audio as AUDIO (takes priority over auto)."}),
                "expected_clips": ("INT", {"default": 0, "min": 0, "max": 9999,
                                           "tooltip": "0 = use the plan saved in the project."}),
                "project_dir": ("STRING", {"forceInput": True,
                                "tooltip": "Connect the Render's project_dir output: replaces project_name and runs the Stitch after the Render."}),
            },
            "hidden": {"api_prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("path", "report")
    FUNCTION = "stitch"
    CATEGORY = "H3 LongTake"
    OUTPUT_NODE = True
    DESCRIPTION = "Concatenates the chunks of an H3 LongTake project (no re-encoding) and puts the original audio back."

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, audio_file=None):
        if (audio_file and audio_file not in (AUDIO_AUTO, AUDIO_NONE, NO_FILE)
                and not folder_paths.exists_annotated_filepath(audio_file)):
            return f"Video not found in input/: {audio_file}"
        return True

    def stitch(self, project_name, output_name, audio_file=AUDIO_AUTO, source_audio=None, expected_clips=0,
               project_dir=None, api_prompt=None, extra_pnginfo=None):
        if isinstance(project_dir, str) and project_dir.strip():
            project_dir = project_dir.strip()
            if not os.path.isdir(project_dir):
                raise RuntimeError(f"H3 LongTake: project folder not found: {project_dir}")
        else:
            project_dir = _project_dir(project_name)
        plan = _load_plan(project_dir)
        if expected_clips <= 0:
            if plan is None:
                # Nothing rendered yet (dry run, or the Render has not run): report instead of failing,
                # so a workflow with the Stitch left unmuted still completes.
                report = (f"nothing to stitch: no {PLAN_FILE} in {project_dir}. Run the Render with dry_run=false "
                          "first (or set expected_clips).")
                print("[H3 LongTake] " + report)
                return {"ui": {"text": [report]}, "result": ("", report)}
            expected_clips = len(plan["plan"]["clips"])

        files = []
        for i in range(int(expected_clips)):
            _, mp4_path = _clip_paths(project_dir, i)
            if not os.path.isfile(mp4_path):
                raise RuntimeError(f"H3 LongTake: {os.path.basename(mp4_path)} is missing. Generate all the clips first.")
            files.append(mp4_path)

        out_dir = folder_paths.get_output_directory()
        out_path = os.path.join(out_dir, f"{output_name}.mp4")
        n = 1
        while os.path.exists(out_path):
            out_path = os.path.join(out_dir, f"{output_name}_{n:03d}.mp4")
            n += 1

        # the audio starts from the same point of the source used by the Render (start_seconds)
        offset = float((plan or {}).get("source_offset_seconds", 0.0) or 0.0)
        audio_args = []
        stdin_bytes = None
        audio_note = None
        audio_path = None
        if audio_file in (AUDIO_AUTO, NO_FILE, "", None):
            # NO_FILE: workflows saved with the previous version
            if source_audio is None:
                audio_path = (plan or {}).get("source_path")
                if audio_path and not os.path.isfile(audio_path):
                    _LOG.warning("H3 LongTake: project source not found (%s), assembling without audio.", audio_path)
                    audio_path = None
        elif audio_file != AUDIO_NONE:
            audio_path = folder_paths.get_annotated_filepath(audio_file)
            source_audio = None
        if audio_path:
            audio_args += ["-ss", f"{offset:.6f}", "-i", audio_path]
            audio_note = f"audio from {os.path.basename(audio_path)}" + (f" from {offset:.2f}s" if offset else "")
        elif source_audio is not None:
            wave = source_audio["waveform"][0]  # [C, L]
            sr = int(source_audio["sample_rate"])
            wave = wave[:, int(round(offset * sr)):]
            stdin_bytes = wave.T.contiguous().to(torch.float32).cpu().numpy().tobytes()
            audio_args += ["-f", "f32le", "-ar", str(sr), "-ac", str(int(wave.shape[0])), "-i", "pipe:0"]
            audio_note = "audio from source_audio"
        if audio_note:
            audio_args += ["-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-b:a", "192k"]
            if plan is not None:
                # the video rules: the audio is cut to its duration, never the other way round
                total_frames = sum(int(c.get("new", 0)) for c in plan["plan"]["clips"][:int(expected_clips)])
                if total_frames > 0:
                    audio_args += ["-t", f"{total_frames / float(FPS):.6f}"]
        metadata = _project_metadata(project_dir, api_prompt, extra_pnginfo)
        _concat_mp4(files, out_path, audio_args, stdin_bytes, metadata=metadata)

        lines = [f"assembled {len(files)} clips -> {out_path}"]
        if metadata:
            lines.append("workflow embedded in the mp4 (drop it onto ComfyUI to reopen the project)")
        if audio_note:
            lines.append(audio_note)
        report = "\n".join(lines)
        print("[H3 LongTake Stitch] " + report)
        ui = _ui_video(out_path)
        ui["text"] = [report]
        return {"ui": ui, "result": (out_path, report)}


NODE_CLASS_MAPPINGS = {
    "H3LongTakeRender": H3LongTakeRender,
    "H3LongTakeStitch": H3LongTakeStitch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3LongTakeRender": "H3 LongTake Render",
    "H3LongTakeStitch": "H3 LongTake Stitch",
}
