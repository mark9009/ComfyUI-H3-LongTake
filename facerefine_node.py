"""
H3 LongTake FaceRefine: ComfyUI-H3-FaceRefine (Carasibana) clip by clip on a LongTake project.

It goes between the Refine (or Render / Image to Video) and the Stitch, like the Refine: it reads the chunks of
a project and writes <project>/fr with the same plan, which the Stitch assembles like the original.

For every clip: face tracked and cropped (H3FaceTrackCrop), crop regenerated with H3 img2img with the face as
<Picture 1> (H3InjectVideoLatent + H3PerFrameDenoise), pasted back with a feathered ellipse (H3FaceStitch).
Measured:

  * recipe: denoise 0.25 (with shift 12, 0.25 is already sigma 0.80; above it the head drifts against the
    body), large faces 0.35, ellipse without SAM with feather 24, smooth_window 11, 4 Turbo steps;
  * 17k+5 grid: the chunks after the first have 119 frames; FaceRefine fills the missing tokens with an empty
    latent and the last real frames come out as coloured noise. Here the clip is extended with 16 copies of the
    last frame (the most the grid can be short of) and trimmed afterwards;
  * only where it helps: a clip whose median face is above max_face_px is copied as is. There FaceRefine
    enlarges the face 1.0-1.4x for no gain, and at 1 MP it brought ComfyUI down;
  * wide shot at 0.55 MP (faces 45-100 px): identity 0.25-0.33 -> 0.63-0.67, face steady on the head
    (slip < 1.5% of the side).
"""

import gc
import os

import numpy as np
import torch

import comfy.model_management
import comfy.nested_tensor
import comfy.samplers
import comfy.utils
import node_helpers
import nodes

from .longtake_nodes import (
    CLIP_AUDIO_ARGS, I2V_ENGINE, PLAN_FILE, _clip_paths, _concat_mp4, _delete_clips_from, _empty_av_latent,
    _existing_chunks, _frames_from_mp4, _load_plan, _partial_sigmas, _project_dir, _ref_image_canvas, _resize,
    _sample_sigmas, _save_plan, _save_workflow_copy, _split_av, _ui_video, _write_mp4,
)

FACEREFINE_SUBDIR = "fr"
GRID_PAD = 16   # frames repeated at the end: always enough to reach the next 17k+5 grid size
FACEREFINE_NODES = ("H3FaceTrackCrop", "H3InjectVideoLatent", "H3PerFrameDenoise", "H3FaceStitch")
FACEREFINE_URL = "https://github.com/Carasibana/ComfyUI-H3-FaceRefine"   # also in web/facerefine_check.js


def _facerefine_classes():
    missing = [n for n in FACEREFINE_NODES if n not in nodes.NODE_CLASS_MAPPINGS]
    if missing:
        raise RuntimeError("H3 LongTake FaceRefine: needs the ComfyUI-H3-FaceRefine pack (Carasibana). Install it from "
                           f"the Manager or {FACEREFINE_URL} and restart ComfyUI. Missing: " + ", ".join(missing))
    return [nodes.NODE_CLASS_MAPPINGS[n]() for n in FACEREFINE_NODES]


def _face_detector(tracker):
    """The YOLO face detector file as listed in the tracker's menu (with or without a subfolder)."""
    options = tracker.INPUT_TYPES()["required"]["detector"][0]
    for name in options:
        if "face_yolov8m" in name:
            return name
    faces = [n for n in options if "face" in n.lower()]
    if not faces:
        raise RuntimeError("H3 LongTake FaceRefine: no YOLO face detector (e.g. face_yolov8m.pt) in models/ultralytics.")
    return faces[0]


def _blend_to_original(refined, original, head, tail):
    """In the first `head` and last `tail` frames the result fades (cosine) to the original: at the join it equals
    the original, which is continuous across clips, so the refined face comes in gradually instead of popping.
    Outside the face refined and original are the same, so the whole frame is enough."""
    out = refined.clone()
    n = int(out.shape[0])
    for k in range(min(int(head), n)):
        w = 0.5 - 0.5 * np.cos(np.pi * (k + 1) / (head + 1))
        out[k] = original[k] * (1.0 - w) + refined[k] * w
    for k in range(min(int(tail), n)):
        j = n - 1 - k
        w = 0.5 - 0.5 * np.cos(np.pi * (k + 1) / (tail + 1))
        out[j] = original[j] * (1.0 - w) + refined[j] * w
    return out


def _free():
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    gc.collect()


class H3LongTakeFaceRefine:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "H3 ref2va with the Turbo (8-step v1.0 recommended): the ref2va is needed "
                                               "because the face goes in as <Picture 1>, even if the project was made with the fl2va."}),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "face_image": ("IMAGE", {"tooltip": "A close-up of the face, not the whole photo."}),
                "project_name": ("STRING", {"default": "longtake", "tooltip": "An already rendered project. The result "
                                            "goes to <project>/fr (in the Stitch: project_name = 'name/fr')."}),
                "description": ("STRING", {"multiline": True, "default": "",
                                "tooltip": "The subject's look (hair, outfit): helps not to change person."}),
                "denoise": ("FLOAT", {"default": 0.25, "min": 0.05, "max": 1.0, "step": 0.05,
                            "tooltip": "0.25-0.35. With H3's shift 12, 0.25 already rewrites the face; above it the head "
                                       "drifts against the body."}),
                "steps": ("INT", {"default": 4, "min": 1, "max": 50}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "max_face_px": ("INT", {"default": 150, "min": 0, "max": 2000,
                                "tooltip": "Clips whose median face is taller than this (px) are copied without FaceRefine. "
                                           "0 = refine every clip."}),
                "mode": (["continue", "restart"], {"default": "continue", "tooltip": "continue: skips the clips already done."}),
            },
            "optional": {
                "project_dir": ("STRING", {"forceInput": True, "tooltip": "Connect project_dir from the Render/I2V/Refine."}),
                "large_face_multiplier": ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.05,
                                          "tooltip": "Fraction of the denoise on large faces (the author's default)."}),
                "smooth_window": ("INT", {"default": 11, "min": 1, "max": 201, "step": 2,
                                  "tooltip": "Box smoothing: lower if the camera moves."}),
                "crop_factor": ("FLOAT", {"default": 2.5, "min": 1.5, "max": 4.0, "step": 0.1}),
                "feather": ("INT", {"default": 24, "min": 0, "max": 128}),
                "chunk_crf": ("INT", {"default": 10, "min": 0, "max": 30}),
                "seam_blend_frames": ("INT", {"default": 8, "min": 0, "max": 48,
                                      "tooltip": "At the inner joins the refined clip fades to the original over these "
                                                 "frames, so the face does not pop when going from a copied (or separately "
                                                 "refined) clip to a refined one. 0 = off."}),
            },
            "hidden": {"api_prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("last_clip", "project_dir", "report")
    FUNCTION = "refine_faces"
    CATEGORY = "H3 LongTake"
    OUTPUT_NODE = True
    DESCRIPTION = ("Regenerates small faces clip by clip with ComfyUI-H3-FaceRefine and writes <project>/fr; "
                   "assemble with H3 LongTake Stitch. Skips the clips whose face is already large.")

    def refine_faces(self, model, clip, vae, face_image, project_name, description, denoise, steps, seed,
                     max_face_px, mode, project_dir=None, large_face_multiplier=0.35, smooth_window=11,
                     crop_factor=2.5, feather=24, chunk_crf=10, seam_blend_frames=8, api_prompt=None, extra_pnginfo=None):
        tracker, inject, per_frame, stitch = _facerefine_classes()
        detector = _face_detector(tracker)
        placeholder = torch.zeros(1, 64, 64, 3)
        if isinstance(project_dir, str) and not project_dir.strip():
            # project_dir linked but empty: the Render is in dry_run, there is nothing to refine yet
            report = "nothing to refine: the Render is in dry_run (turn it off to generate the clips)."
            print("[H3 LongTake FaceRefine] " + report)
            return {"ui": {"text": [report]}, "result": (placeholder, "", report)}
        if isinstance(project_dir, str) and project_dir.strip():
            src_dir = project_dir.strip()
        else:
            src_dir = _project_dir(project_name)
        src_plan = _load_plan(src_dir)
        if src_plan is None:
            raise ValueError(f"H3 LongTake FaceRefine: no {PLAN_FILE} in {src_dir}: render the project first.")
        clips = src_plan["plan"]["clips"]
        i2v = src_plan["signature"].get("engine") == I2V_ENGINE
        out_dir = os.path.join(src_dir, FACEREFINE_SUBDIR)
        os.makedirs(out_dir, exist_ok=True)
        if mode == "restart":
            _delete_clips_from(out_dir, 0)
        _save_workflow_copy(out_dir, api_prompt, extra_pnginfo, node_class="H3LongTakeFaceRefine",
                            fresh=(mode == "restart" or _load_plan(out_dir) is None))
        _save_plan(out_dir, dict(src_plan, facerefine={"denoise": float(denoise), "steps": int(steps), "seed": int(seed),
                                                        "max_face_px": int(max_face_px), "refined_from": src_dir}))
        look = (str(description).strip() + " " if str(description).strip() else "") + (
            "The face is the same person shown in <Picture 1>. Sharp facial detail, natural skin texture, "
            "clean eyes, consistent lighting, the same expression and head pose as the video.")
        report_lines = [f"project {src_dir}: {len(clips)} clips -> {out_dir}, fade at the joins "
                        f"{int(seam_blend_frames)} frames" if seam_blend_frames else
                        f"project {src_dir}: {len(clips)} clips -> {out_dir}, no fade at the joins",
                        f"denoise {denoise:g}, {int(steps)} steps, large faces x{large_face_multiplier:g}, "
                        f"skips clips whose median face is > {int(max_face_px)} px" if max_face_px else
                        f"denoise {denoise:g}, {int(steps)} steps, large faces x{large_face_multiplier:g}, every clip"]
        print("[H3 LongTake FaceRefine] start\n" + "\n".join(report_lines))

        pbar = comfy.utils.ProgressBar(len(clips))
        last_images = None
        refined, copied, skipped = [], [], []
        for c in clips:
            comfy.model_management.throw_exception_if_processing_interrupted()
            i = c["index"]
            _, src_mp4 = _clip_paths(src_dir, i)
            _, dst_mp4 = _clip_paths(out_dir, i)
            if not os.path.isfile(src_mp4):
                raise RuntimeError(f"H3 LongTake FaceRefine: {os.path.basename(src_mp4)} is missing from the project.")
            if os.path.isfile(dst_mp4):
                skipped.append(i)
                pbar.update(1)
                continue
            _free()
            frames = _frames_from_mp4(src_mp4)
            n = int(frames.shape[0])
            padded = torch.cat([frames, frames[-1:].repeat(GRID_PAD, 1, 1, 1)], dim=0)
            del frames
            crops, transform, _, track_report, cw, ch, length = tracker.run(
                padded, detector, 0.35, float(crop_factor), 768, 768, "auto_capped_768", int(smooth_window), 51,
                "gaussian", "per_frame", identity_track=True, identity_threshold=0.28)
            boxes = transform["boxes"]
            face_px = float(np.median([b[3] for b in boxes])) / float(transform.get("crop_factor", crop_factor)) if boxes else 0.0
            if not boxes or (max_face_px and face_px > max_face_px):
                why = "no face" if not boxes else f"median face {face_px:.0f} px"
                print(f"[H3 LongTake FaceRefine] clip {i}: {why}, copied as is")
                _write_mp4(dst_mp4 + ".tmp.mp4", padded[:n], chunk_crf, audio_from=src_mp4)
                os.replace(dst_mp4 + ".tmp.mp4", dst_mp4)
                copied.append((i, round(face_px)))
                last_images = padded[:n]
                del padded, crops
                pbar.update(1)
                continue
            print(f"[H3 LongTake FaceRefine] clip {i}: median face {face_px:.0f} px, {n} frames "
                  f"(+{GRID_PAD} for the grid), canvas {cw}x{ch}\n{track_report}")

            # conditioning: the face as <Picture 1>, empty AV latent at the crop canvas
            tw, th = _ref_image_canvas(int(face_image.shape[2]), int(face_image.shape[1]), "match", int(cw), int(ch))
            ref_img = _resize(face_image[:1], tw, th)
            tokens = clip.tokenize(look, minimax_ref_items=[{"type": "image", "data": ref_img}])
            positive = clip.encode_from_tokens_scheduled(tokens)
            positive = node_helpers.conditioning_set_values(positive, {"minimax_refs": [
                {"kind": "image", "latent_h": th // 16, "latent_w": tw // 16, "latent": vae.encode(ref_img)}]})
            av_latent = _empty_av_latent(int(cw), int(ch), int(length))
            av_latent, _ = inject.run(av_latent, crops, vae)
            av_latent, _, patched = per_frame.run(model, av_latent, transform, 1.0, float(large_face_multiplier),
                                                  30.0, 120.0, 1.0, 9, scale_mode="absolute_px")
            _free()
            sigmas = _partial_sigmas(patched, "simple", int(steps), float(denoise))
            samples = _sample_sigmas(patched, positive, av_latent, int(seed) + i, "euler", sigmas,
                                     noise_mask=av_latent.get("noise_mask"))
            video_lat, _ = _split_av(samples)
            del samples, av_latent, positive, crops
            _free()
            out = vae.decode(video_lat.cpu())
            if out.ndim == 5:
                out = out.reshape(-1, out.shape[-3], out.shape[-2], out.shape[-1])
            out = out.cpu()
            del video_lat
            (merged,) = stitch.run(padded, out, transform, "face_ellipse", 16, int(feather), 1.0, 1.0,
                                   undetected_frames="fade_out", feather_scales_with_crop=False)
            merged = merged[:n].contiguous()
            if seam_blend_frames:
                last = i == clips[-1]["index"]
                merged = _blend_to_original(merged, padded[:n], 0 if i == 0 else int(seam_blend_frames),
                                            0 if last else int(seam_blend_frames))
            _write_mp4(dst_mp4 + ".tmp.mp4", merged, chunk_crf, audio_from=src_mp4)
            os.replace(dst_mp4 + ".tmp.mp4", dst_mp4)
            refined.append((i, round(face_px)))
            last_images = merged
            del out, merged, padded
            pbar.update(1)
            _free()

        report_lines.append("refined (clip, face px): " + (str(refined) if refined else "-"))
        report_lines.append("copied with a large face: " + (str(copied) if copied else "-"))
        report_lines.append("already done: " + (str(skipped) if skipped else "-"))
        ui = {}
        chunks = _existing_chunks(out_dir)
        if chunks:
            preview_path = os.path.join(out_dir, "preview.mp4")
            try:
                _concat_mp4(chunks, preview_path, CLIP_AUDIO_ARGS if i2v else None)
                ui = _ui_video(preview_path)
            except Exception as exc:
                report_lines.append(f"preview not created: {exc}")
        report = "\n".join(report_lines)
        print("[H3 LongTake FaceRefine]\n" + report)
        ui["text"] = [report]
        return {"ui": ui, "result": (last_images if last_images is not None else placeholder, out_dir, report)}


NODE_CLASS_MAPPINGS = {"H3LongTakeFaceRefine": H3LongTakeFaceRefine}
NODE_DISPLAY_NAME_MAPPINGS = {"H3LongTakeFaceRefine": "H3 LongTake FaceRefine (small faces)"}
