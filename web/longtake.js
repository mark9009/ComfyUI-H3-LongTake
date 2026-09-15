// H3 LongTake: when `source_role` changes, the `prompt` field is filled with the
// role's template, but only if it is empty or still holds a template (never on top
// of a prompt written by the user). Keep in sync with PROMPT_TEMPLATES in longtake_nodes.py.
import { app } from "../../scripts/app.js";

const TEMPLATES = {
    "reference": "<Video 1> provides the full performance, motion, timing and camera. " +
                 "<Picture 1> defines the identity and appearance of the subject.",
    "guide": "style_transfer: Re-render this video with the style of <Picture 1>: " +
             "bold graphic style, vibrant flat colours, clean outlines, flat shading. " +
             "Keep the original scene, background and framing exactly as in the video; do not add any scenery from <Picture 1>.",
    "guide_retexture": "retexture: change the dress to red silk, glossy finish, keeping everything else unchanged.",
    "guide+reference": "style_transfer: Re-render this video with the style of <Picture 1>: " +
                       "painterly brushwork, warm palette, soft shading, visible canvas texture.",
};

function isTemplate(text) {
    const t = (text || "").trim();
    return t === "" || Object.values(TEMPLATES).some((v) => v.trim() === t);
}

app.registerExtension({
    name: "H3LongTake.prompt_templates",
    nodeCreated(node) {
        if (node.comfyClass !== "H3LongTakeRender") return;
        const role = node.widgets?.find((w) => w.name === "source_role");
        const prompt = node.widgets?.find((w) => w.name === "prompt");
        if (!role || !prompt) return;
        const prev = role.callback;
        role.callback = function (value, ...rest) {
            const r = prev?.apply(this, [value, ...rest]);
            const tpl = TEMPLATES[value];
            if (tpl && isTemplate(prompt.value)) {
                prompt.value = tpl;
                prompt.callback?.(tpl);
                node.setDirtyCanvas?.(true, true);
            }
            return r;
        };
    },
});
