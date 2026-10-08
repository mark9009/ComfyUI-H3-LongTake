// H3 LongTake FaceRefine calls the nodes of ComfyUI-H3-FaceRefine from inside: if that pack is not installed the
// graph still loads, so this shows a dialog with the link. Link as in facerefine_node.py.
import { app } from "../../scripts/app.js";

const URL = "https://github.com/Carasibana/ComfyUI-H3-FaceRefine";
const NEEDED = ["H3FaceTrackCrop", "H3InjectVideoLatent", "H3PerFrameDenoise", "H3FaceStitch"];
let shown = false;

function missingNodes() {
    const types = globalThis.LiteGraph?.registered_node_types || {};
    return NEEDED.filter((n) => !(n in types));
}

function warn() {
    if (shown) return;
    const missing = missingNodes();
    if (!missing.length) return;
    shown = true;
    const html =
        "<h3>H3 LongTake FaceRefine: ComfyUI-H3-FaceRefine is missing</h3>" +
        "<p>This node uses the FaceRefine nodes (Carasibana), which are not installed: " +
        missing.join(", ") + ".</p>" +
        `<p>Install it from the Manager or from <a href="${URL}" target="_blank">${URL}</a>, ` +
        "then restart ComfyUI.</p>";
    app.ui.dialog.show(html);
}

app.registerExtension({
    name: "H3LongTake.facerefine_check",
    nodeCreated(node) {
        if (node.comfyClass === "H3LongTakeFaceRefine") setTimeout(warn, 500);
    },
});
