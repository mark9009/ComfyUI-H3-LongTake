// Stitch, Refine and FaceRefine: with the `project_dir` input linked the node works on the folder it
// receives and ignores `project_name` (see longtake_nodes.py and facerefine_node.py). Here the field is greyed out
// and its label says so, so the old name still shown does not mislead; unlinking restores it.
// The value stays in the workflow.
import { app } from "../../scripts/app.js";

const NODES = ["H3LongTakeStitch", "H3LongTakeRefine", "H3LongTakeFaceRefine"];
const LABEL = "project_name (ignored: uses project_dir)";

function sync(node) {
    const widget = node.widgets?.find((w) => w.name === "project_name");
    const input = node.inputs?.find((i) => i.name === "project_dir");
    if (!widget || !input) return;
    const linked = input.link != null;
    widget.disabled = linked;
    widget.label = linked ? LABEL : undefined;
    node.updateComputedDisabled?.();
    node.setDirtyCanvas?.(true, true);
}

app.registerExtension({
    name: "H3LongTake.project_dir_link",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODES.includes(nodeData.name)) return;
        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (...args) {
            const r = onConnectionsChange?.apply(this, args);
            sync(this);
            return r;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function (...args) {
            const r = onConfigure?.apply(this, args);
            // the loaded graph's links are ready only after every node is configured
            setTimeout(() => sync(this), 0);
            return r;
        };
    },
    nodeCreated(node) {
        if (NODES.includes(node.comfyClass)) sync(node);
    },
});
