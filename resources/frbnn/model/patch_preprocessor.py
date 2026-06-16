"""
Patch frbnn_preprocessor.onnx for batch > 1 correctness.

Root cause: ScatterND_3 writes the global RFI threshold only to batch index 0,
leaving indices 1+ at zero.  Greater(input, 0) is always True for positive
radio data, so every beam except beam 0 gets fully masked → 0/0 NaN.

Fix: replace ScatterND_3 with Expand(/Reshape_9_output_0, Shape(modelInput)),
broadcasting the computed (1, freq, time) threshold to all batch positions.
Beam 0 result is numerically identical before and after the patch.
"""

import sys
import onnx
from onnx import helper, shape_inference

MODEL = "frbnn_preprocessor.onnx"

m = onnx.load(MODEL)

new_nodes = []
patched = 0
for node in m.graph.node:
    if node.name == "/ScatterND_3":
        new_nodes.append(helper.make_node(
            "Expand",
            inputs=["/Reshape_9_output_0", "/Shape_31_output_0"],
            outputs=["/ScatterND_3_output_0"],
            name="/Expand_ScatterND3_fix",
        ))
        patched += 1
    else:
        new_nodes.append(node)

if patched == 0:
    print(f"WARNING: /ScatterND_3 not found in {MODEL} — already patched or wrong model?")
    sys.exit(0)

del m.graph.node[:]
m.graph.node.extend(new_nodes)

m = shape_inference.infer_shapes(m)
onnx.checker.check_model(m)
onnx.save(m, MODEL)
print(f"Patched {MODEL}: ScatterND_3 → Expand (global RFI threshold broadcast to all beams)")
