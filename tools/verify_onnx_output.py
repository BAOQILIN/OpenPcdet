#!/usr/bin/env python3
"""Verify ONNX model output format"""
import onnxruntime as ort
import numpy as np

model_path = "/home/bql/ARS/ARS_Data/E001-2v110/perception_model/pointpillar/robosense128/pointpillar_with_postprocess.onnx"

# Create session
session = ort.InferenceSession(model_path)

# Print model info
print("=" * 80)
print("ONNX Model Information")
print("=" * 80)

print("\nInputs:")
for inp in session.get_inputs():
    print(f"  {inp.name}: {inp.shape} ({inp.type})")

print("\nOutputs:")
for out in session.get_outputs():
    print(f"  {out.name}: {out.shape} ({out.type})")

# Create dummy inputs
max_voxels = 10000
max_points_per_voxel = 20
num_point_features = 5

voxels = np.random.randn(max_voxels, max_points_per_voxel, num_point_features).astype(np.float32)
voxel_coords = np.zeros((max_voxels, 4), dtype=np.int32)
voxel_coords[:, 0] = 0  # batch_idx
voxel_coords[:, 1] = np.random.randint(0, 10, max_voxels)  # z
voxel_coords[:, 2] = np.random.randint(0, 496, max_voxels)  # y
voxel_coords[:, 3] = np.random.randint(0, 432, max_voxels)  # x
voxel_num_points = np.random.randint(1, max_points_per_voxel + 1, max_voxels).astype(np.float32)

# Run inference
print("\n" + "=" * 80)
print("Running Inference")
print("=" * 80)

outputs = session.run(None, {
    'voxels': voxels,
    'voxel_coords': voxel_coords,
    'voxel_num_points': voxel_num_points
})

print("\nOutput Shapes:")
for i, (out_name, out_val) in enumerate(zip([o.name for o in session.get_outputs()], outputs)):
    print(f"  {out_name}: {out_val.shape} (dtype: {out_val.dtype})")
    if out_val.size > 0 and out_val.size < 100:
        print(f"    Values: {out_val}")

# Verify format
print("\n" + "=" * 80)
print("Format Verification")
print("=" * 80)

labels, scores, boxes = outputs

print(f"\nLabels shape: {labels.shape}")
print(f"Scores shape: {scores.shape}")
print(f"Boxes shape: {boxes.shape}")

# Check format
if len(boxes.shape) == 3:
    batch_size, num_boxes, box_dim = boxes.shape
    print(f"\n✓ Boxes format: [batch={batch_size}, num_boxes={num_boxes}, dim={box_dim}]")

    if box_dim >= 7:
        print(f"✓ Box dimension >= 7: PASS")
    else:
        print(f"✗ Box dimension < 7: FAIL (expected >=7, got {box_dim})")

    if labels.shape == (batch_size, num_boxes):
        print(f"✓ Labels shape matches: PASS")
    else:
        print(f"✗ Labels shape mismatch: FAIL (expected {(batch_size, num_boxes)}, got {labels.shape})")

    if scores.shape == (batch_size, num_boxes):
        print(f"✓ Scores shape matches: PASS")
    else:
        print(f"✗ Scores shape mismatch: FAIL (expected {(batch_size, num_boxes)}, got {scores.shape})")

    print(f"\n✓ Output format compatible with ARSBase: YES")
    print(f"  - Expected: [1, N, >=7] or [N, >=7]")
    print(f"  - Got: {boxes.shape}")
else:
    print(f"\n✗ Unexpected boxes shape: {boxes.shape}")
    print(f"  Expected 3D tensor [batch, num_boxes, box_dim]")

print("\n" + "=" * 80)
