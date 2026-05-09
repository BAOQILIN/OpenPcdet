import argparse
import numpy as np
import onnx
import onnxruntime as ort


def parse_args():
    parser = argparse.ArgumentParser(description='Verify ONNX model')
    parser.add_argument('--onnx', type=str, required=True, help='ONNX model path')
    args = parser.parse_args()
    return args


def verify_onnx(args):
    print(f'Loading ONNX model from {args.onnx}...')

    # Load and check ONNX model
    onnx_model = onnx.load(args.onnx)
    onnx.checker.check_model(onnx_model)
    print('✓ ONNX model structure is valid')

    # Print model info
    print('\n=== Model Information ===')
    print(f'IR Version: {onnx_model.ir_version}')
    print(f'Producer: {onnx_model.producer_name} {onnx_model.producer_version}')
    print(f'Opset Version: {onnx_model.opset_import[0].version}')

    # Print inputs
    print('\n=== Model Inputs ===')
    for input_tensor in onnx_model.graph.input:
        print(f'Name: {input_tensor.name}')
        shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' for dim in input_tensor.type.tensor_type.shape.dim]
        print(f'  Shape: {shape}')
        print(f'  Type: {input_tensor.type.tensor_type.elem_type}')

    # Print outputs
    print('\n=== Model Outputs ===')
    for output_tensor in onnx_model.graph.output:
        print(f'Name: {output_tensor.name}')
        shape = [dim.dim_value if dim.dim_value > 0 else 'dynamic' for dim in output_tensor.type.tensor_type.shape.dim]
        print(f'  Shape: {shape}')
        print(f'  Type: {output_tensor.type.tensor_type.elem_type}')

    # Test inference with ONNX Runtime
    print('\n=== Testing ONNX Runtime Inference ===')
    try:
        session = ort.InferenceSession(args.onnx, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        print(f'✓ ONNX Runtime session created successfully')
        print(f'  Providers: {session.get_providers()}')

        # Create dummy inputs
        num_voxels = 1000
        max_points_per_voxel = 20
        num_features = 5

        dummy_inputs = {
            'voxels': np.random.randn(num_voxels, max_points_per_voxel, num_features).astype(np.float32),
            'voxel_coords': np.random.randint(0, 256, (num_voxels, 4)).astype(np.int32),  # Type 6 = int32
            'voxel_num_points': np.random.randint(1, max_points_per_voxel + 1, (num_voxels,)).astype(np.int64)  # Type 7 = int64
        }

        print('\n=== Running Inference ===')
        outputs = session.run(None, dummy_inputs)

        print(f'✓ Inference successful!')
        print(f'\nOutput shapes:')
        for i, output in enumerate(outputs):
            print(f'  Output {i}: {output.shape}, dtype: {output.dtype}')

    except Exception as e:
        print(f'✗ ONNX Runtime inference failed: {e}')
        return False

    print('\n✓ All verification passed!')
    return True


if __name__ == '__main__':
    args = parse_args()
    verify_onnx(args)
