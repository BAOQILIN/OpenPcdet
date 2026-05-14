import argparse
import numpy as np
import onnx
import onnxruntime as ort


def parse_args():
    parser = argparse.ArgumentParser(description='Verify exported PointPillar ONNX model')
    parser.add_argument('--onnx', type=str, required=True, help='Path to ONNX model')
    parser.add_argument('--num_voxels', type=int, default=1000,
                        help='Number of voxels for dummy inference test')
    args = parser.parse_args()
    return args


def verify_onnx(args):
    print(f'Loading ONNX model from {args.onnx}...')

    onnx_model = onnx.load(args.onnx)
    onnx.checker.check_model(onnx_model)
    print('ONNX model structure is valid')

    # Print model info
    print('\n=== Model Information ===')
    print(f'IR Version: {onnx_model.ir_version}')
    opset = onnx_model.opset_import[0]
    print(f'Opset Version: {opset.version}')
    print(f'Producer: {onnx_model.producer_name} {onnx_model.producer_version}')

    # Print inputs
    print('\n=== Model Inputs ===')
    for t in onnx_model.graph.input:
        shape = [d.dim_value if d.dim_value else 'dynamic' for d in t.type.tensor_type.shape.dim]
        print(f'  {t.name}: shape={shape}, type={t.type.tensor_type.elem_type}')

    # Print outputs
    print('\n=== Model Outputs ===')
    for t in onnx_model.graph.output:
        shape = [d.dim_value if d.dim_value else 'dynamic' for d in t.type.tensor_type.shape.dim]
        print(f'  {t.name}: shape={shape}, type={t.type.tensor_type.elem_type}')

    # Test inference with ONNX Runtime
    print('\n=== Testing ONNX Runtime Inference ===')
    try:
        session = ort.InferenceSession(
            args.onnx,
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        print(f'Session created, providers: {session.get_providers()}')

        M = args.num_voxels
        P = 32   # max points per pillar
        F = 4    # x, y, z, intensity

        dummy_inputs = {
            'voxels': np.random.randn(M, P, F).astype(np.float32),
            'voxel_coords': np.random.randint(0, 256, (M, 4)).astype(np.int32),
            'voxel_num_points': np.random.randint(1, P + 1, (M,)).astype(np.int32),
        }

        print('Running inference...')
        outputs = session.run(None, dummy_inputs)

        print('Inference successful!')
        print('\nOutput shapes:')
        for name, out in zip(['cls_preds', 'box_preds', 'dir_cls_preds'], outputs):
            print(f'  {name}: shape={out.shape}, dtype={out.dtype}, '
                  f'range=[{out.min():.4f}, {out.max():.4f}]')

    except Exception as e:
        print(f'ONNX Runtime inference failed: {e}')
        return False

    print('\nAll verification passed!')
    return True


if __name__ == '__main__':
    args = parse_args()
    verify_onnx(args)
