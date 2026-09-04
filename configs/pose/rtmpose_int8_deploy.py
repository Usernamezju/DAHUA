"""Self-contained TensorRT INT8 deploy config for RTMPose-S.

Same fix as ``rtmdet_int8_deploy_v2.py``: the shipped INT8 backend config
enables FP16 alongside INT8, which produced a broken RTMDet engine, so FP16 is
disabled here. The pose input is always a 256x192 TopDownAffine crop, so a
static profile is used and calibration tiles exactly once.
"""

codebase_config = dict(type='mmpose', task='PoseDetection')

onnx_config = dict(
    type='onnx',
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=11,
    save_file='end2end.onnx',
    input_names=['input'],
    # RTMPose-S uses SimCC and exports two tensors (x/y distributions).  The
    # official mmdeploy runtime uses these names to decode keypoints; exposing
    # a single generic output makes PoseDetector fail during post-processing.
    output_names=['simcc_x', 'simcc_y'],
    input_shape=[192, 256],
    optimize=True,
)

backend_config = dict(
    type='tensorrt',
    common_config=dict(
        fp16_mode=False,
        int8_mode=True,
        max_workspace_size=1 << 30,
    ),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 256, 192],
                    opt_shape=[1, 3, 256, 192],
                    max_shape=[1, 3, 256, 192])))
    ],
)

calib_config = dict(create_calib=True, calib_file='calib_data.h5')
