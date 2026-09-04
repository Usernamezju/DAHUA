"""TensorRT INT8 RTMDet-S candidate using MINMAX calibration.

This is a reproducible fallback for the entropy-calibrated candidate.  It
reuses the same ONNX graph and calibration HDF5, while selecting TensorRT's
MINMAX algorithm, which can be more stable for the small campus calibration
set used here.
"""

import tensorrt as trt

codebase_config = dict(
    type='mmdet',
    task='ObjectDetection',
    model_type='end2end',
    post_processing=dict(
        score_threshold=0.05,
        confidence_threshold=0.005,
        iou_threshold=0.5,
        max_output_boxes_per_class=200,
        pre_top_k=5000,
        keep_top_k=100,
        background_label_id=-1,
    ),
)

onnx_config = dict(
    type='onnx',
    input_names=['input'],
    output_names=['dets', 'labels'],
    input_shape=None,
    optimize=True,
    dynamic_axes={
        'input': {0: 'batch', 2: 'height', 3: 'width'},
        'dets': {0: 'batch', 1: 'num_dets'},
        'labels': {0: 'batch', 1: 'num_dets'},
    },
)

backend_config = dict(
    type='tensorrt',
    common_config=dict(
        fp16_mode=False,
        int8_mode=True,
        max_workspace_size=1 << 30,
        int8_param=dict(
            algorithm=trt.CalibrationAlgoType.MINMAX_CALIBRATION,
            calib_file='calib_data.h5',
            model_type='end2end',
        ),
    ),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 320, 320],
                    opt_shape=[1, 3, 640, 640],
                    max_shape=[1, 3, 1344, 1344])))
    ],
)

calib_config = dict(create_calib=False, calib_file='calib_data.h5')
