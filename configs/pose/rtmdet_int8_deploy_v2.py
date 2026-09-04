"""Self-contained TensorRT INT8 deploy config for RTMDet-S.

Differences from the shipped ``detection_tensorrt-int8_dynamic-320x320-1344x1344``:
- ``fp16_mode=False``: the shipped INT8 config enables FP16 alongside INT8,
  which produced a broken engine ("TensorRT encountered issues when converting
  weights between types") with collapsed scores.
- ``opt_shape=640x640``: the runtime pipeline (Resize keep-ratio to 640x640 +
  Pad 114) always feeds 640x640, so calibration data (also 640x640) tiles
  exactly once instead of being mosaicked to 800x1344.
"""

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
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=11,
    save_file='end2end.onnx',
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

calib_config = dict(create_calib=True, calib_file='calib_data.h5')
