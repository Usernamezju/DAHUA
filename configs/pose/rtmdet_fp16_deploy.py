"""Static TensorRT FP16 export for the RTMDet-nano person detector."""

codebase_config = dict(
    type="mmdet",
    task="ObjectDetection",
    model_type="end2end",
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
    type="onnx",
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=11,
    save_file="end2end.onnx",
    input_names=["input"],
    output_names=["dets", "labels"],
    input_shape=[320, 320],
    optimize=True,
)

backend_config = dict(
    type="tensorrt",
    common_config=dict(
        fp16_mode=True,
        int8_mode=False,
        max_workspace_size=1 << 30,
    ),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, 320, 320],
                    opt_shape=[1, 3, 320, 320],
                    max_shape=[1, 3, 320, 320],
                )
            )
        )
    ],
)
