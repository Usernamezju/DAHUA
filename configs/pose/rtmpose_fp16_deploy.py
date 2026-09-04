"""Static TensorRT FP16 export for RTMPose-S SimCC COCO-17."""

codebase_config = dict(type="mmpose", task="PoseDetection")

onnx_config = dict(
    type="onnx",
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=11,
    save_file="end2end.onnx",
    input_names=["input"],
    output_names=["simcc_x", "simcc_y"],
    input_shape=[192, 256],
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
                    min_shape=[1, 3, 256, 192],
                    opt_shape=[1, 3, 256, 192],
                    max_shape=[1, 3, 256, 192],
                )
            )
        )
    ],
)
