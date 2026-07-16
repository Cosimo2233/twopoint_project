# A733 NPU pose backend

The third vision backend runs `model-bin/pose/best_pcq_a733.nb` through the
Cubie A7A VIPLite v2.0 runtime. Build its native bridge on the A7A with:

```bash
cmake \
  -S native/yolo11_pose_npu \
  -B build/npu \
  -DAI_SDK_ROOT=/home/radxa/repositories/ai-sdk \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/npu --parallel
cmake --build build/npu --target test
```

The runtime configuration is:

```dotenv
TWOPOINT_VISION_BACKEND=npu_pose
TWOPOINT_NPU_MODEL_PATH=model-bin/pose/best_pcq_a733.nb
TWOPOINT_NPU_LIBRARY_PATH=build/npu/libyolo11_pose_npu.so
TWOPOINT_NPU_SCORE_THRESHOLD=0.4
TWOPOINT_NPU_NMS_THRESHOLD=0.45
TWOPOINT_NPU_TARGET_KEYPOINT_INDEX=0
TWOPOINT_IMG_SIZE=640
```

`TWOPOINT_NPU_TARGET_KEYPOINT_INDEX` selects which of the model's five pose
keypoints becomes the shared `target_center` prediction. On the current
`best_pcq_a733.nb`, the A7A smoke test confirmed that keypoint 0 is the target
center and keypoints 1-4 are the four target corners.

For a single-image hardware check:

```bash
poetry run python tests/npu_pose_demo.py path/to/image.jpg
```
