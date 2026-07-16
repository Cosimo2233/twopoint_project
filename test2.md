稳定设备路径
    ↓
GStreamer pipeline
    ↓
v4l2src
    ↓
明确指定分辨率 / FPS / Pixel Format
    ↓
必要的解码与颜色转换
    ↓
appsink
    ↓
只保留最新 sample
    ↓
BGR 或 RGB NumPy 图像
    ↓
frame_id + monotonic timestamp
    ↓
CapturedFrame
    ↓
YOLO11n-pose 预处理
    ↓
A7A NPU