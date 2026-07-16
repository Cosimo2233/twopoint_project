请基于当前项目，将 USB 摄像头到 YOLO11n-pose 控制链路整理为低延迟、只保留最新帧和最新结果的实时架构。

目标链路：

稳定的 `/dev/v4l/by-id/...` 设备路径
→ GStreamer `v4l2src` + 明确 caps
→ MJPEG 解码 + `videoconvert`
→ `appsink`
→ 采集线程
→ `LatestCapturedFrame`
→ NPU 推理线程
→ YOLO11n-pose 预处理
→ A7A VIPLite NPU + 后处理
→ `LatestVisionState`
→ 控制线程

请重点修改和确认以下内容：

1. 使用稳定设备路径，不依赖易变化的摄像头数字索引。程序自动枚举
   `/dev/v4l/by-id/*-video-index0`：恰好一个候选时使用；没有候选或存在
   多个候选时明确报错，不能回退到 `/dev/video0`。

2. GStreamer pipeline 中明确配置：

   * 摄像头分辨率
   * FPS
   * MJPEG 格式
   * 解码前使用 `queue max-size-buffers=1 leaky=downstream` 丢弃旧压缩帧
   * `appsink max-buffers=1`
   * 非阻塞丢弃旧帧、保留最新帧
   * `sync=false`

3. 当前设备使用 GStreamer 1.18.4。该版本 appsink 的 `drop=true` 明确定义为
   队列满时丢弃旧 buffer，并且没有新版 `leaky-type` 属性。因此最终使用
   `max-buffers=1 drop=true sync=false emit-signals=false enable-last-sample=false
   wait-on-eos=false`；解码前的 queue 使用 `leaky=downstream` 丢弃旧 buffer。

4. 采集线程只负责：

   * 从 appsink 取 sample
   * 校验帧
   * 生成递增 `frame_id`
   * 记录 `time.monotonic_ns()`
   * 读取并保存 GstBuffer 的 PTS、duration
   * 发布最新 `CapturedFrame`

   不要在采集线程中执行模型预处理、推理、后处理、画图或控制逻辑。

5. 处理好 GstBuffer 到 NumPy 的内存生命周期。不能在 sample 释放或 buffer unmap 后继续持有失效的 NumPy view。第一版可在发布 `CapturedFrame` 前明确复制一次。

6. `LatestCapturedFrame` 必须是非阻塞覆盖式状态槽，不是普通阻塞 FIFO 队列：

   * 容量为 1
   * 新帧覆盖旧帧
   * 生产者不能因满队列阻塞
   * 推理线程只处理尚未处理过的最新 `frame_id`

7. 推理线程每次完成推理后，重新获取当前最新帧，不处理历史积压帧。

8. YOLO11n-pose 预处理保持独立：

   * 可选 ROI
   * Letterbox 到模型尺寸
   * RGB
   * UINT8
   * 按 VIPLite 模型真实输入要求处理 layout 和量化

9. 不要同时在 GStreamer 和模型预处理层重复 resize。GStreamer只负责采集、解码和必要的颜色转换，Letterbox 只在模型预处理层执行。

10. 当前业务绘制、录像、WebRTC 和其他视觉后端统一使用 BGR，因此第一版
    appsink 输出 BGR，由 YOLO11n-pose 预处理转换为 RGB。完成性能埋点后，
    再决定是否把公共图像契约整体迁移为 RGB，不能让 RGB 图像继续使用
    `frame_bgr` 命名。

11. 预处理必须保存：

* 原图宽高
* 模型输入宽高
* scale
* pad_left
* pad_top
* ROI offset

12. 后处理时，bbox 和全部关键点必须使用同一套 scale、padding 和 ROI offset 映射回原始画面。

13. `VisionResult` 必须关联：

* `camera_id`
* `stream_generation`
* `source_frame_id`
* 帧时间戳
* 推理开始时间
* 推理结束时间
* bbox
* bbox confidence
* keypoints
* keypoint confidence

14. `LatestVisionState` 同样使用非阻塞覆盖式状态槽，不使用可能阻塞的容量 1 FIFO 队列。

15. 控制线程只读取最新视觉状态，并检查结果新鲜度。结果超过允许时限后必须视为 stale，不能继续作为有效控制输入。

16. 摄像头重连后递增 `stream_generation`。帧和视觉结果都必须携带该字段，避免重连前的旧结果污染新视频流。

17. 处理 GStreamer ERROR、EOS、sample timeout、USB 拔出和重新插入。重建 pipeline 后重新应用 caps。

18. 分别统计并记录：

* sample 获取间隔
* MJPEG 解码和颜色转换后的采集耗时
* GstBuffer → NumPy copy 耗时
* Letterbox 耗时
* VIPLite 推理耗时
* Pose 后处理耗时
* 端到端帧龄
* 跳过帧数

19. 不要在每帧创建新线程或异步任务。长期线程控制在：

* Capture Thread
* Inference Thread
* Control Thread
* 可选 Display/Recording Thread

   VIPLite context 必须在 Inference Thread 内创建、推理并销毁，不能跨线程使用。

20. 保持现有业务逻辑不变，只调整采集、缓存、推理和结果发布的数据流与并发模型。

最终请输出：

* 修改后的数据流说明
* 涉及的文件和类
* 关键并发结构
* appsink 最终参数
* `CapturedFrame` 和 `VisionResult` 字段
* 性能埋点位置
* 摄像头异常和重连流程




关于摄像头的配置统一走.env，采集策略写死，实时模式，消费最新帧。
设备路径由程序按上面的唯一 `/dev/v4l/by-id/*-video-index0` 规则自动解析，
因此环境变量只配置下面三个参数。

先默认 1280×720@30
TWOPOINT_CAMERA_WIDTH=1280
TWOPOINT_CAMERA_HEIGHT=720
TWOPOINT_CAMERA_FPS=30
只配置这三个
.json里不配置摄像
