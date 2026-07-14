from __future__ import annotations

import asyncio
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
import select
import socket
import subprocess
import sys
import termios
import threading
import time
import tty
from typing import Any, Callable

import cv2
import numpy as np

from twopoint_project.config import CenterFlashTrackConfig, CenterThenFlashConfig, RuntimeConfig
from twopoint_project.contrl.target_center_servo import (
    AimUpdate,
    PointPrediction,
    TargetCenterServo,
    sleep_for_loop_rate,
)
from twopoint_project.f32c.gimbal import open_serial_gimbal
from twopoint_project.flash import open_laser_pointer
from twopoint_project.vision.inferencer import build_vision_inferencer
from twopoint_project.vision.pipeline import VisionProducer


CenterFrameCallback = Callable[[Any, list[PointPrediction], AimUpdate], None]
StopCallback = Callable[[], bool]
DEFAULT_MONITOR_OUTPUT_DIR = Path("outputs")
WEBRTC_INDEX_HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>center_then_flash WebRTC</title>
  <style>
    :root {
      color-scheme: dark;
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #10131a;
      color: #f8fafc;
    }
    body {
      margin: 0;
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(0, 1fr) 320px;
      background: #07090d;
    }
    video {
      width: 100%;
      height: 100vh;
      object-fit: contain;
      background: #000;
    }
    aside {
      padding: 16px;
      border-left: 1px solid #29313d;
      background: #151a23;
      overflow: auto;
    }
    h1 {
      margin: 0 0 16px;
      font-size: 18px;
    }
    dl {
      display: grid;
      grid-template-columns: 86px minmax(0, 1fr);
      gap: 8px 10px;
      margin: 0;
      font-size: 13px;
    }
    dt { color: #94a3b8; }
    dd { margin: 0; overflow-wrap: anywhere; }
    pre {
      margin: 16px 0 0;
      padding: 12px;
      border: 1px solid #29313d;
      border-radius: 6px;
      background: #0b1018;
      color: #dbeafe;
      font-size: 12px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    @media (max-width: 860px) {
      body { display: block; }
      video { height: auto; aspect-ratio: 4 / 3; }
      aside { border-left: 0; border-top: 1px solid #29313d; }
    }
  </style>
</head>
<body>
  <video id="video" autoplay playsinline muted></video>
  <aside>
    <h1>center_then_flash</h1>
    <dl>
      <dt>连接</dt><dd id="connection">idle</dd>
      <dt>ICE</dt><dd id="ice">idle</dd>
      <dt>backend</dt><dd id="backend">-</dd>
      <dt>providers</dt><dd id="providers">-</dd>
      <dt>frame</dt><dd id="frame">-</dd>
      <dt>target</dt><dd id="target">-</dd>
      <dt>reason</dt><dd id="reason">-</dd>
    </dl>
    <pre id="status">{}</pre>
  </aside>
  <script>
    const video = document.getElementById("video");
    const connection = document.getElementById("connection");
    const ice = document.getElementById("ice");
    const backend = document.getElementById("backend");
    const providers = document.getElementById("providers");
    const frame = document.getElementById("frame");
    const target = document.getElementById("target");
    const reason = document.getElementById("reason");
    const status = document.getElementById("status");
    let pc = null;

    function waitForIceGatheringComplete(peer) {
      if (peer.iceGatheringState === "complete") {
        return Promise.resolve();
      }
      return new Promise((resolve) => {
        function checkState() {
          if (peer.iceGatheringState === "complete") {
            peer.removeEventListener("icegatheringstatechange", checkState);
            resolve();
          }
        }
        peer.addEventListener("icegatheringstatechange", checkState);
      });
    }

    function renderStatus(message) {
      backend.textContent = message.backend || "-";
      providers.textContent = (message.providers || []).join(", ") || "-";
      frame.textContent = message.frame_id ?? "-";
      reason.textContent = message.reason || (message.valid ? "ok" : "-");
      if (message.target) {
        target.textContent = `${message.target.x.toFixed(3)}, ${message.target.y.toFixed(3)} / ${message.target.confidence.toFixed(2)}`;
      } else {
        target.textContent = "-";
      }
      status.textContent = JSON.stringify(message, null, 2);
    }

    async function start() {
      pc = new RTCPeerConnection({ iceServers: [] });
      const channel = pc.createDataChannel("status");
      channel.onmessage = (event) => renderStatus(JSON.parse(event.data));
      window.addEventListener("keydown", (event) => {
        if (event.key === "Escape" && channel.readyState === "open") {
          channel.send(JSON.stringify({ type: "stop" }));
        }
      });
      pc.addTransceiver("video", { direction: "recvonly" });
      pc.ontrack = (event) => { video.srcObject = event.streams[0]; };
      pc.onconnectionstatechange = () => { connection.textContent = pc.connectionState; };
      pc.oniceconnectionstatechange = () => { ice.textContent = pc.iceConnectionState; };

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      await waitForIceGatheringComplete(pc);
      const response = await fetch("/offer", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(pc.localDescription),
      });
      await pc.setRemoteDescription(await response.json());
    }

    window.addEventListener("beforeunload", () => {
      if (pc) {
        pc.close();
      }
    });
    start().catch((error) => {
      connection.textContent = "failed";
      status.textContent = String(error.stack || error);
    });
  </script>
</body>
</html>
"""


def open_camera_capture(
    *,
    camera_index: int,
    width: int | None,
    height: int | None,
    fps: int | None,
) -> object:
    from twopoint_project.vision.capture import CameraCapture

    return CameraCapture(
        camera_index=camera_index,
        width=width,
        height=height,
        fps=fps,
    )


def default_monitor_output_path(mode: str = "center_then_flash") -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return DEFAULT_MONITOR_OUTPUT_DIR / f"{mode}_{stamp}.mp4"


def normalized_to_pixel(point: PointPrediction, width: int, height: int) -> tuple[int, int]:
    x = int(round(point["x"] * max(width - 1, 1)))
    y = int(round(point["y"] * max(height - 1, 1)))
    return x, y


def draw_text(image: Any, text: str, origin: tuple[int, int], color: tuple[int, int, int] = (255, 255, 255)) -> None:
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_aim_frame(
    frame_bgr: Any,
    points: list[PointPrediction],
    update: AimUpdate,
    conf_threshold: float,
) -> Any:
    image = frame_bgr.copy()
    height, width = image.shape[:2]
    center = (width // 2, height // 2)

    cv2.line(image, (center[0] - 18, center[1]), (center[0] + 18, center[1]), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(image, (center[0], center[1] - 18), (center[0], center[1] + 18), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(image, center, 22, (255, 255, 255), 1, cv2.LINE_AA)
    draw_text(image, "image center", (center[0] + 8, center[1] - 10))

    colors = {
        "target_center": (0, 220, 0),
        "laser_point": (0, 0, 255),
    }
    confidence_lines: list[tuple[str, tuple[int, int, int]]] = []
    for point in points:
        label = str(point["label"])
        color = colors.get(label, (0, 200, 255))
        x, y = normalized_to_pixel(point, width, height)
        valid = point["confidence"] >= conf_threshold
        confidence_lines.append((f"{label}: conf={point['confidence']:.2f}", color))
        radius = 7 if valid else 4
        thickness = -1 if valid else 1
        cv2.circle(image, (x, y), radius, color, thickness, cv2.LINE_AA)
        cv2.circle(image, (x, y), radius + 3, (255, 255, 255), 1, cv2.LINE_AA)
        if label == "target_center" and valid:
            cv2.arrowedLine(image, center, (x, y), color, 2, cv2.LINE_AA, tipLength=0.12)

    for index, (text, color) in enumerate(confidence_lines):
        draw_text(image, text, (14, 54 + index * 22), color)

    if update.step is not None:
        draw_text(
            image,
            "err=({:+.4f},{:+.4f}) step=({:+.3f},{:+.3f}) settled={}".format(
                update.step.error.x,
                update.step.error.y,
                update.step.x_delta_deg,
                update.step.y_delta_deg,
                update.settled,
            ),
            (14, 26),
            (0, 255, 255),
        )
    else:
        draw_text(image, f"invalid: {update.reason}", (14, 26), (0, 0, 255))
    return image


def dataclass_to_dict(value: Any) -> Any:
    if value is None:
        return None
    if is_dataclass(value):
        return asdict(value)
    return value


def local_urls(host: str, port: int) -> list[str]:
    if host not in {"0.0.0.0", "::"}:
        return [f"http://{host}:{port}/"]

    hosts = {"127.0.0.1"}
    try:
        for item in socket.getaddrinfo(socket.gethostname(), None, family=socket.AF_INET):
            address = item[4][0]
            if not address.startswith("127."):
                hosts.add(address)
    except OSError:
        pass
    try:
        output = subprocess.check_output(["hostname", "-I"], text=True, timeout=1.0)
        for address in output.split():
            if "." in address and not address.startswith("127."):
                hosts.add(address)
    except (OSError, subprocess.SubprocessError):
        pass
    return [f"http://{address}:{port}/" for address in sorted(hosts)]


class AnnotatedVideoRecorder:
    def __init__(self, output_path: Path, fps: float) -> None:
        self.output_path = output_path
        self.fps = fps
        self.writer: cv2.VideoWriter | None = None
        self.frame_count = 0

    def write(self, frame_bgr: Any) -> None:
        if self.writer is None:
            height, width = frame_bgr.shape[:2]
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(str(self.output_path), fourcc, self.fps, (width, height))
            if not self.writer.isOpened():
                raise RuntimeError(f"Failed to open video writer: {self.output_path}")
            print(f"monitor: recording annotated video to {self.output_path}")

        self.writer.write(frame_bgr)
        self.frame_count += 1

    def close(self) -> None:
        if self.writer is not None:
            self.writer.release()
            self.writer = None


class LatestAnnotatedFrame:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._frame_bgr: Any | None = None
        self._status: dict[str, Any] = {"type": "vision_status", "reason": "waiting_for_frame"}

    def publish(self, frame_bgr: Any, status: dict[str, Any]) -> None:
        with self._condition:
            self._frame_bgr = frame_bgr.copy()
            self._status = dict(status)
            self._condition.notify_all()

    def snapshot(self) -> tuple[Any, dict[str, Any]]:
        with self._condition:
            if self._frame_bgr is None:
                return self._waiting_frame(), dict(self._status)
            return self._frame_bgr.copy(), dict(self._status)

    @staticmethod
    def _waiting_frame() -> Any:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(image, "waiting for center_then_flash frame", (28, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return image


class WebRtcStatusBroadcaster:
    def __init__(self) -> None:
        self._channels: set[Any] = set()

    def add(self, channel: Any) -> None:
        self._channels.add(channel)

    def discard(self, channel: Any) -> None:
        self._channels.discard(channel)

    def send(self, message: dict[str, Any]) -> None:
        payload = json.dumps(message, ensure_ascii=False)
        for channel in tuple(self._channels):
            if channel.readyState == "open":
                channel.send(payload)
            elif channel.readyState in {"closing", "closed"}:
                self._channels.discard(channel)


class CenterWebRtcServer:
    def __init__(
        self,
        *,
        host: str,
        port: int,
        frame_buffer: LatestAnnotatedFrame,
        stop_event: threading.Event,
    ) -> None:
        self.host = host
        self.port = port
        self.frame_buffer = frame_buffer
        self.stop_event = stop_event
        self.broadcaster = WebRtcStatusBroadcaster()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._runner: Any | None = None
        self._pcs: set[Any] = set()
        self._started = threading.Event()
        self._startup_error: BaseException | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_thread, name="center-webrtc-server", daemon=True)
        self._thread.start()
        self._started.wait(timeout=5.0)
        if self._startup_error is not None:
            raise RuntimeError("failed to start center WebRTC server") from self._startup_error

    def stop(self) -> None:
        if self._loop is None:
            return
        future = asyncio.run_coroutine_threadsafe(self._cleanup(), self._loop)
        try:
            future.result(timeout=5.0)
        except Exception as exc:
            print(f"monitor: WebRTC cleanup failed: {exc}")
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def _run_thread(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._start_async())
            self._started.set()
            loop.run_forever()
        except BaseException as exc:
            self._startup_error = exc
            self._started.set()
        finally:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

    async def _start_async(self) -> None:
        from aiohttp import web
        from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack
        from aiortc.rtcconfiguration import RTCConfiguration
        from av import VideoFrame

        frame_buffer = self.frame_buffer
        broadcaster = self.broadcaster
        pcs = self._pcs

        class LatestFrameTrack(VideoStreamTrack):
            kind = "video"

            async def recv(self) -> Any:
                pts, time_base = await self.next_timestamp()
                frame_bgr, status = frame_buffer.snapshot()
                broadcaster.send(status)
                frame = VideoFrame.from_ndarray(frame_bgr, format="bgr24")
                frame.pts = pts
                frame.time_base = time_base
                return frame

        async def index(_: Any) -> Any:
            return web.Response(text=WEBRTC_INDEX_HTML, content_type="text/html")

        async def health(_: Any) -> Any:
            _, status = frame_buffer.snapshot()
            return web.json_response({"ok": True, "latest": status})

        async def offer(request: Any) -> Any:
            params = await request.json()
            pc = RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
            pcs.add(pc)

            @pc.on("datachannel")
            def on_datachannel(channel: Any) -> None:
                if channel.label != "status":
                    return
                broadcaster.add(channel)

                @channel.on("close")
                def on_close() -> None:
                    broadcaster.discard(channel)

                @channel.on("message")
                def on_message(message: str) -> None:
                    try:
                        data = json.loads(message)
                    except json.JSONDecodeError:
                        return
                    if isinstance(data, dict) and data.get("type") == "stop":
                        self.stop_event.set()

            @pc.on("connectionstatechange")
            async def on_connectionstatechange() -> None:
                if pc.connectionState in {"failed", "closed"}:
                    await pc.close()
                    pcs.discard(pc)

            pc.addTrack(LatestFrameTrack())
            await pc.setRemoteDescription(RTCSessionDescription(sdp=params["sdp"], type=params["type"]))
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            return web.json_response({"sdp": pc.localDescription.sdp, "type": pc.localDescription.type})

        app = web.Application()
        app.router.add_get("/", index)
        app.router.add_get("/health", health)
        app.router.add_post("/offer", offer)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        print("monitor: WebRTC URLs:")
        for url in local_urls(self.host, self.port):
            print(f"  {url}")

    async def _cleanup(self) -> None:
        coros = [pc.close() for pc in self._pcs]
        if coros:
            await asyncio.gather(*coros)
        self._pcs.clear()
        if self._runner is not None:
            await self._runner.cleanup()
            self._runner = None


class CenterRunMonitor:
    def __init__(
        self,
        *,
        enabled: bool,
        output_path: Path,
        fps: float,
        webrtc_host: str,
        webrtc_port: int,
        backend: str,
        providers: list[str],
        conf_threshold: float,
    ) -> None:
        self.enabled = enabled
        self.output_path = output_path
        self.fps = fps
        self.webrtc_host = webrtc_host
        self.webrtc_port = webrtc_port
        self.backend = backend
        self.providers = providers
        self.conf_threshold = conf_threshold
        self.frame_buffer = LatestAnnotatedFrame()
        self.stop_event = threading.Event()
        self.recorder = AnnotatedVideoRecorder(output_path, fps)
        self.server = CenterWebRtcServer(
            host=webrtc_host,
            port=webrtc_port,
            frame_buffer=self.frame_buffer,
            stop_event=self.stop_event,
        )

    def __enter__(self) -> CenterRunMonitor:
        if self.enabled:
            self.server.start()
        return self

    def __exit__(self, *args: object) -> None:
        if self.enabled:
            self.server.stop()
            self.recorder.close()
            print(f"monitor: saved {self.recorder.frame_count} frame(s) to {self.output_path}")

    def on_frame(self, captured: Any, points: list[PointPrediction], update: AimUpdate) -> None:
        if not self.enabled:
            return

        annotated = draw_aim_frame(captured.frame_bgr, points, update, self.conf_threshold)
        status = {
            "type": "vision_status",
            "task": "center_then_flash",
            "backend": self.backend,
            "providers": self.providers,
            "frame_id": captured.frame_id,
            "timestamp": captured.timestamp,
            "age_ms": round((time.time() - captured.timestamp) * 1000.0, 1),
            "points": points,
            "valid": update.valid,
            "moved": update.moved,
            "settled": update.settled,
            "reason": update.reason,
            "target": dataclass_to_dict(update.target),
            "step": dataclass_to_dict(update.step),
        }
        self.recorder.write(annotated)
        self.frame_buffer.publish(annotated, status)

    def stop_requested(self) -> bool:
        return self.stop_event.is_set()


class EscKeyStopper:
    def __init__(self) -> None:
        self.enabled = sys.stdin.isatty()
        self._old_settings: list[Any] | None = None

    def __enter__(self) -> EscKeyStopper:
        if self.enabled:
            self._old_settings = termios.tcgetattr(sys.stdin)
            tty.setcbreak(sys.stdin.fileno())
            print("track: press Esc to stop safely")
        else:
            print("track: stdin is not a TTY; press Ctrl+C to stop safely")
        return self

    def __exit__(self, *args: object) -> None:
        if self.enabled and self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)

    def should_stop(self) -> bool:
        if not self.enabled:
            return False
        readable, _, _ = select.select([sys.stdin], [], [], 0)
        if not readable:
            return False
        return sys.stdin.read(1) == "\x1b"


def center_target(
    *,
    vision: VisionProducer,
    gimbal: object,
    servo: TargetCenterServo,
    loop_hz: float,
    timeout: float,
    stable_frames: int,
    on_frame: CenterFrameCallback | None = None,
) -> bool:
    deadline = time.monotonic() + timeout
    settled_frames = 0

    while time.monotonic() < deadline:
        loop_started_at = time.monotonic()
        read_timeout = min(max(deadline - time.monotonic(), 0.0), 1.0)
        try:
            vision_frame = vision.read_latest(timeout=read_timeout)
        except TimeoutError:
            continue

        captured = vision_frame.captured
        points = vision_frame.points
        update = servo.update(gimbal, points)
        if on_frame is not None:
            on_frame(captured, points, update)

        if update.settled:
            settled_frames += 1
        else:
            settled_frames = 0

        if update.step is not None:
            print(
                "center: frame={frame_id} err=({err_x:+.4f},{err_y:+.4f}) "
                "step=({step_x:+.3f},{step_y:+.3f}) settled={settled}".format(
                    frame_id=captured.frame_id,
                    err_x=update.step.error.x,
                    err_y=update.step.error.y,
                    step_x=update.step.x_delta_deg,
                    step_y=update.step.y_delta_deg,
                    settled=update.settled,
                )
            )
        else:
            print(f"center: frame={captured.frame_id} target_center invalid: {update.reason}")

        if settled_frames >= stable_frames:
            print(f"center: settled for {settled_frames} frame(s)")
            return True

        sleep_for_loop_rate(loop_started_at, loop_hz)

    print(f"center: timeout after {timeout:.2f}s")
    return False


def track_target(
    *,
    vision: VisionProducer,
    gimbal: object,
    servo: TargetCenterServo,
    loop_hz: float,
    stop_requested: StopCallback,
    on_frame: CenterFrameCallback | None = None,
) -> None:
    while not stop_requested():
        loop_started_at = time.monotonic()
        try:
            vision_frame = vision.read_latest(timeout=0.2)
        except TimeoutError:
            continue

        captured = vision_frame.captured
        points = vision_frame.points
        update = servo.update(gimbal, points)
        if on_frame is not None:
            on_frame(captured, points, update)

        if update.step is not None:
            print(
                "track: frame={frame_id} err=({err_x:+.4f},{err_y:+.4f}) "
                "step=({step_x:+.3f},{step_y:+.3f}) settled={settled}".format(
                    frame_id=captured.frame_id,
                    err_x=update.step.error.x,
                    err_y=update.step.error.y,
                    step_x=update.step.x_delta_deg,
                    step_y=update.step.y_delta_deg,
                    settled=update.settled,
                )
            )
        else:
            print(f"track: frame={captured.frame_id} target_center invalid: {update.reason}")

        sleep_for_loop_rate(loop_started_at, loop_hz)

    print("track: stop requested")


def sleep_with_stop(seconds: float, stop_requested: StopCallback) -> bool:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if stop_requested():
            return True
        time.sleep(min(0.05, max(deadline - time.monotonic(), 0.0)))
    return stop_requested()


def validate_runtime(task_config: CenterThenFlashConfig | CenterFlashTrackConfig, runtime_config: RuntimeConfig) -> None:
    if runtime_config.webrtc.enabled and task_config.center.loop_hz <= 0:
        raise ValueError("center.loop_hz must be greater than 0 when WebRTC is enabled")
    if runtime_config.webrtc.port <= 0:
        raise ValueError("TWOPOINT_WEBRTC_PORT must be greater than 0")


def run(task_config: CenterThenFlashConfig, runtime_config: RuntimeConfig) -> bool:
    validate_runtime(task_config, runtime_config)

    inferencer = build_vision_inferencer(
        backend=runtime_config.vision.backend,
        onnx_path=runtime_config.vision.onnx_path,
        img_size=runtime_config.vision.img_size,
    )
    servo = TargetCenterServo(
        center_x=task_config.center.target_x,
        center_y=task_config.center.target_y,
        x_gain_deg=task_config.center.x_gain_deg,
        y_gain_deg=task_config.center.y_gain_deg,
        max_step_deg=task_config.center.max_step_deg,
        deadband=task_config.center.deadband,
        conf_threshold=task_config.center.conf_threshold,
        x_pid=task_config.center.pid.x if task_config.center.pid is not None else None,
        y_pid=task_config.center.pid.y if task_config.center.pid is not None else None,
    )
    monitor_output_path = default_monitor_output_path(task_config.mode)

    with open_camera_capture(
        camera_index=task_config.camera.index,
        width=task_config.camera.width,
        height=task_config.camera.height,
        fps=task_config.camera.fps,
    ) as capture, open_serial_gimbal(
        port=task_config.f32c.port,
        baudrate=task_config.f32c.baudrate,
        x_id=task_config.f32c.x_id,
        y_id=task_config.f32c.y_id,
        speed_rpm=task_config.f32c.speed_rpm,
        startup_delay=task_config.f32c.startup_delay,
        command_interval=task_config.f32c.command_interval,
        enable_settle_delay=task_config.f32c.enable_settle_delay,
        debug_frames=task_config.f32c.debug_frames,
    ) as gimbal, VisionProducer(
        capture=capture,
        inferencer=inferencer,
    ) as vision, open_laser_pointer(initial_on=False) as laser:
        with CenterRunMonitor(
            enabled=runtime_config.webrtc.enabled,
            output_path=monitor_output_path,
            fps=task_config.center.loop_hz,
            webrtc_host=runtime_config.webrtc.host,
            webrtc_port=runtime_config.webrtc.port,
            backend=runtime_config.vision.backend,
            providers=inferencer.providers,
            conf_threshold=task_config.center.conf_threshold,
        ) as monitor:
            print(f"task: {task_config.mode}")
            print(f"task: backend={runtime_config.vision.backend}")
            print(f"task: providers={inferencer.providers}")
            print(f"task: center_timeout={task_config.center.timeout:.2f}s")
            print(f"task: laser_hold_seconds={task_config.laser.hold_seconds:.2f}s")
            print(f"monitor: record_webrtc={runtime_config.webrtc.enabled}")

            laser.off()
            gimbal.initialize()

            try:
                centered = center_target(
                    vision=vision,
                    gimbal=gimbal,
                    servo=servo,
                    loop_hz=task_config.center.loop_hz,
                    timeout=task_config.center.timeout,
                    stable_frames=task_config.center.stable_frames,
                    on_frame=monitor.on_frame if runtime_config.webrtc.enabled else None,
                )

                if centered or task_config.behavior.fire_after_timeout:
                    print("laser: on")
                    laser.on()
                    time.sleep(task_config.laser.hold_seconds)
                    print("laser: off")
                else:
                    print("laser: skipped because centering did not settle")
                return centered
            finally:
                laser.off()
                gimbal.disable()


def run_track(
    task_config: CenterFlashTrackConfig,
    runtime_config: RuntimeConfig,
    stop_requested: StopCallback | None = None,
) -> bool:
    validate_runtime(task_config, runtime_config)

    inferencer = build_vision_inferencer(
        backend=runtime_config.vision.backend,
        onnx_path=runtime_config.vision.onnx_path,
        img_size=runtime_config.vision.img_size,
    )
    servo = TargetCenterServo(
        center_x=task_config.center.target_x,
        center_y=task_config.center.target_y,
        x_gain_deg=task_config.center.x_gain_deg,
        y_gain_deg=task_config.center.y_gain_deg,
        max_step_deg=task_config.center.max_step_deg,
        deadband=task_config.center.deadband,
        conf_threshold=task_config.center.conf_threshold,
        x_pid=task_config.center.pid.x if task_config.center.pid is not None else None,
        y_pid=task_config.center.pid.y if task_config.center.pid is not None else None,
    )
    monitor_output_path = default_monitor_output_path(task_config.mode)

    with open_camera_capture(
        camera_index=task_config.camera.index,
        width=task_config.camera.width,
        height=task_config.camera.height,
        fps=task_config.camera.fps,
    ) as capture, open_serial_gimbal(
        port=task_config.f32c.port,
        baudrate=task_config.f32c.baudrate,
        x_id=task_config.f32c.x_id,
        y_id=task_config.f32c.y_id,
        speed_rpm=task_config.f32c.speed_rpm,
        startup_delay=task_config.f32c.startup_delay,
        command_interval=task_config.f32c.command_interval,
        enable_settle_delay=task_config.f32c.enable_settle_delay,
        debug_frames=task_config.f32c.debug_frames,
    ) as gimbal, VisionProducer(
        capture=capture,
        inferencer=inferencer,
    ) as vision, open_laser_pointer(initial_on=False) as laser:
        with CenterRunMonitor(
            enabled=runtime_config.webrtc.enabled,
            output_path=monitor_output_path,
            fps=task_config.center.loop_hz,
            webrtc_host=runtime_config.webrtc.host,
            webrtc_port=runtime_config.webrtc.port,
            backend=runtime_config.vision.backend,
            providers=inferencer.providers,
            conf_threshold=task_config.center.conf_threshold,
        ) as monitor:
            print(f"task: {task_config.mode}")
            print(f"task: backend={runtime_config.vision.backend}")
            print(f"task: providers={inferencer.providers}")
            print(f"task: center_timeout={task_config.center.timeout:.2f}s")
            print(f"task: laser_hold_seconds={task_config.laser.hold_seconds:.2f}s")
            print(f"monitor: record_webrtc={runtime_config.webrtc.enabled}")

            laser.off()
            gimbal.initialize()

            try:
                centered = center_target(
                    vision=vision,
                    gimbal=gimbal,
                    servo=servo,
                    loop_hz=task_config.center.loop_hz,
                    timeout=task_config.center.timeout,
                    stable_frames=task_config.center.stable_frames,
                    on_frame=monitor.on_frame if runtime_config.webrtc.enabled else None,
                )

                stopper_context = EscKeyStopper() if stop_requested is None else None
                if stopper_context is None:
                    external_stop_requested = stop_requested
                    print("track: using injected stop callback")
                    if external_stop_requested is None:
                        raise RuntimeError("stop callback is not available")

                    def should_stop() -> bool:
                        return monitor.stop_requested() or external_stop_requested()

                    if centered or task_config.behavior.fire_after_timeout:
                        print("laser: on")
                        laser.on()
                        stopped_during_fire = sleep_with_stop(task_config.laser.hold_seconds, should_stop)
                        print("laser: off")
                        laser.off()
                        if stopped_during_fire:
                            return centered
                    else:
                        print("laser: skipped because centering did not settle")
                    track_target(
                        vision=vision,
                        gimbal=gimbal,
                        servo=servo,
                        loop_hz=task_config.center.loop_hz,
                        stop_requested=should_stop,
                        on_frame=monitor.on_frame if runtime_config.webrtc.enabled else None,
                    )
                    return centered

                with stopper_context as stopper:
                    def should_stop() -> bool:
                        return monitor.stop_requested() or stopper.should_stop()

                    if centered or task_config.behavior.fire_after_timeout:
                        print("laser: on")
                        laser.on()
                        stopped_during_fire = sleep_with_stop(task_config.laser.hold_seconds, should_stop)
                        print("laser: off")
                        laser.off()
                        if stopped_during_fire:
                            return centered
                    else:
                        print("laser: skipped because centering did not settle")
                    track_target(
                        vision=vision,
                        gimbal=gimbal,
                        servo=servo,
                        loop_hz=task_config.center.loop_hz,
                        stop_requested=should_stop,
                        on_frame=monitor.on_frame if runtime_config.webrtc.enabled else None,
                    )
                return centered
            except KeyboardInterrupt:
                print("track: interrupted")
                return False
            finally:
                laser.off()
                gimbal.disable()
