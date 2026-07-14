from __future__ import annotations

import argparse
import asyncio
from argparse import ArgumentParser
from dataclasses import asdict, is_dataclass
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Callable

import cv2
import numpy as np

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:

    def load_dotenv() -> bool:
        return False

if __package__ in {None, ""}:
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from twopoint_project.contrl.target_center_servo import (
    AimUpdate,
    PointPrediction,
    TargetCenterServo,
    control_conf_threshold,
    sleep_for_loop_rate,
)
from twopoint_project.f32c.gimbal import (
    DEFAULT_BAUDRATE,
    DEFAULT_COMMAND_INTERVAL,
    DEFAULT_ENABLE_SETTLE_DELAY,
    DEFAULT_SERIAL_PORT,
    DEFAULT_SPEED_RPM,
    DEFAULT_STARTUP_DELAY,
    DEFAULT_X_ID,
    DEFAULT_Y_ID,
    open_serial_gimbal,
)
from twopoint_project.flash import open_laser_pointer
from twopoint_project.vision.inferencer import (
    DEFAULT_IMG_SIZE,
    DEFAULT_ONNX_PATH,
    DEFAULT_VISION_BACKEND,
    build_vision_inferencer,
)
from twopoint_project.vision.pipeline import VisionProducer


CenterFrameCallback = Callable[[Any, list[PointPrediction], AimUpdate], None]
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


def env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None or value == "" else float(value)


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None or value == "" else int(value)


def env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value is None or value == "" else value


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


def default_monitor_output_path() -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return DEFAULT_MONITOR_OUTPUT_DIR / f"center_then_flash_{stamp}.mp4"


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
) -> Any:
    image = frame_bgr.copy()
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    conf_threshold = control_conf_threshold()

    cv2.line(image, (center[0] - 18, center[1]), (center[0] + 18, center[1]), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.line(image, (center[0], center[1] - 18), (center[0], center[1] + 18), (255, 255, 255), 1, cv2.LINE_AA)
    cv2.circle(image, center, 22, (255, 255, 255), 1, cv2.LINE_AA)
    draw_text(image, "image center", (center[0] + 8, center[1] - 10))

    colors = {
        "target_center": (0, 220, 0),
        "laser_point": (0, 0, 255),
    }
    for point in points:
        label = str(point["label"])
        color = colors.get(label, (0, 200, 255))
        x, y = normalized_to_pixel(point, width, height)
        valid = point["confidence"] >= conf_threshold
        radius = 7 if valid else 4
        thickness = -1 if valid else 1
        cv2.circle(image, (x, y), radius, color, thickness, cv2.LINE_AA)
        cv2.circle(image, (x, y), radius + 3, (255, 255, 255), 1, cv2.LINE_AA)
        draw_text(image, f"{label} conf={point['confidence']:.2f}", (x + 10, max(y - 8, 18)), color)
        if label == "target_center" and valid:
            cv2.arrowedLine(image, center, (x, y), color, 2, cv2.LINE_AA, tipLength=0.12)

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
    ) -> None:
        self.host = host
        self.port = port
        self.frame_buffer = frame_buffer
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
    ) -> None:
        self.enabled = enabled
        self.output_path = output_path
        self.fps = fps
        self.webrtc_host = webrtc_host
        self.webrtc_port = webrtc_port
        self.backend = backend
        self.providers = providers
        self.frame_buffer = LatestAnnotatedFrame()
        self.recorder = AnnotatedVideoRecorder(output_path, fps)
        self.server = CenterWebRtcServer(host=webrtc_host, port=webrtc_port, frame_buffer=self.frame_buffer)

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

        annotated = draw_aim_frame(captured.frame_bgr, points, update)
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


def parse_args() -> argparse.Namespace:
    load_dotenv()
    parser = ArgumentParser(
        description=(
            "Center the target paper in the camera view, then turn the laser on "
            "for a fixed duration before safely shutting down."
        )
    )
    parser.add_argument("--camera", type=int, default=env_int("TWOPOINT_CAMERA_INDEX", 0))
    parser.add_argument("--camera-width", type=int, default=env_int("TWOPOINT_CAMERA_WIDTH", 640))
    parser.add_argument("--camera-height", type=int, default=env_int("TWOPOINT_CAMERA_HEIGHT", 480))
    parser.add_argument("--camera-fps", type=int, default=env_int("TWOPOINT_CAMERA_FPS", 30))
    parser.add_argument("--vision-backend", default=env_str("TWOPOINT_VISION_BACKEND", DEFAULT_VISION_BACKEND))
    parser.add_argument("--onnx", default=env_str("TWOPOINT_ONNX_PATH", DEFAULT_ONNX_PATH))
    parser.add_argument("--img-size", type=int, default=env_int("TWOPOINT_IMG_SIZE", DEFAULT_IMG_SIZE))

    parser.add_argument("--port", default=env_str("F32C_SERIAL_PORT", DEFAULT_SERIAL_PORT))
    parser.add_argument("--baudrate", type=int, default=env_int("F32C_BAUDRATE", DEFAULT_BAUDRATE))
    parser.add_argument("--x-id", type=int, default=env_int("F32C_X_ID", DEFAULT_X_ID))
    parser.add_argument("--y-id", type=int, default=env_int("F32C_Y_ID", DEFAULT_Y_ID))
    parser.add_argument("--speed-rpm", type=int, default=env_int("F32C_SPEED_RPM", DEFAULT_SPEED_RPM))
    parser.add_argument("--startup-delay", type=float, default=env_float("F32C_STARTUP_DELAY", DEFAULT_STARTUP_DELAY))
    parser.add_argument(
        "--command-interval",
        type=float,
        default=env_float("F32C_COMMAND_INTERVAL", DEFAULT_COMMAND_INTERVAL),
    )
    parser.add_argument(
        "--enable-settle-delay",
        type=float,
        default=env_float("F32C_ENABLE_SETTLE_DELAY", DEFAULT_ENABLE_SETTLE_DELAY),
    )
    parser.add_argument("--debug-frames", action="store_true")

    parser.add_argument("--center-x", type=float, default=env_float("CENTER_TARGET_X", 0.5))
    parser.add_argument("--center-y", type=float, default=env_float("CENTER_TARGET_Y", 0.5))
    parser.add_argument("--x-gain-deg", type=float, default=env_float("CENTER_X_GAIN_DEG", -8.0))
    parser.add_argument("--y-gain-deg", type=float, default=env_float("CENTER_Y_GAIN_DEG", 8.0))
    parser.add_argument("--max-step-deg", type=float, default=env_float("CENTER_MAX_STEP_DEG", 1.0))
    parser.add_argument("--deadband", type=float, default=env_float("CENTER_DEADBAND", 0.006))
    parser.add_argument("--loop-hz", type=float, default=env_float("CENTER_LOOP_HZ", 15.0))
    parser.add_argument("--center-timeout", type=float, default=env_float("CENTER_TIMEOUT", 100.0))
    parser.add_argument("--stable-frames", type=int, default=env_int("CENTER_STABLE_FRAMES", 3))
    parser.add_argument("--laser-hold-seconds", type=float, default=env_float("LASER_HOLD_SECONDS", 5.0))
    parser.add_argument(
        "--record-webrtc",
        action=argparse.BooleanOptionalAction,
        default=env_bool("CENTER_RECORD_WEBRTC_ENABLED", False),
        help="Record annotated frames and serve the same annotated frames over WebRTC.",
    )
    parser.add_argument(
        "--record-webrtc-output",
        default=env_str("CENTER_RECORD_WEBRTC_OUTPUT", ""),
        help="Annotated MP4 output path. Defaults to outputs/center_then_flash_YYYYmmdd_HHMMSS.mp4.",
    )
    parser.add_argument(
        "--record-webrtc-fps",
        type=float,
        default=env_float("CENTER_RECORD_WEBRTC_FPS", env_float("CENTER_LOOP_HZ", 15.0)),
    )
    parser.add_argument(
        "--record-webrtc-host",
        default=env_str("CENTER_RECORD_WEBRTC_HOST", env_str("WEBRTC_HOST", "0.0.0.0")),
    )
    parser.add_argument(
        "--record-webrtc-port",
        type=int,
        default=env_int("CENTER_RECORD_WEBRTC_PORT", env_int("WEBRTC_PORT", 8080)),
    )
    args = parser.parse_args()
    control_conf_threshold()
    if args.record_webrtc_fps <= 0:
        raise ValueError("record-webrtc-fps must be greater than 0")
    if args.record_webrtc_port <= 0:
        raise ValueError("record-webrtc-port must be greater than 0")
    args.record_webrtc_output = (
        Path(args.record_webrtc_output) if args.record_webrtc_output else default_monitor_output_path()
    )
    return args


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


def run(args: argparse.Namespace) -> bool:
    if args.center_timeout < 0:
        raise ValueError("center-timeout must be non-negative")
    if args.laser_hold_seconds < 0:
        raise ValueError("laser-hold-seconds must be non-negative")
    if args.stable_frames <= 0:
        raise ValueError("stable-frames must be greater than 0")

    inferencer = build_vision_inferencer(
        backend=args.vision_backend,
        onnx_path=args.onnx,
        img_size=args.img_size,
    )
    servo = TargetCenterServo(
        center_x=args.center_x,
        center_y=args.center_y,
        x_gain_deg=args.x_gain_deg,
        y_gain_deg=args.y_gain_deg,
        max_step_deg=args.max_step_deg,
        deadband=args.deadband,
    )

    with open_camera_capture(
        camera_index=args.camera,
        width=args.camera_width,
        height=args.camera_height,
        fps=args.camera_fps,
    ) as capture, open_serial_gimbal(
        port=args.port,
        baudrate=args.baudrate,
        x_id=args.x_id,
        y_id=args.y_id,
        speed_rpm=args.speed_rpm,
        startup_delay=args.startup_delay,
        command_interval=args.command_interval,
        enable_settle_delay=args.enable_settle_delay,
        debug_frames=args.debug_frames,
    ) as gimbal, VisionProducer(
        capture=capture,
        inferencer=inferencer,
    ) as vision, open_laser_pointer(initial_on=False) as laser:
        with CenterRunMonitor(
            enabled=args.record_webrtc,
            output_path=args.record_webrtc_output,
            fps=args.record_webrtc_fps,
            webrtc_host=args.record_webrtc_host,
            webrtc_port=args.record_webrtc_port,
            backend=args.vision_backend,
            providers=inferencer.providers,
        ) as monitor:
            print("task: center_then_flash")
            print(f"task: backend={args.vision_backend}")
            print(f"task: providers={inferencer.providers}")
            print(f"task: center_timeout={args.center_timeout:.2f}s")
            print(f"task: laser_hold_seconds={args.laser_hold_seconds:.2f}s")
            print(f"monitor: record_webrtc={args.record_webrtc}")

            laser.off()
            gimbal.initialize()

            try:
                centered = center_target(
                    vision=vision,
                    gimbal=gimbal,
                    servo=servo,
                    loop_hz=args.loop_hz,
                    timeout=args.center_timeout,
                    stable_frames=args.stable_frames,
                    on_frame=monitor.on_frame if args.record_webrtc else None,
                )

                print("laser: on")
                laser.on()
                time.sleep(args.laser_hold_seconds)
                print("laser: off")
                return centered
            finally:
                laser.off()
                gimbal.disable()


def main() -> None:
    raise SystemExit(0 if run(parse_args()) else 1)


if __name__ == "__main__":
    main()
