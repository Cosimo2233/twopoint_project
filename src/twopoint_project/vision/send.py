from __future__ import annotations

import json
import socket
import time
from typing import Any


class UnixJsonLineSender:
    def __init__(self, socket_path: str, reconnect_interval: float = 1.0) -> None:
        self.socket_path = socket_path
        self.reconnect_interval = reconnect_interval
        self._socket: socket.socket | None = None

    def __enter__(self) -> UnixJsonLineSender:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def connect(self) -> None:
        self.close()
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            client.connect(self.socket_path)
        except OSError:
            client.close()
            raise
        self._socket = client

    def ensure_connected(self) -> None:
        while self._socket is None:
            try:
                self.connect()
            except OSError:
                time.sleep(self.reconnect_interval)

    def send(self, payload: dict[str, Any]) -> bool:
        self.ensure_connected()
        message = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
        try:
            if self._socket is None:
                return False
            self._socket.sendall(message.encode("utf-8"))
            return True
        except OSError:
            self.close()
            return False

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._socket.close()
            finally:
                self._socket = None
