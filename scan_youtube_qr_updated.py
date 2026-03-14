#!/usr/bin/env python3
"""
Scan a local video file and extract QR codes with timestamps.

Features:
- Local-file-only input validation
- ROI-first detection with full-frame fallback
- Two-pass turbo mode for local videos
- OpenCV detection plus optional zxingcpp fallback
- Temporal grouping, deduplication, CSV and JSON export
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlparse

try:
    import numpy as np
except ImportError:  # pragma: no cover - hard dependency at runtime
    np = None

try:
    import cv2
except ImportError:  # pragma: no cover - hard dependency at runtime
    cv2 = None

try:
    import zxingcpp
except ImportError:  # pragma: no cover - optional backend
    zxingcpp = None

try:
    import requests
except ImportError:  # pragma: no cover - optional HTTP client
    requests = None


EXIT_OK = 0
EXIT_METADATA_ERROR = 2
EXIT_ANALYSIS_ERROR = 3
EXIT_DEPENDENCY_ERROR = 4
DISCORD_WEBHOOK_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36 AutoQR/1.0"
)


_THREAD_STATE = threading.local()
ROI_PRESETS = ("upper-left", "upper-right", "center")
NDArray = Any if np is None else np.ndarray
FrameCandidate = tuple[Optional[str], NDArray, Optional[NDArray]]


@dataclass(frozen=True)
class ROIHint:
    """User-provided ROI, either explicit coordinates or a preset."""

    coords: Optional[tuple[int, int, int, int]] = None
    preset: Optional[str] = None


@dataclass
class QRResult:
    """Serializable result for one grouped QR detection."""

    timestamp_seconds: float
    timestamp_hhmmss_ms: str
    qr_content: Optional[str]
    image_file: str
    frame_number: int
    occurrences: int = 1
    confirmed: bool = False
    last_timestamp_seconds: float = 0.0
    last_timestamp_hhmmss_ms: str = ""
    last_frame_number: int = 0


@dataclass
class TemporalHitState:
    """Tracks the last grouped occurrence for one dedupe key."""

    result_index: int
    last_timestamp: float


@dataclass
class PendingUndecodedState:
    """Temporary state for undecoded QR candidates awaiting confirmation."""

    first_timestamp: float
    first_frame_number: int
    last_timestamp: float
    last_frame_number: int
    occurrences: int = 1
    qr_image: Optional[np.ndarray] = None


class DiscordNotifier:
    """Asynchronous Discord webhook notifier for confirmed QR detections."""

    def __init__(
        self,
        webhook_url: Optional[str],
        *,
        username: Optional[str] = None,
        avatar_url: Optional[str] = None,
        mentions: Optional[str] = None,
        debug: bool = False,
    ) -> None:
        self.webhook_url = (webhook_url or "").strip()
        self.username = (username or "").strip() or None
        self.avatar_url = (avatar_url or "").strip() or None
        self.mentions = self._normalize_mentions(mentions)
        self.debug = bool(debug)
        self.enabled = bool(self.webhook_url)
        self._queue: queue.Queue[Optional[dict[str, Any]]] = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self._lock = threading.Lock()
        self._sent_result_ids: set[int] = set()

    @staticmethod
    def _normalize_mentions(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        tokens = [
            token.strip()
            for token in str(value).replace(",", " ").split()
            if token.strip()
        ]
        return " ".join(tokens) or None

    @staticmethod
    def _truncate_text(value: str, limit: int) -> str:
        text = str(value)
        if len(text) <= limit:
            return text
        if limit <= 1:
            return text[:limit]
        return text[: limit - 1] + "…"

    @staticmethod
    def _is_web_link(value: Optional[str]) -> bool:
        if not value:
            return False
        parsed = urlparse(str(value).strip())
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    def _format_qr_description(self, value: Optional[str]) -> str:
        if not value:
            return "QR detecte sans texte decode."
        text = str(value).strip()
        if self._is_web_link(text):
            return self._truncate_text(text, 512)
        return self._truncate_text(text, 900)

    def _is_meaningful_result(self, result: QRResult) -> bool:
        if not result.confirmed:
            return False
        if result.qr_content is None:
            return False

        text = str(result.qr_content).strip()
        if not text:
            return False
        if self._is_web_link(text):
            return True

        alnum_chars = [char for char in text if char.isalnum()]
        unique_alnum = {char.lower() for char in alnum_chars}
        if len(alnum_chars) < 4:
            return False
        if len(unique_alnum) < 2:
            return False
        return True

    @staticmethod
    def _parse_retry_after(
        headers: dict[str, Any],
        payload: Optional[dict[str, Any]],
    ) -> Optional[float]:
        if payload and isinstance(payload, dict):
            raw_retry = payload.get("retry_after")
            try:
                retry_value = float(raw_retry)
                if retry_value > 0:
                    return retry_value
            except (TypeError, ValueError):
                pass

        raw_header = headers.get("Retry-After") or headers.get("retry-after")
        if raw_header is None:
            return None
        text = str(raw_header).strip()
        if not text:
            return None
        try:
            retry_value = float(text)
            if retry_value > 0:
                return retry_value
        except ValueError:
            pass

        try:
            retry_at = parsedate_to_datetime(text)
        except (TypeError, ValueError, IndexError, OverflowError):
            return None
        now = time.time()
        return max(0.0, retry_at.timestamp() - now)

    def start(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._started:
                return
            self._thread = threading.Thread(
                target=self._worker_loop,
                name="discord-webhook-worker",
                daemon=True,
            )
            self._thread.start()
            self._started = True

    def stop(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if not self._started:
                return
            self._queue.put(None)
            thread = self._thread
            self._thread = None
            self._started = False
        if thread is not None:
            thread.join()

    def enqueue_detection(
        self,
        *,
        result: QRResult,
        video_path: Path,
    ) -> None:
        if not self.enabled:
            return
        result_id = id(result)
        if result_id in self._sent_result_ids:
            return
        if not self._is_meaningful_result(result):
            if self.debug:
                preview = (result.qr_content or "null").replace("\n", " ")
                print(
                    f"[DEBUG] Discord notification skipped for low-signal content: {preview}",
                    flush=True,
                )
            return
        payload = self._build_payload(result=result, video_path=video_path)
        self._sent_result_ids.add(result_id)
        self._queue.put(payload)

    def _build_payload(
        self,
        *,
        result: QRResult,
        video_path: Path,
    ) -> dict[str, Any]:
        file_path = str(video_path)
        is_link = self._is_web_link(result.qr_content)
        fields: list[dict[str, Any]] = [
            {
                "name": "Fichier",
                "value": self._truncate_text(video_path.name, 256),
                "inline": False,
            },
            {
                "name": "Temps",
                "value": result.timestamp_hhmmss_ms,
                "inline": True,
            },
            {
                "name": "Frame",
                "value": str(result.frame_number),
                "inline": True,
            },
        ]
        if (
            result.last_timestamp_hhmmss_ms
            and result.last_timestamp_hhmmss_ms != result.timestamp_hhmmss_ms
        ):
            fields.append(
                {
                    "name": "Dernier timestamp",
                    "value": result.last_timestamp_hhmmss_ms,
                    "inline": True,
                }
            )
        if result.occurrences > 1:
            fields.append(
                {
                    "name": "Occurrences",
                    "value": str(result.occurrences),
                    "inline": True,
                }
            )

        embed = {
            "title": "Nouveau QR confirme",
            "description": self._format_qr_description(result.qr_content),
            "color": 0x2ECC71,
            "fields": fields,
            "footer": {"text": self._truncate_text(file_path, 180)},
        }
        if is_link:
            embed["url"] = str(result.qr_content).strip()

        payload: dict[str, Any] = {"embeds": [embed]}
        if self.mentions:
            payload["content"] = self.mentions
        if self.username:
            payload["username"] = self.username
        if self.avatar_url:
            payload["avatar_url"] = self.avatar_url
        return payload

    def _worker_loop(self) -> None:
        while True:
            payload = self._queue.get()
            try:
                if payload is None:
                    return
                self._send_payload(payload)
            finally:
                self._queue.task_done()

    def _post_with_requests(
        self, payload: dict[str, Any]
    ) -> tuple[int, dict[str, Any], str, Optional[dict[str, Any]]]:
        assert requests is not None
        response = requests.post(
            self.webhook_url,
            json=payload,
            headers={
                "User-Agent": DISCORD_WEBHOOK_USER_AGENT,
                "Accept": "application/json",
            },
            timeout=(5.0, 15.0),
        )
        body_text = response.text or ""
        body_json: Optional[dict[str, Any]] = None
        if body_text:
            try:
                parsed = response.json()
                if isinstance(parsed, dict):
                    body_json = parsed
            except ValueError:
                body_json = None
        return response.status_code, dict(response.headers), body_text, body_json

    def _post_with_urllib(
        self, payload: dict[str, Any]
    ) -> tuple[int, dict[str, Any], str, Optional[dict[str, Any]]]:
        raw_body = json.dumps(payload).encode("utf-8")
        request = urllib_request.Request(
            self.webhook_url,
            data=raw_body,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": DISCORD_WEBHOOK_USER_AGENT,
            },
            method="POST",
        )
        try:
            with urllib_request.urlopen(request, timeout=15.0) as response:
                body_bytes = response.read()
                status = int(getattr(response, "status", response.getcode()))
                headers = dict(response.headers.items())
        except urllib_error.HTTPError as exc:
            body_bytes = exc.read()
            status = int(exc.code)
            headers = dict(exc.headers.items())
        body_text = body_bytes.decode("utf-8", errors="replace") if body_bytes else ""
        body_json: Optional[dict[str, Any]] = None
        if body_text:
            try:
                parsed = json.loads(body_text)
                if isinstance(parsed, dict):
                    body_json = parsed
            except json.JSONDecodeError:
                body_json = None
        return status, headers, body_text, body_json

    def _post_payload(
        self, payload: dict[str, Any]
    ) -> tuple[int, dict[str, Any], str, Optional[dict[str, Any]]]:
        if requests is not None:
            return self._post_with_requests(payload)
        return self._post_with_urllib(payload)

    def _send_payload(self, payload: dict[str, Any]) -> None:
        while True:
            try:
                status_code, headers, body_text, body_json = self._post_payload(payload)
            except Exception as exc:
                print(
                    f"[WARN] Discord webhook send failed: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                return

            if status_code == 429:
                retry_after = self._parse_retry_after(headers, body_json)
                if retry_after is None:
                    print(
                        "[WARN] Discord rate limit received without retry_after; notification skipped.",
                        file=sys.stderr,
                        flush=True,
                    )
                    return
                if self.debug:
                    print(
                        f"[DEBUG] Discord rate limit hit, retrying in {retry_after:.3f}s.",
                        flush=True,
                    )
                time.sleep(max(0.0, retry_after))
                continue

            if 200 <= status_code < 300:
                if self.debug:
                    print("[DEBUG] Discord notification sent.", flush=True)
                return

            body_preview = (
                self._truncate_text(body_text.strip(), 240) if body_text else ""
            )
            if body_preview:
                print(
                    f"[WARN] Discord webhook returned HTTP {status_code}: {body_preview}",
                    file=sys.stderr,
                    flush=True,
                )
            else:
                print(
                    f"[WARN] Discord webhook returned HTTP {status_code}.",
                    file=sys.stderr,
                    flush=True,
                )
            return


def validate_local_video_path(raw_path: str) -> Path:
    """Validate that the input video exists and is readable."""
    video_path = Path(raw_path).expanduser().resolve()
    if not video_path.exists():
        raise RuntimeError(f"Video file not found: {video_path}")
    if not video_path.is_file():
        raise RuntimeError(f"Video path is not a file: {video_path}")
    if not os.access(video_path, os.R_OK):
        raise RuntimeError(f"Video file is not readable: {video_path}")
    return video_path


def build_ffmpeg_command(
    input_source: str,
    fps_scan: float,
    full_scan: bool,
    *,
    scale_width: Optional[int] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    pixel_format: str = "bgr24",
) -> list[str]:
    """Build an ffmpeg command that emits raw frames to stdout."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-fflags",
        "+genpts",
    ]

    if start_time is not None and start_time > 0:
        cmd.extend(["-ss", f"{start_time:.6f}"])

    cmd.extend(["-i", input_source])

    if end_time is not None and end_time > 0:
        if start_time is not None and start_time > 0:
            segment_duration = max(0.001, end_time - start_time)
            cmd.extend(["-t", f"{segment_duration:.6f}"])
        else:
            cmd.extend(["-to", f"{end_time:.6f}"])

    vf_parts: list[str] = []
    if not full_scan:
        vf_parts.append(f"fps={fps_scan:g}")
    if scale_width is not None and scale_width > 0:
        vf_parts.append(f"scale={scale_width}:-2:flags=fast_bilinear")
    if vf_parts:
        cmd.extend(["-vf", ",".join(vf_parts)])

    cmd.extend(
        [
            "-vsync",
            "0",
            "-an",
            "-sn",
            "-dn",
            "-pix_fmt",
            pixel_format,
            "-f",
            "rawvideo",
            "-",
        ]
    )
    return cmd


def parse_fps(value: Any) -> float:
    """Parse fps value from a number or ratio string."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return 0.0
    if "/" in text:
        num_s, den_s = text.split("/", 1)
        try:
            num = float(num_s)
            den = float(den_s)
            if den == 0:
                return 0.0
            return num / den
        except ValueError:
            return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def ffprobe_stream_info(input_source: str) -> dict[str, float]:
    """
    Probe width/height/fps for a media input using ffprobe.
    Returns an empty dict on failure.
    """
    if shutil.which("ffprobe") is None:
        return {}

    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-of",
        "json",
        input_source,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except Exception:
        return {}

    if proc.returncode != 0 or not proc.stdout:
        return {}

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {}

    streams = payload.get("streams") or []
    if not streams:
        return {}

    stream0 = streams[0]
    return {
        "width": float(stream0.get("width") or 0.0),
        "height": float(stream0.get("height") or 0.0),
        "fps": parse_fps(stream0.get("r_frame_rate")),
    }


def get_local_video_metadata(video_path: Path) -> dict[str, Any]:
    """Probe a local video file for title, dimensions, fps, and duration."""
    probe = ffprobe_stream_info(str(video_path))
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open local video: {video_path}")

    try:
        width = int(probe.get("width", 0)) or int(
            capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0
        )
        height = int(probe.get("height", 0)) or int(
            capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0
        )
        fps = float(probe.get("fps", 0.0)) or float(
            capture.get(cv2.CAP_PROP_FPS) or 0.0
        )
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        capture.release()

    duration_seconds = (total_frames / fps) if total_frames > 0 and fps > 0 else 0.0
    return {
        "title": video_path.stem,
        "width": width,
        "height": height,
        "fps": fps,
        "duration": duration_seconds,
    }


def parse_roi_string(value: str) -> tuple[int, int, int, int]:
    """Parse --roi x1,y1,x2,y2."""
    parts = [item.strip() for item in value.split(",")]
    if len(parts) != 4:
        raise ValueError("ROI must be x1,y1,x2,y2.")
    try:
        x1, y1, x2, y2 = (int(round(float(item))) for item in parts)
    except ValueError as exc:
        raise ValueError("ROI values must be numeric.") from exc
    return x1, y1, x2, y2


def resolve_roi(
    frame_width: int,
    frame_height: int,
    roi_hint: Optional[ROIHint],
) -> Optional[tuple[int, int, int, int]]:
    """Resolve, clip, and validate a ROI against frame bounds."""
    if frame_width <= 0 or frame_height <= 0 or roi_hint is None:
        return None

    if roi_hint.coords is not None:
        x1, y1, x2, y2 = roi_hint.coords
    elif roi_hint.preset == "upper-left":
        x1, y1, x2, y2 = (
            0,
            0,
            int(round(frame_width * 0.5)),
            int(round(frame_height * 0.5)),
        )
    elif roi_hint.preset == "upper-right":
        x1 = int(round(frame_width * 0.5))
        y1 = 0
        x2 = frame_width
        y2 = int(round(frame_height * 0.5))
    elif roi_hint.preset == "center":
        x1 = int(round(frame_width * 0.25))
        y1 = int(round(frame_height * 0.25))
        x2 = int(round(frame_width * 0.75))
        y2 = int(round(frame_height * 0.75))
    else:
        return None

    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1

    x1 = min(max(x1, 0), frame_width)
    x2 = min(max(x2, 0), frame_width)
    y1 = min(max(y1, 0), frame_height)
    y2 = min(max(y2, 0), frame_height)

    if (x2 - x1) < 2 or (y2 - y1) < 2:
        return None
    return (x1, y1, x2, y2)


def scale_roi(
    roi: Optional[tuple[int, int, int, int]],
    src_width: int,
    src_height: int,
    dst_width: int,
    dst_height: int,
) -> Optional[tuple[int, int, int, int]]:
    """Scale a ROI between source and destination frame sizes."""
    if roi is None:
        return None
    if src_width <= 0 or src_height <= 0 or dst_width <= 0 or dst_height <= 0:
        return None

    sx = float(dst_width) / float(src_width)
    sy = float(dst_height) / float(src_height)
    x1, y1, x2, y2 = roi
    scaled = (
        int(round(x1 * sx)),
        int(round(y1 * sy)),
        int(round(x2 * sx)),
        int(round(y2 * sy)),
    )
    return resolve_roi(dst_width, dst_height, ROIHint(coords=scaled))


def crop_with_roi(
    image: np.ndarray,
    roi: Optional[tuple[int, int, int, int]],
) -> Optional[np.ndarray]:
    """Return a cropped ROI, or None if the ROI is empty."""
    if roi is None:
        return None
    x1, y1, x2, y2 = roi
    if x2 <= x1 or y2 <= y1:
        return None
    cropped = image[y1:y2, x1:x2]
    if cropped.size == 0:
        return None
    return cropped


def get_thread_qr_detector() -> cv2.QRCodeDetector:
    """Return one QR detector per worker thread."""
    detector = getattr(_THREAD_STATE, "qr_detector", None)
    if detector is None:
        detector = cv2.QRCodeDetector()
        _THREAD_STATE.qr_detector = detector
    return detector


def get_thread_clahe() -> cv2.CLAHE:
    """Return one CLAHE instance per worker thread."""
    clahe = getattr(_THREAD_STATE, "clahe", None)
    if clahe is None:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        _THREAD_STATE.clahe = clahe
    return clahe


def turbo_prescan_backend_name() -> str:
    """Return the active backend name for turbo phase 1."""
    return "zxingcpp" if zxingcpp is not None else "opencv"


def ensure_uint8_image(image: Any) -> Optional[np.ndarray]:
    """Normalize image dtype/shape to uint8 for save/hash operations."""
    if image is None:
        return None
    arr = np.asarray(image)
    if arr.size == 0:
        return None
    if arr.ndim == 3 and arr.shape[2] == 1:
        arr = arr[:, :, 0]
    if arr.dtype != np.uint8:
        arr = cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        arr = arr.copy()
    return arr


def normalize_points(points: Any) -> list[np.ndarray]:
    """Normalize OpenCV point output to a list of (4,2) float32 arrays."""
    if points is None:
        return []

    arr = np.asarray(points, dtype=np.float32)
    if arr.size == 0:
        return []
    if arr.ndim == 2 and arr.shape == (4, 2):
        return [arr]
    if arr.ndim == 3 and arr.shape[1:] == (4, 2):
        return [arr[i] for i in range(arr.shape[0])]
    if arr.ndim == 4 and arr.shape[0] == 1 and arr.shape[2:] == (4, 2):
        return [arr[0, i] for i in range(arr.shape[1])]

    try:
        reshaped = arr.reshape(-1, 4, 2)
        return [reshaped[i] for i in range(reshaped.shape[0])]
    except ValueError:
        return []


def straight_item_at(straight_qrcode: Any, idx: int) -> Optional[np.ndarray]:
    """Safely access one straight_qrcode item by index."""
    if straight_qrcode is None:
        return None
    if isinstance(straight_qrcode, (list, tuple)):
        if 0 <= idx < len(straight_qrcode):
            return ensure_uint8_image(straight_qrcode[idx])
        return None

    arr = np.asarray(straight_qrcode)
    if arr.size == 0:
        return None
    if arr.ndim >= 3 and arr.shape[0] > idx:
        return ensure_uint8_image(arr[idx])
    if idx == 0:
        return ensure_uint8_image(arr)
    return None


def order_quad_points(pts: np.ndarray) -> np.ndarray:
    """Order four points as top-left, top-right, bottom-right, bottom-left."""
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).ravel()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def sanitize_candidate_quad(
    frame_shape: tuple[int, ...], quad: Any
) -> Optional[np.ndarray]:
    """Reject implausible QR quads that would lead to bogus allocations."""
    try:
        pts = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    except ValueError:
        return None

    if not np.isfinite(pts).all():
        return None

    frame_height = int(frame_shape[0]) if len(frame_shape) >= 1 else 0
    frame_width = int(frame_shape[1]) if len(frame_shape) >= 2 else 0
    if frame_height <= 0 or frame_width <= 0:
        return None

    margin = max(24.0, 0.35 * max(frame_width, frame_height))
    x_min = float(np.min(pts[:, 0]))
    x_max = float(np.max(pts[:, 0]))
    y_min = float(np.min(pts[:, 1]))
    y_max = float(np.max(pts[:, 1]))
    if x_min < -margin or y_min < -margin:
        return None
    if x_max > ((frame_width - 1) + margin) or y_max > ((frame_height - 1) + margin):
        return None

    ordered = order_quad_points(pts)
    width_a = float(np.linalg.norm(ordered[2] - ordered[3]))
    width_b = float(np.linalg.norm(ordered[1] - ordered[0]))
    height_a = float(np.linalg.norm(ordered[1] - ordered[2]))
    height_b = float(np.linalg.norm(ordered[0] - ordered[3]))
    side = max(width_a, width_b, height_a, height_b)
    max_allowed_side = float(max(frame_width, frame_height) * 1.5)
    if side < 4.0 or side > max_allowed_side:
        return None

    quad_area = abs(float(cv2.contourArea(ordered)))
    max_allowed_area = float(frame_width * frame_height) * 1.25
    if quad_area < 16.0 or quad_area > max_allowed_area:
        return None

    return ordered


def warp_qr_from_points(frame: np.ndarray, quad: np.ndarray) -> Optional[np.ndarray]:
    """Perspective-warp a QR area from the 4-corner polygon."""
    ordered = sanitize_candidate_quad(frame.shape, quad)
    if ordered is None:
        return None

    width_a = np.linalg.norm(ordered[2] - ordered[3])
    width_b = np.linalg.norm(ordered[1] - ordered[0])
    height_a = np.linalg.norm(ordered[1] - ordered[2])
    height_b = np.linalg.norm(ordered[0] - ordered[3])
    side = int(round(max(width_a, width_b, height_a, height_b)))
    if side < 2:
        return None

    dst = np.array(
        [[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(frame, matrix, (side, side))
    return ensure_uint8_image(warped)


def crop_qr_bbox(frame: np.ndarray, quad: np.ndarray) -> Optional[np.ndarray]:
    """Fallback crop from QR polygon bounds."""
    try:
        pts = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    except ValueError:
        return None

    x_min = max(int(np.floor(np.min(pts[:, 0]))), 0)
    y_min = max(int(np.floor(np.min(pts[:, 1]))), 0)
    x_max = min(int(np.ceil(np.max(pts[:, 0]))), frame.shape[1] - 1)
    y_max = min(int(np.ceil(np.max(pts[:, 1]))), frame.shape[0] - 1)
    if x_max <= x_min or y_max <= y_min:
        return None

    crop = frame[y_min : y_max + 1, x_min : x_max + 1]
    return ensure_uint8_image(crop)


def zxing_position_to_quad(barcode: Any) -> Optional[np.ndarray]:
    """Convert a zxing-cpp barcode position to a (4,2) float32 quad."""
    position = getattr(barcode, "position", None)
    if position is None:
        return None

    try:
        quad = np.array(
            [
                [position.top_left.x, position.top_left.y],
                [position.top_right.x, position.top_right.y],
                [position.bottom_right.x, position.bottom_right.y],
                [position.bottom_left.x, position.bottom_left.y],
            ],
            dtype=np.float32,
        )
    except Exception:
        return None

    if quad.shape != (4, 2) or not np.isfinite(quad).all():
        return None
    return quad


def zxing_barcode_text(barcode: Any) -> Optional[str]:
    """Extract decoded text from a zxing-cpp barcode object."""
    raw_text = getattr(barcode, "text", None)
    if isinstance(raw_text, bytes):
        raw_text = raw_text.decode("utf-8", errors="ignore")
    if isinstance(raw_text, str):
        normalized = raw_text.strip()
        if normalized:
            return normalized

    raw_bytes = getattr(barcode, "bytes", None)
    if isinstance(raw_bytes, (bytes, bytearray)):
        normalized = bytes(raw_bytes).decode("utf-8", errors="ignore").strip()
        if normalized:
            return normalized

    return None


def zxing_read_barcodes(image: np.ndarray) -> list[Any]:
    """Run zxing-cpp with a QR-focused configuration when available."""
    if zxingcpp is None:
        return []

    kwargs: dict[str, Any] = {
        "formats": zxingcpp.BarcodeFormat.QRCode,
        "try_rotate": True,
        "try_downscale": False,
        "try_invert": True,
    }
    try:
        return list(zxingcpp.read_barcodes(image, return_errors=True, **kwargs))
    except TypeError:
        try:
            return list(zxingcpp.read_barcodes(image, **kwargs))
        except Exception:
            return []
    except Exception:
        return []


def is_plausible_zxing_candidate(barcode: Any, frame_shape: tuple[int, int]) -> bool:
    """Filter weak zxing-cpp error results for prescan usage."""
    quad = zxing_position_to_quad(barcode)
    if quad is None:
        return False

    ordered = order_quad_points(quad)
    width_a = float(np.linalg.norm(ordered[2] - ordered[3]))
    width_b = float(np.linalg.norm(ordered[1] - ordered[0]))
    height_a = float(np.linalg.norm(ordered[1] - ordered[2]))
    height_b = float(np.linalg.norm(ordered[0] - ordered[3]))
    min_side = min(width_a, width_b, height_a, height_b)
    max_side = max(width_a, width_b, height_a, height_b)
    if min_side < 14.0 or max_side <= 0.0:
        return False

    aspect = max(width_a, width_b) / max(max(height_a, height_b), 1e-6)
    aspect = max(aspect, 1.0 / max(aspect, 1e-6))
    if aspect > 1.8:
        return False

    frame_height, frame_width = frame_shape[:2]
    frame_area = float(max(1, frame_width * frame_height))
    area_ratio = abs(cv2.contourArea(ordered)) / frame_area
    if area_ratio < 0.00010 or area_ratio > 0.45:
        return False

    fill_ratio = abs(float(cv2.contourArea(ordered))) / max(min_side * max_side, 1e-6)
    if fill_ratio < 0.42:
        return False
    return True


def prescan_zxing_scan(image: np.ndarray) -> list[Any]:
    """Run zxing-cpp prescan with stable options for QR-only detection."""
    return zxing_read_barcodes(image)


def select_best_plausible_zxing_quad(
    results: list[Any],
    frame_shape: tuple[int, int],
) -> Optional[np.ndarray]:
    """Return the largest plausible zxing candidate quad."""
    best_quad: Optional[np.ndarray] = None
    best_area = 0.0
    for barcode in results:
        quad = zxing_position_to_quad(barcode)
        if quad is None or not is_plausible_zxing_candidate(barcode, frame_shape):
            continue
        area = abs(float(cv2.contourArea(order_quad_points(quad))))
        if area > best_area:
            best_area = area
            best_quad = quad
    return best_quad


def prescan_valid_decode_found(results: list[Any]) -> bool:
    """Whether zxing-cpp already produced a valid QR decode."""
    for barcode in results:
        if bool(getattr(barcode, "valid", False)):
            return True
        if zxing_barcode_text(barcode):
            return True
    return False


def prescan_extract_roi(
    gray_frame: np.ndarray,
    quad: Optional[np.ndarray],
    *,
    padding_ratio: float = 0.45,
) -> Optional[np.ndarray]:
    """Extract a padded ROI around a prescan candidate quad."""
    if quad is None:
        return None

    safe_quad = sanitize_candidate_quad(gray_frame.shape, quad)
    if safe_quad is None:
        return None

    frame_height, frame_width = gray_frame.shape[:2]
    x_min = float(np.min(safe_quad[:, 0]))
    x_max = float(np.max(safe_quad[:, 0]))
    y_min = float(np.min(safe_quad[:, 1]))
    y_max = float(np.max(safe_quad[:, 1]))
    pad_x = max(8.0, (x_max - x_min) * padding_ratio)
    pad_y = max(8.0, (y_max - y_min) * padding_ratio)

    left = max(0, int(np.floor(x_min - pad_x)))
    top = max(0, int(np.floor(y_min - pad_y)))
    right = min(frame_width, int(np.ceil(x_max + pad_x)))
    bottom = min(frame_height, int(np.ceil(y_max + pad_y)))
    if right - left < 16 or bottom - top < 16:
        return None

    roi = gray_frame[top:bottom, left:right]
    if roi.size == 0:
        return None
    return roi


def opencv_prescan_detect_presence(
    gray_frame: np.ndarray,
    candidate_quad: Optional[np.ndarray] = None,
) -> bool:
    """OpenCV confirm step for weak prescan candidates."""
    detector = get_thread_qr_detector()
    frames_to_try: list[np.ndarray] = []

    roi = prescan_extract_roi(gray_frame, candidate_quad)
    if roi is not None:
        frames_to_try.append(roi)
        if min(roi.shape[:2]) < 160:
            frames_to_try.append(
                cv2.resize(roi, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            )

    if not frames_to_try:
        frames_to_try.append(gray_frame)

    for probe in frames_to_try:
        try:
            ok_single, single_points = detector.detect(probe)
        except cv2.error:
            ok_single, single_points = False, None
        if ok_single and normalize_points(single_points):
            return True

    if roi is None:
        try:
            ok_detect, detect_points = detector.detectMulti(gray_frame)
        except cv2.error:
            ok_detect, detect_points = False, None
        return bool(ok_detect and normalize_points(detect_points))
    return False


def opencv_prescan_confirm_candidate(
    gray_frame: np.ndarray,
    candidate_quad: Optional[np.ndarray],
) -> bool:
    """Require at least two independent positive signals for weak candidates."""
    detector = get_thread_qr_detector()
    roi = prescan_extract_roi(gray_frame, candidate_quad)
    if roi is None:
        return False

    positive_signals = 0
    probes: list[np.ndarray] = [roi]
    if min(roi.shape[:2]) < 160:
        probes.append(
            cv2.resize(roi, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        )
    probes.append(get_thread_clahe().apply(roi))

    for probe in probes:
        try:
            ok_single, single_points = detector.detect(probe)
        except cv2.error:
            ok_single, single_points = False, None
        if ok_single and normalize_points(single_points):
            positive_signals += 1
            if positive_signals >= 2:
                return True
    return False


def prescan_frame_needs_boost(gray_frame: np.ndarray) -> bool:
    """Cheap heuristic to decide whether CLAHE rescue is worth trying."""
    if gray_frame.size == 0:
        return False

    height, width = gray_frame.shape[:2]
    thumb_w = 96 if width >= 96 else max(24, width)
    thumb_h = max(24, int(round((height * thumb_w) / max(width, 1))))
    thumb = cv2.resize(gray_frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)

    mean, stddev = cv2.meanStdDev(thumb)
    contrast_std = float(stddev[0][0]) if stddev.size else 0.0
    p10, p90 = np.percentile(thumb, [10, 90])
    dynamic_range = float(p90 - p10)
    avg_luma = float(mean[0][0]) if mean.size else 0.0

    if contrast_std < 42.0 or dynamic_range < 96.0:
        return True
    if (avg_luma < 70.0 or avg_luma > 185.0) and contrast_std < 52.0:
        return True
    return False


def _prescan_detect_score_impl(gray_frame: np.ndarray) -> int:
    """
    Fast-but-recall-oriented QR score for turbo pass 1.

    Returns:
    - 0: no plausible QR signal
    - 1: weak but confirmed signal
    - 2: strong signal
    """
    if zxingcpp is not None:
        results = prescan_zxing_scan(gray_frame)
        if prescan_valid_decode_found(results):
            return 2
        weak_quad = select_best_plausible_zxing_quad(results, gray_frame.shape[:2])
        if weak_quad is not None and opencv_prescan_confirm_candidate(
            gray_frame, weak_quad
        ):
            return 1

        should_try_boost = weak_quad is not None or prescan_frame_needs_boost(gray_frame)
        if should_try_boost:
            boosted = get_thread_clahe().apply(gray_frame)
            results = prescan_zxing_scan(boosted)
            if prescan_valid_decode_found(results):
                return 2
            weak_quad = select_best_plausible_zxing_quad(results, boosted.shape[:2])
            if weak_quad is not None and opencv_prescan_confirm_candidate(
                boosted, weak_quad
            ):
                return 1

        height, width = gray_frame.shape[:2]
        if should_try_boost and max(width, height) < 1280:
            upscaled = cv2.resize(
                gray_frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC
            )
            results = prescan_zxing_scan(upscaled)
            if prescan_valid_decode_found(results):
                return 2
            weak_quad = select_best_plausible_zxing_quad(results, upscaled.shape[:2])
            if weak_quad is not None and opencv_prescan_confirm_candidate(
                upscaled, weak_quad
            ):
                return 1
        return 0

    if opencv_prescan_detect_presence(gray_frame):
        return 1

    should_try_boost = prescan_frame_needs_boost(gray_frame)
    if should_try_boost:
        boosted = get_thread_clahe().apply(gray_frame)
        if opencv_prescan_detect_presence(boosted):
            return 1

    height, width = gray_frame.shape[:2]
    if should_try_boost and max(width, height) < 1280:
        upscaled = cv2.resize(
            gray_frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC
        )
        if opencv_prescan_detect_presence(upscaled):
            return 1
    return 0


def prescan_detect_score(
    gray_frame: np.ndarray,
    roi: Optional[tuple[int, int, int, int]] = None,
) -> int:
    """ROI-first prescan score with full-frame fallback."""
    if roi is not None:
        roi_frame = crop_with_roi(gray_frame, roi)
        if roi_frame is not None:
            score = _prescan_detect_score_impl(roi_frame)
            if score > 0:
                return score
    return _prescan_detect_score_impl(gray_frame)


def prescan_detect_scores_batch(
    frames: list[np.ndarray],
    roi: Optional[tuple[int, int, int, int]] = None,
) -> list[int]:
    """Process a small batch of prescan frames inside one worker task."""
    return [prescan_detect_score(frame, roi) for frame in frames]


def compute_scaled_dimensions(
    source_width: int,
    source_height: int,
    max_width: int,
) -> tuple[int, int]:
    """Compute downscaled dimensions preserving aspect ratio."""
    if source_width <= 0 or source_height <= 0:
        return 0, 0
    if max_width <= 0 or source_width <= max_width:
        return source_width, source_height

    target_width = max_width
    target_height = int(round((source_height * target_width) / source_width))
    target_height = max(2, target_height)
    if target_height % 2 != 0:
        target_height += 1
    return target_width, target_height


def merge_time_windows(
    timestamps: list[float],
    padding_seconds: float,
    merge_gap_seconds: float,
    max_end_seconds: float,
) -> list[tuple[float, float]]:
    """Build and merge candidate time windows around detected timestamps."""
    if not timestamps:
        return []

    padding = max(0.0, padding_seconds)
    merge_gap = max(0.0, merge_gap_seconds)
    intervals: list[tuple[float, float]] = []

    for ts in timestamps:
        start = max(0.0, ts - padding)
        end = ts + padding
        if max_end_seconds > 0:
            end = min(end, max_end_seconds)
        intervals.append((start, max(start, end)))

    intervals.sort(key=lambda item: item[0])
    merged: list[tuple[float, float]] = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= (last_end + merge_gap):
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def optimize_precise_scan_windows(
    windows: list[tuple[float, float]],
    *,
    base_merge_gap_seconds: float,
    max_end_seconds: float,
) -> list[tuple[float, float]]:
    """Merge nearby precise windows to reduce ffmpeg process churn."""
    if len(windows) <= 1:
        return windows

    normalized: list[tuple[float, float]] = []
    for start_s, end_s in sorted(windows, key=lambda item: item[0]):
        start = max(0.0, float(start_s))
        end = max(start, float(end_s))
        if max_end_seconds > 0:
            start = min(start, max_end_seconds)
            end = min(end, max_end_seconds)
        if end > start:
            normalized.append((start, end))

    if len(normalized) <= 1:
        return normalized

    total_duration = sum(end - start for start, end in normalized)
    extra_budget = max(2.0, min(8.0, total_duration * 0.30))
    adaptive_gap = max(float(base_merge_gap_seconds), 0.75)

    merged: list[tuple[float, float]] = [normalized[0]]
    extra_added = 0.0
    for start, end in normalized[1:]:
        last_start, last_end = merged[-1]
        gap = max(0.0, start - last_end)
        if gap <= adaptive_gap and (extra_added + gap) <= extra_budget:
            merged[-1] = (last_start, max(last_end, end))
            extra_added += gap
        else:
            merged.append((start, end))
    return merged


def build_precise_windows_from_hit_spans(
    hit_spans: list[tuple[float, float]],
    prescan_fps: float,
    max_padding_seconds: float,
    merge_gap_seconds: float,
    max_end_seconds: float,
) -> list[tuple[float, float]]:
    """Build tighter precise windows from accepted prescan hit spans."""
    if not hit_spans:
        return []

    safe_prescan_fps = max(float(prescan_fps), 1e-6)
    max_padding = max(0.0, float(max_padding_seconds))
    merge_gap = max(0.0, float(merge_gap_seconds))
    min_padding = min(max_padding, max(0.08, 3.0 / safe_prescan_fps))
    lead_in = min(max_padding, max(min_padding, 8.0 / safe_prescan_fps))
    tail_out = min(max_padding, max(min_padding, 10.0 / safe_prescan_fps))

    intervals: list[tuple[float, float]] = []
    for start_ts, end_ts in hit_spans:
        start = max(0.0, float(start_ts) - lead_in)
        end = max(start, float(end_ts) + tail_out)
        if max_end_seconds > 0:
            end = min(end, max_end_seconds)
        if end > start:
            intervals.append((start, end))

    if not intervals:
        return []

    intervals.sort(key=lambda item: item[0])
    merged: list[tuple[float, float]] = [intervals[0]]
    for start, end in intervals[1:]:
        last_start, last_end = merged[-1]
        if start <= (last_end + merge_gap):
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def filter_prescan_hit_events(
    hit_events: list[tuple[float, int]],
    fps_for_timestamps: float,
) -> tuple[list[float], list[tuple[float, float]]]:
    """Keep only temporally consistent prescan signals."""
    if not hit_events:
        return [], []

    events = sorted((float(ts), int(score)) for ts, score in hit_events if score > 0)
    if not events:
        return [], []

    confirm_window_seconds = max(0.15, min(0.35, 4.0 / max(fps_for_timestamps, 1e-6)))
    accepted_timestamps: list[float] = []
    accepted_spans: list[tuple[float, float]] = []
    cluster: list[tuple[float, int]] = [events[0]]

    def flush_cluster(cluster_events: list[tuple[float, int]]) -> None:
        if not cluster_events:
            return

        scores = [score for _, score in cluster_events]
        strong_present = any(score >= 2 for score in scores)
        weak_count = sum(1 for score in scores if score >= 1)
        if not strong_present and weak_count < 2:
            return

        total_weight = sum(max(score, 1) for _, score in cluster_events)
        ts_center = sum(ts * max(score, 1) for ts, score in cluster_events) / float(
            total_weight
        )
        accepted_timestamps.append(ts_center)
        accepted_spans.append((cluster_events[0][0], cluster_events[-1][0]))

    for ts, score in events[1:]:
        prev_ts = cluster[-1][0]
        if (ts - prev_ts) <= confirm_window_seconds:
            cluster.append((ts, score))
        else:
            flush_cluster(cluster)
            cluster = [(ts, score)]
    flush_cluster(cluster)

    return accepted_timestamps, accepted_spans


def prescan_candidate_windows_from_pipe(
    ffmpeg_cmd: list[str],
    width: int,
    height: int,
    fps_for_timestamps: float,
    duration_seconds: float,
    window_padding_seconds: float,
    merge_gap_seconds: float,
    *,
    timestamp_offset_seconds: float = 0.0,
    gray_input: bool = False,
    motion_threshold: float = 2.0,
    max_skip_frames: int = 2,
    workers: int = 1,
    skip_detection_until_seconds: float = 0.0,
    roi: Optional[tuple[int, int, int, int]] = None,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], int, int, int]:
    """
    Fast pass:
    - runs ROI-first detect-only scoring on downscaled frames
    - returns merged candidate windows for precise pass
    """
    if width <= 0 or height <= 0:
        raise ValueError("Invalid frame dimensions for prescan.")
    if fps_for_timestamps <= 0:
        raise ValueError("Invalid fps_for_timestamps for prescan.")

    channels = 1 if gray_input else 3
    frame_size = width * height * channels
    frame_index = 0
    frames_analyzed = 0
    raw_hit_frames = 0
    hit_events: list[tuple[float, int]] = []
    frame_errors = 0
    detector_calls = 0
    completed_calls = 0
    last_progress = time.monotonic()
    prev_thumb: Optional[np.ndarray] = None
    prev_thumb_quantized: Optional[np.ndarray] = None
    skipped_since_detect = 0
    motion_gating_enabled = max_skip_frames > 0 and motion_threshold > 0.0
    thumb_h = max(32, min(120, height // 6 if height > 0 else 60))
    thumb_w = max(32, min(160, width // 6 if width > 0 else 80))
    max_workers = max(1, int(workers))
    max_inflight = max_workers * 6
    batch_size = 1 if max_workers <= 1 else min(4, max(2, max_workers // 3))
    pending: dict[concurrent.futures.Future[list[int]], list[float]] = {}
    batch_frames: list[np.ndarray] = []
    batch_timestamps: list[float] = []

    def drain_completed(
        wait_for_all: bool = False, block_until_done: bool = False
    ) -> None:
        nonlocal completed_calls, raw_hit_frames
        if not pending:
            return

        done, _ = concurrent.futures.wait(
            pending.keys(),
            timeout=None if (wait_for_all or block_until_done) else 0,
            return_when=(
                concurrent.futures.ALL_COMPLETED
                if wait_for_all
                else concurrent.futures.FIRST_COMPLETED
            ),
        )
        for future in done:
            ts_list = pending.pop(future)
            try:
                scores = [int(score) for score in future.result()]
                completed_calls += len(scores)
                for ts_done, score in zip(ts_list, scores):
                    if score > 0:
                        raw_hit_frames += 1
                        hit_events.append((ts_done, score))
            except Exception as exc:
                print(
                    f"[WARN] Prescan worker failed: {exc}", file=sys.stderr, flush=True
                )

    def submit_batch() -> None:
        nonlocal batch_frames, batch_timestamps
        if not batch_frames:
            return
        future = executor.submit(
            prescan_detect_scores_batch,
            batch_frames,
            roi,
        )
        pending[future] = batch_timestamps
        batch_frames = []
        batch_timestamps = []

    process = subprocess.Popen(
        ffmpeg_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=frame_size * 4,
    )

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            try:
                while True:
                    if process.stdout is None:
                        break
                    chunk = process.stdout.read(frame_size)
                    if not chunk or len(chunk) < frame_size:
                        break

                    ts = timestamp_offset_seconds + (frame_index / fps_for_timestamps)
                    frames_analyzed += 1

                    try:
                        if ts < skip_detection_until_seconds:
                            prev_thumb = None
                            prev_thumb_quantized = None
                            skipped_since_detect = 0
                            frame_index += 1
                            now = time.monotonic()
                            if now - last_progress >= 2.0:
                                drain_completed(wait_for_all=False)
                                print(
                                    f"[PRESCAN] frames={frames_analyzed} | detect_calls={detector_calls} | "
                                    f"done={completed_calls} | t={short_timestamp(ts)} | raw_hits={raw_hit_frames}",
                                    flush=True,
                                )
                                last_progress = now
                            continue

                        if gray_input:
                            frame_gray = np.frombuffer(chunk, dtype=np.uint8).reshape(
                                (height, width)
                            )
                        else:
                            frame_bgr = np.frombuffer(chunk, dtype=np.uint8).reshape(
                                (height, width, 3)
                            )
                            frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

                        run_detection = True
                        if motion_gating_enabled:
                            thumb = cv2.resize(
                                frame_gray,
                                (thumb_w, thumb_h),
                                interpolation=cv2.INTER_AREA,
                            )
                            thumb_quantized = np.right_shift(thumb, 4)
                            if prev_thumb_quantized is not None and np.array_equal(
                                thumb_quantized,
                                prev_thumb_quantized,
                            ):
                                extended_skip_limit = max_skip_frames * 4
                                if skipped_since_detect < extended_skip_limit:
                                    run_detection = False
                                    skipped_since_detect += 1
                                else:
                                    skipped_since_detect = 0
                            elif prev_thumb is not None:
                                diff = cv2.absdiff(thumb, prev_thumb)
                                motion_score = float(np.mean(diff))
                                hot_diff_threshold = max(
                                    6, int(round(motion_threshold * 4.0))
                                )
                                hot_ratio = float(
                                    np.count_nonzero(diff >= hot_diff_threshold)
                                ) / float(diff.size)
                                peak_diff = int(np.max(diff))
                                extended_skip_limit = (
                                    max_skip_frames * 2
                                    if hot_ratio < 0.002
                                    and peak_diff < hot_diff_threshold
                                    else max_skip_frames
                                )
                                if (
                                    motion_score < motion_threshold
                                    and hot_ratio < 0.008
                                    and skipped_since_detect < extended_skip_limit
                                ):
                                    run_detection = False
                                    skipped_since_detect += 1
                                else:
                                    skipped_since_detect = 0
                            prev_thumb = thumb
                            prev_thumb_quantized = thumb_quantized

                        if run_detection:
                            detector_calls += 1
                            batch_frames.append(frame_gray)
                            batch_timestamps.append(ts)
                            if len(batch_frames) >= batch_size:
                                submit_batch()
                            while len(pending) >= max_inflight:
                                drain_completed(
                                    wait_for_all=False, block_until_done=True
                                )
                    except Exception as exc:
                        frame_errors += 1
                        if frame_errors <= 5:
                            print(
                                f"[WARN] Prescan skip frame {frame_index + 1}: {exc}",
                                file=sys.stderr,
                                flush=True,
                            )

                    frame_index += 1
                    now = time.monotonic()
                    if now - last_progress >= 2.0:
                        drain_completed(wait_for_all=False)
                        print(
                            f"[PRESCAN] frames={frames_analyzed} | detect_calls={detector_calls} | "
                            f"done={completed_calls} | t={short_timestamp(ts)} | raw_hits={raw_hit_frames}",
                            flush=True,
                        )
                        last_progress = now
            finally:
                submit_batch()
                drain_completed(wait_for_all=True)
    finally:
        if process.stdout is not None:
            process.stdout.close()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    if frames_analyzed == 0:
        raise RuntimeError("No frame decoded during turbo prescan.")
    if process.returncode not in (0, None):
        print(
            f"[WARN] ffmpeg exited with status {process.returncode}",
            file=sys.stderr,
            flush=True,
        )

    confirmed_timestamps, confirmed_spans = filter_prescan_hit_events(
        hit_events=hit_events,
        fps_for_timestamps=fps_for_timestamps,
    )
    confirmed_clusters = len(confirmed_spans)
    windows = merge_time_windows(
        timestamps=confirmed_timestamps,
        padding_seconds=window_padding_seconds,
        merge_gap_seconds=merge_gap_seconds,
        max_end_seconds=duration_seconds,
    )
    return windows, confirmed_spans, frames_analyzed, confirmed_clusters, detector_calls


def generate_targeted_variants(
    image: np.ndarray,
    *,
    aggressive: bool,
) -> list[tuple[np.ndarray, float]]:
    """
    Generate targeted preprocessing variants for ROI/crops.

    Variants are only used on ROI-sized inputs or candidate crops, never on
    every frame globally during the main scan loop.
    """
    base = ensure_uint8_image(image)
    if base is None:
        return []

    variants: list[tuple[np.ndarray, float]] = [(base, 1.0)]

    if base.ndim == 3:
        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        variants.append((gray, 1.0))
    else:
        gray = base

    clahe = get_thread_clahe().apply(gray)
    variants.append((clahe, 1.0))

    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append((otsu, 1.0))

    if aggressive and min(gray.shape[:2]) >= 32:
        adaptive = cv2.adaptiveThreshold(
            clahe,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            2,
        )
        variants.append((adaptive, 1.0))
        variants.append((cv2.bitwise_not(adaptive), 1.0))

    variants.append((cv2.bitwise_not(gray), 1.0))
    variants.append((cv2.bitwise_not(clahe), 1.0))

    upscale_limit = 1400 if aggressive else 900
    if max(gray.shape[:2]) <= upscale_limit:
        upscale_sources: list[np.ndarray] = [gray, clahe, otsu]
        if aggressive and base.ndim == 3:
            upscale_sources.insert(0, base)
        for source in upscale_sources:
            upscaled = cv2.resize(
                source, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC
            )
            variants.append((upscaled, 2.0))

    return variants


def extract_candidates_from_frame(
    frame: np.ndarray,
    detector: Optional[cv2.QRCodeDetector] = None,
) -> list[FrameCandidate]:
    """Return QR candidates as (decoded_text_or_none, points_4x2, straight_qr_or_none)."""
    detector = detector or get_thread_qr_detector()
    candidates: list[FrameCandidate] = []

    decoded_info: list[Any] = []
    points_multi: Any = None
    straight_multi: Any = None

    try:
        multi_output = detector.detectAndDecodeMulti(frame)
    except cv2.error:
        multi_output = None

    if isinstance(multi_output, tuple):
        if len(multi_output) == 4:
            _, decoded_info, points_multi, straight_multi = multi_output
        elif len(multi_output) == 3:
            decoded_info, points_multi, straight_multi = multi_output

        points_list = normalize_points(points_multi)
        if points_list:
            for idx, quad in enumerate(points_list):
                content: Optional[str] = None
                if decoded_info and idx < len(decoded_info):
                    text = decoded_info[idx]
                    if isinstance(text, bytes):
                        text = text.decode("utf-8", errors="ignore")
                    if isinstance(text, str):
                        content = text.strip() or None
                straight = straight_item_at(straight_multi, idx)
                candidates.append((content, quad, straight))
            return candidates

    try:
        single_text, single_points, single_straight = detector.detectAndDecode(frame)
    except cv2.error:
        single_text, single_points, single_straight = "", None, None

    single_points_list = normalize_points(single_points)
    if single_points_list:
        text = single_text.strip() if isinstance(single_text, str) else ""
        candidates.append(
            (text or None, single_points_list[0], ensure_uint8_image(single_straight))
        )
        return candidates

    try:
        ok_detect, detect_points = detector.detectMulti(frame)
    except cv2.error:
        ok_detect, detect_points = False, None

    if ok_detect:
        for quad in normalize_points(detect_points):
            candidates.append((None, quad, None))
    return candidates


def scale_candidates(
    candidates: list[FrameCandidate],
    scale_factor: float,
) -> list[FrameCandidate]:
    """Map candidate coordinates from a scaled variant back to the base image."""
    if abs(scale_factor - 1.0) < 1e-6:
        return candidates

    mapped: list[FrameCandidate] = []
    for content, points, straight in candidates:
        pts = np.asarray(points, dtype=np.float32) / float(scale_factor)
        mapped.append((content, pts, straight))
    return mapped


def translate_candidates(
    candidates: list[FrameCandidate],
    offset_x: int,
    offset_y: int,
) -> list[FrameCandidate]:
    """Translate candidate coordinates from ROI-local space to full-frame space."""
    translated: list[FrameCandidate] = []
    for content, points, straight in candidates:
        pts = np.asarray(points, dtype=np.float32).copy()
        pts[:, 0] += float(offset_x)
        pts[:, 1] += float(offset_y)
        translated.append((content, pts, straight))
    return translated


def extract_candidates_from_zxing(image: np.ndarray) -> list[FrameCandidate]:
    """Return decoded QR candidates from zxing-cpp when available."""
    results = zxing_read_barcodes(image)
    candidates: list[FrameCandidate] = []
    for barcode in results:
        quad = zxing_position_to_quad(barcode)
        if quad is None:
            continue
        content = zxing_barcode_text(barcode)
        valid = bool(getattr(barcode, "valid", bool(content)))
        if not valid and not content:
            continue
        candidates.append((content, quad, None))
    return candidates


def decode_crop_content(
    crop: np.ndarray,
    detector: Optional[cv2.QRCodeDetector] = None,
) -> tuple[Optional[str], Optional[np.ndarray]]:
    """Decode one candidate crop using targeted variants and backend fallbacks."""
    detector = detector or get_thread_qr_detector()
    variants = generate_targeted_variants(crop, aggressive=True)

    for variant, _scale in variants:
        try:
            text, _points, straight = detector.detectAndDecode(variant)
        except cv2.error:
            text, straight = "", None
        normalized = text.strip() if isinstance(text, str) else ""
        if normalized:
            return normalized, ensure_uint8_image(straight) or ensure_uint8_image(
                variant
            )

        try:
            multi_output = detector.detectAndDecodeMulti(variant)
        except cv2.error:
            multi_output = None
        if isinstance(multi_output, tuple):
            if len(multi_output) == 4:
                _, decoded_info, _points_multi, straight_multi = multi_output
            elif len(multi_output) == 3:
                decoded_info, _points_multi, straight_multi = multi_output
            else:
                decoded_info, straight_multi = [], None
            for idx, item in enumerate(decoded_info or []):
                if isinstance(item, bytes):
                    item = item.decode("utf-8", errors="ignore")
                if isinstance(item, str) and item.strip():
                    return item.strip(), straight_item_at(
                        straight_multi, idx
                    ) or ensure_uint8_image(variant)

    if zxingcpp is not None:
        for variant, _scale in variants:
            for barcode in zxing_read_barcodes(variant):
                content = zxing_barcode_text(barcode)
                valid = bool(getattr(barcode, "valid", bool(content)))
                if valid or content:
                    return content, None
    return None, None


def dedupe_candidate_list(
    frame_shape: tuple[int, ...],
    candidates: list[FrameCandidate],
) -> list[FrameCandidate]:
    """Dedupe per-frame candidates using content, image hash, or geometry."""
    deduped: list[FrameCandidate] = []
    seen: set[str] = set()

    for content, points, straight in candidates:
        safe_points = sanitize_candidate_quad(frame_shape, points)
        if safe_points is None:
            continue

        normalized_content = content.strip() if isinstance(content, str) else None
        normalized_content = normalized_content or None
        normalized_straight = ensure_uint8_image(straight)
        key = dedupe_key(normalized_content, normalized_straight)
        if key is None:
            geom_hash = hashlib.sha1(
                np.asarray(safe_points, dtype=np.float32).tobytes()
            ).hexdigest()
            key = f"points:{geom_hash}"
        if key in seen:
            continue
        seen.add(key)
        deduped.append((normalized_content, safe_points, normalized_straight))

    return deduped


def refine_candidates_with_crop_fallback(
    frame: np.ndarray,
    candidates: list[FrameCandidate],
    detector: Optional[cv2.QRCodeDetector] = None,
) -> list[FrameCandidate]:
    """Refine OpenCV candidates using targeted crop decoding and zxing fallback."""
    detector = detector or get_thread_qr_detector()
    refined: list[FrameCandidate] = []

    for content, points, straight in candidates:
        safe_points = sanitize_candidate_quad(frame.shape, points)
        if safe_points is None:
            continue

        qr_image = ensure_uint8_image(straight)
        if qr_image is None:
            qr_image = warp_qr_from_points(frame, safe_points)
        if qr_image is None:
            qr_image = crop_qr_bbox(frame, safe_points)

        normalized_content = content.strip() if isinstance(content, str) else None
        normalized_content = normalized_content or None
        if qr_image is not None and not normalized_content:
            decoded_content, decoded_straight = decode_crop_content(qr_image, detector)
            if decoded_content:
                normalized_content = decoded_content
            if decoded_straight is not None:
                qr_image = decoded_straight

        refined.append((normalized_content, safe_points, qr_image))

    return dedupe_candidate_list(frame.shape, refined)


def extract_candidates_in_targeted_region(
    frame: np.ndarray,
    detector: Optional[cv2.QRCodeDetector] = None,
) -> list[FrameCandidate]:
    """
    Run precise detection on an ROI or crop.

    Order:
    - OpenCV on original
    - OpenCV on targeted variants
    - zxing-cpp on targeted variants
    """
    detector = detector or get_thread_qr_detector()
    base_candidates = extract_candidates_from_frame(frame, detector)
    if base_candidates:
        refined = refine_candidates_with_crop_fallback(frame, base_candidates, detector)
        if refined:
            return refined

    variants = generate_targeted_variants(frame, aggressive=True)
    for variant, scale in variants[1:]:
        candidates = extract_candidates_from_frame(variant, detector)
        if candidates:
            mapped = scale_candidates(candidates, scale)
            refined = refine_candidates_with_crop_fallback(frame, mapped, detector)
            if refined:
                return refined

    if zxingcpp is not None:
        for variant, scale in variants:
            candidates = extract_candidates_from_zxing(variant)
            if candidates:
                mapped = scale_candidates(candidates, scale)
                refined = refine_candidates_with_crop_fallback(frame, mapped, detector)
                if refined:
                    return refined

    return []


def extract_candidates_global_rescue(
    frame: np.ndarray,
    detector: Optional[cv2.QRCodeDetector] = None,
) -> list[FrameCandidate]:
    """
    Full-frame fallback for hard QR codes after the direct pass has failed.

    This stays limited to a small set of variants to avoid a large regression
    on the main scan path.
    """
    detector = detector or get_thread_qr_detector()
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return []

    variants: list[tuple[np.ndarray, float]] = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boosted = get_thread_clahe().apply(gray)
    variants.append((boosted, 1.0))
    variants.append((cv2.bitwise_not(boosted), 1.0))

    if max(h, w) < 1600:
        up_factor = 1.6
        frame_up = cv2.resize(
            frame, None, fx=up_factor, fy=up_factor, interpolation=cv2.INTER_CUBIC
        )
        variants.append((frame_up, up_factor))
        boosted_up = cv2.resize(
            boosted, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC
        )
        variants.append((boosted_up, 2.0))

    for variant, scale in variants:
        candidates = extract_candidates_from_frame(variant, detector)
        if candidates:
            mapped = scale_candidates(candidates, scale)
            refined = refine_candidates_with_crop_fallback(frame, mapped, detector)
            if refined:
                return refined

    if zxingcpp is not None:
        for variant, scale in variants:
            candidates = extract_candidates_from_zxing(variant)
            if candidates:
                mapped = scale_candidates(candidates, scale)
                refined = refine_candidates_with_crop_fallback(frame, mapped, detector)
                if refined:
                    return refined

    return []


def detect_frame_candidates(
    frame: np.ndarray,
    detector: Optional[cv2.QRCodeDetector] = None,
    roi: Optional[tuple[int, int, int, int]] = None,
) -> list[FrameCandidate]:
    """Run the precise detection pipeline on one frame with ROI-first logic."""
    detector = detector or get_thread_qr_detector()

    if roi is not None:
        roi_frame = crop_with_roi(frame, roi)
        if roi_frame is not None:
            roi_candidates = extract_candidates_in_targeted_region(roi_frame, detector)
            if roi_candidates:
                return translate_candidates(roi_candidates, roi[0], roi[1])

    candidates = extract_candidates_from_frame(frame, detector)
    if candidates:
        refined = refine_candidates_with_crop_fallback(frame, candidates, detector)
        if refined:
            return refined

    return extract_candidates_global_rescue(frame, detector)


def make_timestamp(seconds: float) -> tuple[str, str]:
    """Return human and filename-safe timestamps."""
    total_ms = max(int(round(seconds * 1000.0)), 0)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    m = (total_s // 60) % 60
    h = total_s // 3600
    human = f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"
    safe = f"{h:02d}-{m:02d}-{s:02d}-{ms:03d}"
    return human, safe


def short_timestamp(seconds: float) -> str:
    """Compact timestamp for terminal logs."""
    human, _ = make_timestamp(seconds)
    hh, mm, rest = human.split(":", 2)
    if hh == "00":
        return f"{mm}:{rest}"
    return human


def format_elapsed(seconds: float) -> str:
    """Human-readable elapsed duration for summaries."""
    total_ms = max(int(round(float(seconds) * 1000.0)), 0)
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    m = (total_s // 60) % 60
    h = total_s // 3600
    if h > 0:
        return f"{h:d}h {m:02d}m {s:02d}.{ms:03d}s"
    if m > 0:
        return f"{m:d}m {s:02d}.{ms:03d}s"
    return f"{s:d}.{ms:03d}s"


def normalize_qr_link(content: Optional[str]) -> Optional[str]:
    """Return a normalized terminal-printable URL if QR content is a web link."""
    if not content:
        return None
    text = str(content).strip()
    if not text:
        return None
    parsed = urlparse(text)
    if parsed.scheme in {"http", "https"} and parsed.netloc:
        return text
    return None


def normalize_qr_content(content: Optional[str]) -> Optional[str]:
    """Normalize decoded QR text for summary and dedupe reporting."""
    if not content:
        return None
    text = str(content).strip()
    return text or None


def compute_qr_visual_hash(image: Optional[np.ndarray]) -> Optional[str]:
    """Build a normalized visual hash for one QR-like image array."""
    arr = ensure_uint8_image(image)
    if arr is None or arr.size == 0:
        return None

    gray = arr if arr.ndim == 2 else cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if min(gray.shape[:2]) < 8:
        return None

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    border_pixels = np.concatenate(
        [
            binary[0, :],
            binary[-1, :],
            binary[:, 0],
            binary[:, -1],
        ]
    )
    if border_pixels.size > 0 and float(np.mean(border_pixels)) < 127.0:
        binary = cv2.bitwise_not(binary)

    dark_pixels = cv2.findNonZero(255 - binary)
    if dark_pixels is not None:
        x, y, w, h = cv2.boundingRect(dark_pixels)
        pad = max(2, int(round(max(w, h) * 0.08)))
        left = max(0, x - pad)
        top = max(0, y - pad)
        right = min(binary.shape[1], x + w + pad)
        bottom = min(binary.shape[0], y + h + pad)
        if right - left >= 8 and bottom - top >= 8:
            binary = binary[top:bottom, left:right]

    binary = cv2.resize(
        binary,
        (64, 64),
        interpolation=cv2.INTER_AREA if max(gray.shape[:2]) >= 64 else cv2.INTER_CUBIC,
    )
    return hashlib.sha1(binary.tobytes()).hexdigest()


def collect_unique_qr_links(results: list[QRResult]) -> list[str]:
    """Collect unique QR web links in first-seen order."""
    unique_links: list[str] = []
    seen_links: set[str] = set()
    for result in results:
        link = normalize_qr_link(result.qr_content)
        if link and link not in seen_links:
            seen_links.add(link)
            unique_links.append(link)
    return unique_links


def compute_summary_visual_hash(image_path: Path) -> Optional[str]:
    """Build a normalized visual hash for one saved QR image."""
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    return compute_qr_visual_hash(image)


def collect_qr_summary_stats(results: list[QRResult], images_dir: Path) -> dict[str, int]:
    """Return clearer summary metrics than the raw grouped result count alone."""
    unique_contents: set[str] = set()
    unique_links: set[str] = set()
    unique_null_visuals: set[str] = set()
    decoded_groups = 0
    undecoded_groups = 0

    for result in results:
        normalized_content = normalize_qr_content(result.qr_content)
        if normalized_content is not None:
            decoded_groups += 1
            unique_contents.add(normalized_content)
            link = normalize_qr_link(normalized_content)
            if link is not None:
                unique_links.add(link)
            continue

        undecoded_groups += 1
        visual_hash = compute_summary_visual_hash(images_dir / result.image_file)
        if visual_hash is not None:
            unique_null_visuals.add(visual_hash)

    return {
        "grouped_results": len(results),
        "decoded_groups": decoded_groups,
        "undecoded_groups": undecoded_groups,
        "unique_contents": len(unique_contents),
        "unique_null_visuals": len(unique_null_visuals),
        "unique_links": len(unique_links),
    }


def dedupe_key(
    content: Optional[str], straight_image: Optional[np.ndarray]
) -> Optional[str]:
    """
    Build a deduplication key:
    - prefer decoded content
    - otherwise hash the straight QR image
    """
    if content:
        normalized = content.strip()
        if normalized:
            return f"content:{normalized}"

    image = ensure_uint8_image(straight_image)
    if image is None:
        return None

    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return None
    digest = hashlib.sha1(encoded.tobytes()).hexdigest()
    return f"hash:{digest}"


def build_undecoded_candidate_key(
    qr_image: Optional[np.ndarray], safe_points: np.ndarray
) -> Optional[str]:
    """Build a lightweight stable grouping key for undecoded QR candidates."""
    try:
        ordered = order_quad_points(np.asarray(safe_points, dtype=np.float32).reshape(4, 2))
    except ValueError:
        return None

    x_min = float(np.min(ordered[:, 0]))
    x_max = float(np.max(ordered[:, 0]))
    y_min = float(np.min(ordered[:, 1]))
    y_max = float(np.max(ordered[:, 1]))
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    width = max(1.0, x_max - x_min)
    height = max(1.0, y_max - y_min)

    signature = np.array(
        [
            round(cx / 24.0),
            round(cy / 24.0),
            round(width / 20.0),
            round(height / 20.0),
        ],
        dtype=np.int16,
    )
    geom_hash = hashlib.sha1(signature.tobytes()).hexdigest()
    return f"null-geom:{geom_hash}"


def choose_better_qr_image(
    current_image: Optional[np.ndarray], new_image: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    """Keep the sharper/larger-looking QR crop when merging undecoded hits."""
    current = ensure_uint8_image(current_image)
    incoming = ensure_uint8_image(new_image)
    if current is None:
        return incoming
    if incoming is None:
        return current

    current_area = int(current.shape[0]) * int(current.shape[1])
    incoming_area = int(incoming.shape[0]) * int(incoming.shape[1])
    if incoming_area > current_area:
        return incoming
    return current


def save_qr_result(
    images_dir: Path,
    qr_image: Optional[np.ndarray],
    timestamp_seconds: float,
    frame_number: int,
    content: Optional[str],
    sequence_number: int,
) -> QRResult:
    """Save one QR PNG and return structured metadata for export."""
    human_ts, safe_ts = make_timestamp(timestamp_seconds)
    image_name = f"qr_{sequence_number:04d}_{safe_ts}.png"
    image_path = images_dir / image_name

    image = ensure_uint8_image(qr_image)
    if image is None:
        image = np.zeros((128, 128, 3), dtype=np.uint8)

    ok = cv2.imwrite(str(image_path), image)
    if not ok:
        raise RuntimeError(f"Unable to save QR image: {image_path}")

    return QRResult(
        timestamp_seconds=round(timestamp_seconds, 6),
        timestamp_hhmmss_ms=human_ts,
        qr_content=content if content else None,
        image_file=image_name,
        frame_number=frame_number,
        occurrences=1,
        confirmed=False,
        last_timestamp_seconds=round(timestamp_seconds, 6),
        last_timestamp_hhmmss_ms=human_ts,
        last_frame_number=frame_number,
    )


def update_grouped_result(
    result: QRResult,
    timestamp_seconds: float,
    frame_number: int,
    content: Optional[str],
) -> None:
    """Update temporal grouping metadata for an existing result entry."""
    human_ts, _ = make_timestamp(timestamp_seconds)
    result.occurrences += 1
    result.confirmed = result.confirmed or result.occurrences >= 2
    result.last_timestamp_seconds = round(timestamp_seconds, 6)
    result.last_timestamp_hhmmss_ms = human_ts
    result.last_frame_number = frame_number
    if result.qr_content is None and content:
        result.qr_content = content


def print_progress(
    frames_analyzed: int, timestamp_seconds: float, duration_seconds: float
) -> None:
    """Print a concise progress line."""
    ts = short_timestamp(timestamp_seconds)
    if duration_seconds > 0:
        pct = min((timestamp_seconds / duration_seconds) * 100.0, 100.0)
        print(f"[PROGRESS] frames={frames_analyzed} | t={ts} | {pct:5.1f}%", flush=True)
    else:
        print(f"[PROGRESS] frames={frames_analyzed} | t={ts}", flush=True)


def finalize_frame_candidates(
    frame: np.ndarray,
    frame_number: int,
    timestamp_seconds: float,
    candidates: list[FrameCandidate],
    images_dir: Path,
    results: list[QRResult],
    seen_state: dict[str, TemporalHitState],
    min_interval: float,
    pending_undecoded: Optional[dict[str, PendingUndecodedState]] = None,
    pending_decoded_nonlink: Optional[dict[str, PendingUndecodedState]] = None,
    announced_links: Optional[set[str]] = None,
    notifier: Optional[DiscordNotifier] = None,
    video_path: Optional[Path] = None,
) -> None:
    """Group, dedupe, save, and append candidates already detected on one frame."""
    if not candidates:
        return

    seen_in_frame: set[str] = set()
    group_gap = max(0.35, float(min_interval))
    undecoded_confirm_occurrences = 2
    if pending_undecoded is None:
        pending_undecoded = {}
    if pending_decoded_nonlink is None:
        pending_decoded_nonlink = {}

    for content, points, straight in candidates:
        safe_points = sanitize_candidate_quad(frame.shape, points)
        if safe_points is None:
            continue

        qr_image = ensure_uint8_image(straight)
        if qr_image is None:
            qr_image = warp_qr_from_points(frame, safe_points)
        if qr_image is None:
            qr_image = crop_qr_bbox(frame, safe_points)

        normalized_content = content.strip() if isinstance(content, str) else None
        normalized_content = normalized_content or None
        is_link_content = normalize_qr_link(normalized_content) is not None
        if normalized_content is not None:
            key = dedupe_key(normalized_content, qr_image)
            if key is None:
                geom_hash = hashlib.sha1(
                    np.asarray(safe_points, dtype=np.float32).tobytes()
                ).hexdigest()
                key = f"points:{geom_hash}"
        else:
            key = build_undecoded_candidate_key(qr_image, safe_points)
            if key is None:
                continue

        if key in seen_in_frame:
            continue
        seen_in_frame.add(key)

        state = seen_state.get(key)
        if (
            state is not None
            and (timestamp_seconds - state.last_timestamp) <= group_gap
        ):
            result = results[state.result_index]
            update_grouped_result(
                result, timestamp_seconds, frame_number, normalized_content
            )
            state.last_timestamp = timestamp_seconds
            if result.confirmed and notifier is not None and video_path is not None:
                notifier.enqueue_detection(result=result, video_path=video_path)
            continue

        if normalized_content is None:
            pending_state = pending_undecoded.get(key)
            if (
                pending_state is not None
                and (timestamp_seconds - pending_state.last_timestamp) <= group_gap
            ):
                pending_state.last_timestamp = timestamp_seconds
                pending_state.last_frame_number = frame_number
                pending_state.occurrences += 1
                pending_state.qr_image = choose_better_qr_image(
                    pending_state.qr_image, qr_image
                )
            else:
                pending_state = PendingUndecodedState(
                    first_timestamp=timestamp_seconds,
                    first_frame_number=frame_number,
                    last_timestamp=timestamp_seconds,
                    last_frame_number=frame_number,
                    occurrences=1,
                    qr_image=ensure_uint8_image(qr_image),
                )
                pending_undecoded[key] = pending_state
                continue

            if pending_state.occurrences < undecoded_confirm_occurrences:
                continue

            result = save_qr_result(
                images_dir=images_dir,
                qr_image=pending_state.qr_image,
                timestamp_seconds=pending_state.first_timestamp,
                frame_number=pending_state.first_frame_number,
                content=None,
                sequence_number=len(results) + 1,
            )
            result.occurrences = pending_state.occurrences
            result.confirmed = pending_state.occurrences >= undecoded_confirm_occurrences
            result.last_timestamp_seconds = round(pending_state.last_timestamp, 6)
            result.last_timestamp_hhmmss_ms = make_timestamp(
                pending_state.last_timestamp
            )[0]
            result.last_frame_number = pending_state.last_frame_number
            results.append(result)
            seen_state[key] = TemporalHitState(
                result_index=len(results) - 1, last_timestamp=timestamp_seconds
            )
            pending_undecoded.pop(key, None)

            print(
                f"[QR] {short_timestamp(result.timestamp_seconds)} | contenu=null | image={result.image_file}",
                flush=True,
            )
            if result.confirmed and notifier is not None and video_path is not None:
                notifier.enqueue_detection(result=result, video_path=video_path)
            continue

        if not is_link_content:
            pending_state = pending_decoded_nonlink.get(key)
            if (
                pending_state is not None
                and (timestamp_seconds - pending_state.last_timestamp) <= group_gap
            ):
                pending_state.last_timestamp = timestamp_seconds
                pending_state.last_frame_number = frame_number
                pending_state.occurrences += 1
                pending_state.qr_image = choose_better_qr_image(
                    pending_state.qr_image, qr_image
                )
            else:
                pending_state = PendingUndecodedState(
                    first_timestamp=timestamp_seconds,
                    first_frame_number=frame_number,
                    last_timestamp=timestamp_seconds,
                    last_frame_number=frame_number,
                    occurrences=1,
                    qr_image=ensure_uint8_image(qr_image),
                )
                pending_decoded_nonlink[key] = pending_state
                continue

            if pending_state.occurrences < undecoded_confirm_occurrences:
                continue

            result = save_qr_result(
                images_dir=images_dir,
                qr_image=pending_state.qr_image,
                timestamp_seconds=pending_state.first_timestamp,
                frame_number=pending_state.first_frame_number,
                content=normalized_content,
                sequence_number=len(results) + 1,
            )
            result.occurrences = pending_state.occurrences
            result.confirmed = pending_state.occurrences >= undecoded_confirm_occurrences
            result.last_timestamp_seconds = round(pending_state.last_timestamp, 6)
            result.last_timestamp_hhmmss_ms = make_timestamp(
                pending_state.last_timestamp
            )[0]
            result.last_frame_number = pending_state.last_frame_number
            results.append(result)
            seen_state[key] = TemporalHitState(
                result_index=len(results) - 1, last_timestamp=timestamp_seconds
            )
            pending_decoded_nonlink.pop(key, None)

            preview = result.qr_content if result.qr_content is not None else "null"
            preview = preview.replace("\n", " ")
            if len(preview) > 90:
                preview = preview[:87] + "..."
            print(
                f"[QR] {short_timestamp(result.timestamp_seconds)} | contenu={preview} | image={result.image_file}",
                flush=True,
            )
            if result.confirmed and notifier is not None and video_path is not None:
                notifier.enqueue_detection(result=result, video_path=video_path)
            continue

        result = save_qr_result(
            images_dir=images_dir,
            qr_image=qr_image,
            timestamp_seconds=timestamp_seconds,
            frame_number=frame_number,
            content=normalized_content,
            sequence_number=len(results) + 1,
        )
        results.append(result)
        seen_state[key] = TemporalHitState(
            result_index=len(results) - 1, last_timestamp=timestamp_seconds
        )

        preview = result.qr_content if result.qr_content is not None else "null"
        preview = preview.replace("\n", " ")
        if len(preview) > 90:
            preview = preview[:87] + "..."
        print(
            f"[QR] {short_timestamp(timestamp_seconds)} | contenu={preview} | image={result.image_file}",
            flush=True,
        )
        link = normalize_qr_link(result.qr_content)
        if link and announced_links is not None and link not in announced_links:
            announced_links.add(link)
            print(f"[LINK] {link}", flush=True)


def process_frame_detections(
    frame: np.ndarray,
    frame_number: int,
    timestamp_seconds: float,
    detector: cv2.QRCodeDetector,
    images_dir: Path,
    results: list[QRResult],
    seen_state: dict[str, TemporalHitState],
    min_interval: float,
    pending_undecoded: Optional[dict[str, PendingUndecodedState]] = None,
    pending_decoded_nonlink: Optional[dict[str, PendingUndecodedState]] = None,
    announced_links: Optional[set[str]] = None,
    roi: Optional[tuple[int, int, int, int]] = None,
    notifier: Optional[DiscordNotifier] = None,
    video_path: Optional[Path] = None,
) -> None:
    """Detect QRs on one frame, group them, and append distinct results."""
    candidates = detect_frame_candidates(frame, detector, roi)
    finalize_frame_candidates(
        frame=frame,
        frame_number=frame_number,
        timestamp_seconds=timestamp_seconds,
        candidates=candidates,
        images_dir=images_dir,
        results=results,
        seen_state=seen_state,
        min_interval=min_interval,
        pending_undecoded=pending_undecoded,
        pending_decoded_nonlink=pending_decoded_nonlink,
        announced_links=announced_links,
        notifier=notifier,
        video_path=video_path,
    )


def scan_frames_from_pipe(
    ffmpeg_cmd: list[str],
    width: int,
    height: int,
    fps_for_timestamps: float,
    detector: cv2.QRCodeDetector,
    images_dir: Path,
    min_interval: float,
    duration_seconds: float,
    *,
    timestamp_offset_seconds: float = 0.0,
    frame_number_offset: int = 0,
    results: Optional[list[QRResult]] = None,
    seen_state: Optional[dict[str, TemporalHitState]] = None,
    pending_undecoded: Optional[dict[str, PendingUndecodedState]] = None,
    pending_decoded_nonlink: Optional[dict[str, PendingUndecodedState]] = None,
    workers: int = 1,
    announced_links: Optional[set[str]] = None,
    roi: Optional[tuple[int, int, int, int]] = None,
    notifier: Optional[DiscordNotifier] = None,
    video_path: Optional[Path] = None,
) -> tuple[list[QRResult], int]:
    """Scan frames emitted by an ffmpeg rawvideo pipe."""
    if width <= 0 or height <= 0:
        raise ValueError("Invalid frame dimensions for pipe scanning.")
    if fps_for_timestamps <= 0:
        raise ValueError("Invalid fps_for_timestamps for pipe scanning.")

    frame_size = width * height * 3
    if results is None:
        results = []
    if seen_state is None:
        seen_state = {}
    if pending_undecoded is None:
        pending_undecoded = {}
    if pending_decoded_nonlink is None:
        pending_decoded_nonlink = {}

    frame_errors = 0
    frames_analyzed = 0
    max_workers = max(1, int(workers))
    use_parallel = max_workers > 1
    max_inflight = max_workers * 4

    process = subprocess.Popen(
        ffmpeg_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        bufsize=frame_size * 4,
    )

    last_progress = time.monotonic()
    frame_index = 0

    try:
        if use_parallel:
            next_seq = 0
            next_commit_seq = 0
            pending: dict[
                concurrent.futures.Future[list[FrameCandidate]],
                tuple[int, int, np.ndarray],
            ] = {}
            ready: dict[
                int,
                tuple[
                    int, np.ndarray, Optional[list[FrameCandidate]], Optional[Exception]
                ],
            ] = {}

            def drain_completed(
                wait_for_all: bool = False, block_until_done: bool = False
            ) -> None:
                nonlocal frame_errors, next_commit_seq, last_progress
                if not pending:
                    return

                done, _ = concurrent.futures.wait(
                    pending.keys(),
                    timeout=None if (wait_for_all or block_until_done) else 0,
                    return_when=(
                        concurrent.futures.ALL_COMPLETED
                        if wait_for_all
                        else concurrent.futures.FIRST_COMPLETED
                    ),
                )
                for future in done:
                    seq, frame_index_done, frame_done = pending.pop(future)
                    try:
                        ready[seq] = (
                            frame_index_done,
                            frame_done,
                            future.result(),
                            None,
                        )
                    except Exception as exc:
                        ready[seq] = (frame_index_done, frame_done, None, exc)

                while next_commit_seq in ready:
                    frame_index_done, frame_done, candidates_done, error_done = (
                        ready.pop(next_commit_seq)
                    )
                    timestamp_done = timestamp_offset_seconds + (
                        frame_index_done / fps_for_timestamps
                    )
                    frame_number_done = frame_number_offset + frame_index_done + 1
                    if error_done is not None:
                        frame_errors += 1
                        if frame_errors <= 5:
                            print(
                                f"[WARN] Skipping frame {frame_number_done}: {error_done}",
                                file=sys.stderr,
                                flush=True,
                            )
                    else:
                        try:
                            finalize_frame_candidates(
                                frame=frame_done,
                                frame_number=frame_number_done,
                                timestamp_seconds=timestamp_done,
                                candidates=candidates_done or [],
                                images_dir=images_dir,
                                results=results,
                                seen_state=seen_state,
                                min_interval=min_interval,
                                pending_undecoded=pending_undecoded,
                                pending_decoded_nonlink=pending_decoded_nonlink,
                                announced_links=announced_links,
                                notifier=notifier,
                                video_path=video_path,
                            )
                        except Exception as exc:
                            frame_errors += 1
                            if frame_errors <= 5:
                                print(
                                    f"[WARN] Skipping frame {frame_number_done}: {exc}",
                                    file=sys.stderr,
                                    flush=True,
                                )
                    now = time.monotonic()
                    if now - last_progress >= 2.0:
                        print_progress(
                            frames_analyzed, timestamp_done, duration_seconds
                        )
                        last_progress = now
                    next_commit_seq += 1

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                while True:
                    if process.stdout is None:
                        break
                    chunk = process.stdout.read(frame_size)
                    if not chunk or len(chunk) < frame_size:
                        break

                    try:
                        frame = (
                            np.frombuffer(chunk, dtype=np.uint8)
                            .reshape((height, width, 3))
                            .copy()
                        )
                    except Exception as exc:
                        frame_errors += 1
                        if frame_errors <= 5:
                            frame_number = frame_number_offset + frame_index + 1
                            print(
                                f"[WARN] Skipping frame {frame_number}: {exc}",
                                file=sys.stderr,
                                flush=True,
                            )
                        frame_index += 1
                        continue

                    frames_analyzed += 1
                    future = executor.submit(detect_frame_candidates, frame, None, roi)
                    pending[future] = (next_seq, frame_index, frame)
                    next_seq += 1
                    frame_index += 1

                    while len(pending) >= max_inflight:
                        drain_completed(wait_for_all=False, block_until_done=True)

                drain_completed(wait_for_all=True)
        else:
            while True:
                if process.stdout is None:
                    break
                chunk = process.stdout.read(frame_size)
                if not chunk or len(chunk) < frame_size:
                    break

                timestamp_seconds = timestamp_offset_seconds + (
                    frame_index / fps_for_timestamps
                )
                frame_number = frame_number_offset + frame_index + 1
                frames_analyzed += 1

                try:
                    frame = np.frombuffer(chunk, dtype=np.uint8).reshape(
                        (height, width, 3)
                    )
                    process_frame_detections(
                        frame=frame,
                        frame_number=frame_number,
                        timestamp_seconds=timestamp_seconds,
                        detector=detector,
                        images_dir=images_dir,
                        results=results,
                        seen_state=seen_state,
                        min_interval=min_interval,
                        pending_undecoded=pending_undecoded,
                        pending_decoded_nonlink=pending_decoded_nonlink,
                        announced_links=announced_links,
                        roi=roi,
                        notifier=notifier,
                        video_path=video_path,
                    )
                except Exception as exc:
                    frame_errors += 1
                    if frame_errors <= 5:
                        print(
                            f"[WARN] Skipping frame {frame_number}: {exc}",
                            file=sys.stderr,
                            flush=True,
                        )

                frame_index += 1
                now = time.monotonic()
                if now - last_progress >= 2.0:
                    print_progress(frames_analyzed, timestamp_seconds, duration_seconds)
                    last_progress = now
    finally:
        if process.stdout is not None:
            process.stdout.close()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    if frames_analyzed == 0:
        raise RuntimeError("No frame decoded from ffmpeg stream.")
    if process.returncode not in (0, None):
        print(
            f"[WARN] ffmpeg exited with status {process.returncode}",
            file=sys.stderr,
            flush=True,
        )
    if frame_errors > 0:
        print(
            f"[WARN] {frame_errors} frame(s) failed and were skipped.",
            file=sys.stderr,
            flush=True,
        )

    return results, frames_analyzed


def scan_video_windows_from_file_with_ffmpeg(
    video_path: Path,
    windows: list[tuple[float, float]],
    width: int,
    height: int,
    fps_for_timestamps: float,
    detector: cv2.QRCodeDetector,
    images_dir: Path,
    min_interval: float,
    duration_seconds: float,
    workers: int = 1,
    announced_links: Optional[set[str]] = None,
    roi: Optional[tuple[int, int, int, int]] = None,
    notifier: Optional[DiscordNotifier] = None,
) -> tuple[list[QRResult], int]:
    """Analyze only candidate windows from a local file using ffmpeg."""
    if width <= 0 or height <= 0:
        raise ValueError("Invalid frame dimensions for segmented ffmpeg scanning.")
    if fps_for_timestamps <= 0:
        raise ValueError("Invalid fps_for_timestamps for segmented ffmpeg scanning.")
    if not windows:
        return [], 0

    normalized_windows: list[tuple[float, float]] = []
    for start_s, end_s in windows:
        start = max(0.0, float(start_s))
        end = max(start, float(end_s))
        if duration_seconds > 0:
            start = min(start, duration_seconds)
            end = min(end, duration_seconds)
        if end > start:
            normalized_windows.append((start, end))

    if not normalized_windows:
        return [], 0

    results: list[QRResult] = []
    seen_state: dict[str, TemporalHitState] = {}
    pending_undecoded: dict[str, PendingUndecodedState] = {}
    pending_decoded_nonlink: dict[str, PendingUndecodedState] = {}
    total_frames_analyzed = 0
    total_window_seconds = sum(
        max(0.0, end - start) for start, end in normalized_windows
    )
    print(
        f"[INFO] Turbo precise segments: count={len(normalized_windows)} | "
        f"total_duration={total_window_seconds:.2f}s",
        flush=True,
    )

    for idx, (start_s, end_s) in enumerate(normalized_windows, start=1):
        print(
            f"[INFO] Turbo segment {idx}/{len(normalized_windows)} | "
            f"start={short_timestamp(start_s)} | end={short_timestamp(end_s)}",
            flush=True,
        )
        ffmpeg_cmd = build_ffmpeg_command(
            input_source=str(video_path),
            fps_scan=fps_for_timestamps,
            full_scan=True,
            start_time=start_s,
            end_time=end_s,
        )
        frame_offset = max(0, int(round(start_s * fps_for_timestamps)))
        _, segment_frames = scan_frames_from_pipe(
            ffmpeg_cmd=ffmpeg_cmd,
            width=width,
            height=height,
            fps_for_timestamps=fps_for_timestamps,
            detector=detector,
            images_dir=images_dir,
            min_interval=min_interval,
            duration_seconds=duration_seconds,
            timestamp_offset_seconds=start_s,
            frame_number_offset=frame_offset,
            results=results,
            seen_state=seen_state,
            pending_undecoded=pending_undecoded,
            pending_decoded_nonlink=pending_decoded_nonlink,
            workers=workers,
            announced_links=announced_links,
            roi=roi,
            notifier=notifier,
            video_path=video_path,
        )
        total_frames_analyzed += segment_frames

    return results, total_frames_analyzed


def scan_local_video(
    video_path: Path,
    fps_scan: float,
    full_scan: bool,
    detector: cv2.QRCodeDetector,
    images_dir: Path,
    min_interval: float,
    windows: Optional[list[tuple[float, float]]] = None,
    workers: int = 1,
    announced_links: Optional[set[str]] = None,
    roi: Optional[tuple[int, int, int, int]] = None,
    notifier: Optional[DiscordNotifier] = None,
) -> tuple[list[QRResult], int]:
    """Scan a local video with cv2.VideoCapture."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open local video: {video_path}")

    real_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if real_fps <= 0:
        real_fps = fps_scan if fps_scan > 0 else 30.0

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_seconds = (total_frames / real_fps) if total_frames > 0 else 0.0
    frame_step = 1 if full_scan else max(1, int(round(real_fps / fps_scan)))

    results: list[QRResult] = []
    seen_state: dict[str, TemporalHitState] = {}
    pending_undecoded: dict[str, PendingUndecodedState] = {}
    pending_decoded_nonlink: dict[str, PendingUndecodedState] = {}
    frame_errors = 0
    frames_analyzed = 0
    last_progress = time.monotonic()
    max_workers = max(1, int(workers))
    use_parallel = max_workers > 1
    max_inflight = max_workers * 4

    def finalize_one_frame(
        frame_index: int,
        frame: np.ndarray,
        candidates: list[FrameCandidate],
    ) -> None:
        nonlocal frame_errors, frames_analyzed, last_progress

        timestamp_seconds = frame_index / real_fps
        frame_number = frame_index + 1
        frames_analyzed += 1
        try:
            finalize_frame_candidates(
                frame=frame,
                frame_number=frame_number,
                timestamp_seconds=timestamp_seconds,
                candidates=candidates,
                images_dir=images_dir,
                results=results,
                seen_state=seen_state,
                min_interval=min_interval,
                pending_undecoded=pending_undecoded,
                pending_decoded_nonlink=pending_decoded_nonlink,
                announced_links=announced_links,
                notifier=notifier,
                video_path=video_path,
            )
        except Exception as exc:
            frame_errors += 1
            if frame_errors <= 5:
                print(
                    f"[WARN] Skipping frame {frame_number}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

        now = time.monotonic()
        if now - last_progress >= 2.0:
            print_progress(frames_analyzed, timestamp_seconds, duration_seconds)
            last_progress = now

    def process_one_frame(
        frame_index: int, frame: np.ndarray, region_start_frame: int = 0
    ) -> None:
        nonlocal frame_errors
        if not full_scan and (frame_index - region_start_frame) % frame_step != 0:
            return
        try:
            candidates = detect_frame_candidates(frame, detector, roi)
            finalize_one_frame(frame_index, frame, candidates)
        except Exception as exc:
            frame_errors += 1
            if frame_errors <= 5:
                print(
                    f"[WARN] Skipping frame {frame_index + 1}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )

    def scan_frames_parallel(frame_iter: Any) -> None:
        nonlocal frame_errors

        next_seq = 0
        next_commit_seq = 0
        pending: dict[concurrent.futures.Future[Any], tuple[int, int, np.ndarray]] = {}
        ready: dict[
            int,
            tuple[int, np.ndarray, Optional[list[FrameCandidate]], Optional[Exception]],
        ] = {}

        def drain_completed(
            wait_for_all: bool = False, block_until_done: bool = False
        ) -> None:
            nonlocal frame_errors, next_commit_seq
            if not pending:
                return

            done, _ = concurrent.futures.wait(
                pending.keys(),
                timeout=None if (wait_for_all or block_until_done) else 0,
                return_when=(
                    concurrent.futures.ALL_COMPLETED
                    if wait_for_all
                    else concurrent.futures.FIRST_COMPLETED
                ),
            )
            for future in done:
                seq, frame_index_done, frame_done = pending.pop(future)
                try:
                    ready[seq] = (frame_index_done, frame_done, future.result(), None)
                except Exception as exc:
                    ready[seq] = (frame_index_done, frame_done, None, exc)

            while next_commit_seq in ready:
                frame_index_done, frame_done, candidates_done, error_done = ready.pop(
                    next_commit_seq
                )
                if error_done is not None:
                    frame_errors += 1
                    if frame_errors <= 5:
                        print(
                            f"[WARN] Skipping frame {frame_index_done + 1}: {error_done}",
                            file=sys.stderr,
                            flush=True,
                        )
                else:
                    finalize_one_frame(
                        frame_index_done, frame_done, candidates_done or []
                    )
                next_commit_seq += 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for frame_index, region_start_frame, frame in frame_iter:
                if (
                    not full_scan
                    and (frame_index - region_start_frame) % frame_step != 0
                ):
                    continue

                future = executor.submit(detect_frame_candidates, frame, None, roi)
                pending[future] = (next_seq, frame_index, frame)
                next_seq += 1

                while len(pending) >= max_inflight:
                    drain_completed(wait_for_all=False, block_until_done=True)

            drain_completed(wait_for_all=True)

    try:
        if windows:
            normalized_windows: list[tuple[float, float]] = []
            for start_s, end_s in windows:
                start = max(0.0, float(start_s))
                end = max(start, float(end_s))
                if duration_seconds > 0:
                    start = min(start, duration_seconds)
                    end = min(end, duration_seconds)
                if end > start:
                    normalized_windows.append((start, end))

            normalized_windows.sort(key=lambda item: item[0])

            def iter_window_frames() -> Any:
                for start_s, end_s in normalized_windows:
                    start_frame = max(0, int(np.floor(start_s * real_fps)))
                    end_frame = max(start_frame, int(np.ceil(end_s * real_fps)))
                    capture.set(cv2.CAP_PROP_POS_FRAMES, float(start_frame))
                    frame_index = start_frame
                    while frame_index <= end_frame:
                        ok, frame = capture.read()
                        if not ok:
                            break
                        yield frame_index, start_frame, frame
                        frame_index += 1

            if use_parallel:
                scan_frames_parallel(iter_window_frames())
            else:
                for frame_index, region_start_frame, frame in iter_window_frames():
                    process_one_frame(
                        frame_index, frame, region_start_frame=region_start_frame
                    )
        else:

            def iter_all_frames() -> Any:
                frame_index = 0
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    yield frame_index, 0, frame
                    frame_index += 1

            if use_parallel:
                scan_frames_parallel(iter_all_frames())
            else:
                for frame_index, region_start_frame, frame in iter_all_frames():
                    process_one_frame(
                        frame_index, frame, region_start_frame=region_start_frame
                    )
    finally:
        capture.release()

    if frames_analyzed == 0:
        raise RuntimeError("No frame analyzed from local video.")
    if frame_errors > 0:
        print(
            f"[WARN] {frame_errors} frame(s) failed and were skipped.",
            file=sys.stderr,
            flush=True,
        )

    return results, frames_analyzed


def write_reports(
    results: list[QRResult],
    output_dir: Path,
    output_json_path: Optional[Path] = None,
) -> tuple[Path, Path]:
    """Write CSV and JSON reports."""
    csv_path = output_dir / "qr_results.csv"
    json_path = (
        output_json_path
        if output_json_path is not None
        else (output_dir / "qr_results.json")
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp_seconds",
        "timestamp_hhmmss_ms",
        "last_timestamp_seconds",
        "last_timestamp_hhmmss_ms",
        "qr_content",
        "image_file",
        "frame_number",
        "last_frame_number",
        "occurrences",
        "confirmed",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "timestamp_seconds": f"{result.timestamp_seconds:.6f}",
                    "timestamp_hhmmss_ms": result.timestamp_hhmmss_ms,
                    "last_timestamp_seconds": f"{result.last_timestamp_seconds:.6f}",
                    "last_timestamp_hhmmss_ms": result.last_timestamp_hhmmss_ms,
                    "qr_content": (
                        result.qr_content if result.qr_content is not None else "null"
                    ),
                    "image_file": result.image_file,
                    "frame_number": result.frame_number,
                    "last_frame_number": result.last_frame_number,
                    "occurrences": result.occurrences,
                    "confirmed": result.confirmed,
                }
            )

    json_payload = [asdict(result) for result in results]
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_payload, handle, ensure_ascii=False, indent=2)

    return csv_path, json_path


def parse_args() -> argparse.Namespace:
    """CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze a local video file and extract unique QR codes with timestamps."
    )
    parser.add_argument("video_path", help="Path to a local video file to scan.")
    parser.add_argument(
        "--scan-fps",
        "--fps-scan",
        dest="scan_fps",
        type=float,
        default=10.0,
        help="Frame rate used for scan sampling (ignored by --full-scan).",
    )
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="Force dense scan and analyze all decoded frames.",
    )
    parser.add_argument(
        "--output",
        default="qr_results",
        help="Output directory for images and CSV report.",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional JSON output path. Default: <output>/qr_results.json.",
    )
    parser.add_argument(
        "--min-interval",
        type=float,
        default=1.0,
        help="Minimum temporal gap (seconds) between distinct grouped results for the same QR.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Worker threads used by precise scan and turbo prescan.",
    )
    parser.add_argument(
        "--roi",
        default=None,
        help="ROI in source pixel coordinates: x1,y1,x2,y2.",
    )
    parser.add_argument(
        "--roi-preset",
        choices=ROI_PRESETS,
        default=None,
        help="Simple ROI preset to try before falling back to the full frame.",
    )
    parser.add_argument(
        "--turbo-precise",
        action="store_true",
        help="Two-pass mode: fast prescan, then precise decode on candidate windows.",
    )
    parser.add_argument(
        "--turbo-prescan-fps",
        type=float,
        default=24.0,
        help="Frame rate for the turbo prescan detect-only pass.",
    )
    parser.add_argument(
        "--turbo-scale-width",
        type=int,
        default=960,
        help="Max frame width used during turbo prescan.",
    )
    parser.add_argument(
        "--precise-window",
        "--turbo-window",
        dest="precise_window",
        type=float,
        default=0.75,
        help="Padding in seconds around each prescan hit for the precise pass.",
    )
    parser.add_argument(
        "--turbo-merge-gap",
        type=float,
        default=0.30,
        help="Merge nearby turbo candidate windows separated by less than this gap.",
    )
    parser.add_argument(
        "--turbo-motion-threshold",
        type=float,
        default=2.0,
        help="Motion gating threshold for turbo prescan.",
    )
    parser.add_argument(
        "--turbo-max-skip-frames",
        type=int,
        default=2,
        help="Maximum consecutive low-motion frames skipped during turbo prescan.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print additional diagnostic information.",
    )
    parser.add_argument(
        "--discord-webhook-url",
        default=None,
        help="Discord webhook URL. Overrides DISCORD_WEBHOOK_URL if provided.",
    )
    parser.add_argument(
        "--discord-username",
        default=None,
        help="Optional username used by the Discord webhook.",
    )
    parser.add_argument(
        "--discord-avatar-url",
        default=None,
        help="Optional avatar URL used by the Discord webhook.",
    )
    parser.add_argument(
        "--discord-mentions",
        default=None,
        help="Optional mentions added to the Discord message content, e.g. '@here' or '<@123>'.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if np is None:
        print("[ERROR] numpy is required to scan videos.", file=sys.stderr)
        return EXIT_DEPENDENCY_ERROR
    if cv2 is None:
        print("[ERROR] OpenCV (cv2) is required to scan videos.", file=sys.stderr)
        return EXIT_DEPENDENCY_ERROR

    if args.scan_fps <= 0:
        print("[ERROR] --scan-fps must be > 0.", file=sys.stderr)
        return EXIT_ANALYSIS_ERROR
    if args.min_interval < 0:
        print("[ERROR] --min-interval must be >= 0.", file=sys.stderr)
        return EXIT_ANALYSIS_ERROR
    if args.threads <= 0:
        print("[ERROR] --threads must be > 0.", file=sys.stderr)
        return EXIT_ANALYSIS_ERROR
    if args.turbo_prescan_fps <= 0:
        print("[ERROR] --turbo-prescan-fps must be > 0.", file=sys.stderr)
        return EXIT_ANALYSIS_ERROR
    if args.turbo_scale_width <= 0:
        print("[ERROR] --turbo-scale-width must be > 0.", file=sys.stderr)
        return EXIT_ANALYSIS_ERROR
    if args.precise_window < 0:
        print("[ERROR] --precise-window must be >= 0.", file=sys.stderr)
        return EXIT_ANALYSIS_ERROR
    if args.turbo_merge_gap < 0:
        print("[ERROR] --turbo-merge-gap must be >= 0.", file=sys.stderr)
        return EXIT_ANALYSIS_ERROR
    if args.turbo_motion_threshold < 0:
        print("[ERROR] --turbo-motion-threshold must be >= 0.", file=sys.stderr)
        return EXIT_ANALYSIS_ERROR
    if args.turbo_max_skip_frames < 0:
        print("[ERROR] --turbo-max-skip-frames must be >= 0.", file=sys.stderr)
        return EXIT_ANALYSIS_ERROR

    try:
        video_path = validate_local_video_path(args.video_path)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return EXIT_METADATA_ERROR

    output_dir = Path(args.output).expanduser().resolve()
    images_dir = output_dir / "qr_images"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    output_json_path = (
        Path(args.output_json).expanduser().resolve() if args.output_json else None
    )

    try:
        local_meta = get_local_video_metadata(video_path)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return EXIT_METADATA_ERROR

    title = str(local_meta.get("title") or video_path.stem)
    duration_seconds = float(local_meta.get("duration") or 0.0)
    width = int(local_meta.get("width") or 0)
    height = int(local_meta.get("height") or 0)
    source_fps = float(local_meta.get("fps") or 0.0)
    resolution_text = f"{width}x{height}" if width > 0 and height > 0 else "unknown"

    roi_hint: Optional[ROIHint] = None
    if args.roi:
        try:
            roi_hint = ROIHint(coords=parse_roi_string(args.roi))
        except ValueError as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return EXIT_ANALYSIS_ERROR
    elif args.roi_preset:
        roi_hint = ROIHint(preset=args.roi_preset)

    resolved_roi = resolve_roi(width, height, roi_hint)
    if roi_hint is not None and resolved_roi is None:
        print(
            "[WARN] ROI vide ou invalide apres clipping, fallback frame complet.",
            file=sys.stderr,
            flush=True,
        )

    discord_webhook_url = (
        args.discord_webhook_url or os.environ.get("DISCORD_WEBHOOK_URL") or ""
    ).strip()
    notifier = DiscordNotifier(
        discord_webhook_url,
        username=args.discord_username,
        avatar_url=args.discord_avatar_url,
        mentions=args.discord_mentions,
        debug=args.debug,
    )
    notifier.start()

    print(f"[INFO] Local video file: {video_path}", flush=True)
    print(f"[INFO] Titre: {title}", flush=True)
    print(f"[INFO] Resolution source: {resolution_text}", flush=True)
    if source_fps > 0:
        print(f"[INFO] FPS source: {source_fps:.3f}", flush=True)
    if resolved_roi is not None:
        print(
            f"[INFO] ROI active: {resolved_roi[0]},{resolved_roi[1]},{resolved_roi[2]},{resolved_roi[3]}",
            flush=True,
        )
    if zxingcpp is None:
        print("[INFO] zxingcpp absent: fallback precise limite a OpenCV.", flush=True)
    elif args.debug:
        print(
            "[INFO] zxingcpp disponible pour la passe precise et le prescan.",
            flush=True,
        )
    if notifier.enabled:
        print("[INFO] Notifications Discord actives via webhook.", flush=True)
    elif args.debug:
        print("[DEBUG] Notifications Discord desactivees.", flush=True)

    detector = cv2.QRCodeDetector()
    frames_analyzed = 0
    results: list[QRResult] = []
    announced_links: set[str] = set()
    turbo_prescan_frames = 0
    turbo_prescan_hits = 0
    turbo_prescan_detect_calls = 0
    turbo_precise_frames = 0
    turbo_windows: list[tuple[float, float]] = []
    turbo_hit_spans: list[tuple[float, float]] = []
    turbo_elapsed_seconds: Optional[float] = None
    mode_used = "local-file"

    ffmpeg_available = shutil.which("ffmpeg") is not None
    use_turbo = bool(args.turbo_precise and ffmpeg_available)
    if args.turbo_precise and not ffmpeg_available:
        print(
            "[WARN] ffmpeg absent: fallback sur le scan local direct.",
            file=sys.stderr,
            flush=True,
        )

    try:
        if use_turbo:
            turbo_started_at = time.monotonic()
            print("[INFO] Mode: turbo-precise(local-file)", flush=True)

            prescan_width = width
            prescan_height = height
            if prescan_width <= 0 or prescan_height <= 0:
                probe = ffprobe_stream_info(str(video_path))
                prescan_width = prescan_width or int(probe.get("width", 0))
                prescan_height = prescan_height or int(probe.get("height", 0))

            if prescan_width <= 0 or prescan_height <= 0:
                print(
                    "[WARN] Dimensions introuvables pour le prescan turbo, fallback scan direct.",
                    file=sys.stderr,
                    flush=True,
                )
                use_turbo = False
            else:
                scan_width, scan_height = compute_scaled_dimensions(
                    source_width=prescan_width,
                    source_height=prescan_height,
                    max_width=args.turbo_scale_width,
                )
                should_scale = scan_width < prescan_width
                scale_width = scan_width if should_scale else None
                if not should_scale:
                    scan_width = prescan_width
                    scan_height = prescan_height

                prescan_fps = max(args.turbo_prescan_fps, args.scan_fps)
                prescan_roi = scale_roi(
                    resolved_roi,
                    src_width=prescan_width,
                    src_height=prescan_height,
                    dst_width=scan_width,
                    dst_height=scan_height,
                )
                prescan_cmd = build_ffmpeg_command(
                    input_source=str(video_path),
                    fps_scan=prescan_fps,
                    full_scan=False,
                    scale_width=scale_width,
                    pixel_format="gray",
                )
                print(
                    f"[INFO] Turbo pass 1/2: prescan detect-only | fps={prescan_fps:.1f} | "
                    f"size={scan_width}x{scan_height} | backend={turbo_prescan_backend_name()} | "
                    f"workers={args.threads}",
                    flush=True,
                )
                (
                    turbo_windows,
                    turbo_hit_spans,
                    turbo_prescan_frames,
                    turbo_prescan_hits,
                    turbo_prescan_detect_calls,
                ) = prescan_candidate_windows_from_pipe(
                    ffmpeg_cmd=prescan_cmd,
                    width=scan_width,
                    height=scan_height,
                    fps_for_timestamps=prescan_fps,
                    duration_seconds=duration_seconds,
                    window_padding_seconds=args.precise_window,
                    merge_gap_seconds=args.turbo_merge_gap,
                    gray_input=True,
                    motion_threshold=args.turbo_motion_threshold,
                    max_skip_frames=args.turbo_max_skip_frames,
                    workers=args.threads,
                    roi=prescan_roi,
                )
                print(
                    f"[INFO] Turbo pass 1 done | frames={turbo_prescan_frames} | "
                    f"detect_calls={turbo_prescan_detect_calls} | confirmed_clusters={turbo_prescan_hits} | "
                    f"candidate_windows={len(turbo_windows)}",
                    flush=True,
                )

                precise_windows = build_precise_windows_from_hit_spans(
                    hit_spans=turbo_hit_spans,
                    prescan_fps=prescan_fps,
                    max_padding_seconds=args.precise_window,
                    merge_gap_seconds=args.turbo_merge_gap,
                    max_end_seconds=duration_seconds,
                )
                if precise_windows:
                    turbo_windows = precise_windows

                if turbo_windows:
                    turbo_windows.sort(key=lambda item: item[0])
                    merged_windows: list[tuple[float, float]] = []
                    for start_s, end_s in turbo_windows:
                        start = max(0.0, float(start_s))
                        end = max(start, float(end_s))
                        if duration_seconds > 0:
                            start = min(start, duration_seconds)
                            end = min(end, duration_seconds)
                        if end <= start:
                            continue
                        if not merged_windows:
                            merged_windows.append((start, end))
                            continue
                        last_start, last_end = merged_windows[-1]
                        if start <= (last_end + args.turbo_merge_gap):
                            merged_windows[-1] = (last_start, max(last_end, end))
                        else:
                            merged_windows.append((start, end))
                    turbo_windows = optimize_precise_scan_windows(
                        merged_windows,
                        base_merge_gap_seconds=args.turbo_merge_gap,
                        max_end_seconds=duration_seconds,
                    )

                if turbo_windows:
                    precise_fps = source_fps or 30.0
                    print(
                        f"[INFO] Turbo pass 2/2: precise segment scan | fps={precise_fps:.3f} | "
                        f"resolution={resolution_text} | windows={len(turbo_windows)}",
                        flush=True,
                    )
                    try:
                        results, turbo_precise_frames = (
                            scan_video_windows_from_file_with_ffmpeg(
                                video_path=video_path,
                                windows=turbo_windows,
                                width=width,
                                height=height,
                                fps_for_timestamps=precise_fps,
                                detector=detector,
                                images_dir=images_dir,
                                min_interval=args.min_interval,
                                duration_seconds=duration_seconds,
                                workers=args.threads,
                                announced_links=announced_links,
                                roi=resolved_roi,
                                notifier=notifier,
                            )
                        )
                        frames_analyzed = turbo_prescan_frames + turbo_precise_frames
                        mode_used = "turbo-precise(local-file)"
                    except Exception as precise_exc:
                        print(
                            f"[WARN] Turbo segment scan failed, fallback VideoCapture: {precise_exc}",
                            file=sys.stderr,
                            flush=True,
                        )
                        results, turbo_precise_frames = scan_local_video(
                            video_path=video_path,
                            fps_scan=max(args.scan_fps, 20.0),
                            full_scan=True,
                            detector=detector,
                            images_dir=images_dir,
                            min_interval=args.min_interval,
                            windows=turbo_windows,
                            workers=args.threads,
                            announced_links=announced_links,
                            roi=resolved_roi,
                            notifier=notifier,
                        )
                        frames_analyzed = turbo_prescan_frames + turbo_precise_frames
                        mode_used = "turbo-precise(local-file-fallback)"
                else:
                    print(
                        "[INFO] Aucun segment turbo retenu, fallback sur le scan local direct.",
                        flush=True,
                    )
                    use_turbo = False

                turbo_elapsed_seconds = max(0.0, time.monotonic() - turbo_started_at)

        if not use_turbo:
            print("[INFO] Mode: local-file", flush=True)
            results, frames_analyzed = scan_local_video(
                video_path=video_path,
                fps_scan=args.scan_fps,
                full_scan=args.full_scan,
                detector=detector,
                images_dir=images_dir,
                min_interval=args.min_interval,
                workers=args.threads,
                announced_links=announced_links,
                roi=resolved_roi,
                notifier=notifier,
            )
            mode_used = "local-file"
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return EXIT_ANALYSIS_ERROR
    finally:
        notifier.stop()

    csv_path, json_path = write_reports(
        results, output_dir, output_json_path=output_json_path
    )

    print("\n=== RESUME ===", flush=True)
    print(f"Titre: {title}", flush=True)
    print(f"Resolution choisie: {resolution_text}", flush=True)
    print(f"Mode utilise: {mode_used}", flush=True)
    print(f"Frames analysees: {frames_analyzed}", flush=True)
    if turbo_prescan_frames > 0:
        print(
            f"Turbo pass1 frames: {turbo_prescan_frames} | detect_calls: {turbo_prescan_detect_calls} | "
            f"confirmed clusters: {turbo_prescan_hits} | windows: {len(turbo_windows)}",
            flush=True,
        )
        print(f"Turbo pass2 frames: {turbo_precise_frames}", flush=True)
        if turbo_elapsed_seconds is not None:
            print(
                f"Turbo temps total: {format_elapsed(turbo_elapsed_seconds)}",
                flush=True,
            )
    summary_stats = collect_qr_summary_stats(results, images_dir)
    print(f"QR groupes trouves: {summary_stats['grouped_results']}", flush=True)
    print(
        f"QR groupes decodes: {summary_stats['decoded_groups']} | "
        f"QR groupes non decodes: {summary_stats['undecoded_groups']}",
        flush=True,
    )
    print(
        f"QR uniques par contenu: {summary_stats['unique_contents']}",
        flush=True,
    )
    if summary_stats["undecoded_groups"] > 0:
        print(
            f"QR non decodes approx uniques (image/hash): "
            f"{summary_stats['unique_null_visuals']}",
            flush=True,
        )
    summary_links = collect_unique_qr_links(results)
    print(f"Liens QR uniques: {summary_stats['unique_links']}", flush=True)
    if summary_links:
        print("Liens QR uniques:", flush=True)
        for link in summary_links:
            print(f"  {link}", flush=True)
    print(f"Images QR: {images_dir}", flush=True)
    print(f"CSV: {csv_path}", flush=True)
    print(f"JSON: {json_path}", flush=True)
    return EXIT_OK


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[ERROR] Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
