#!/usr/bin/env python3
"""
Scan a published YouTube video and extract QR codes with timestamps.

Features:
- Metadata and format selection through yt-dlp Python API
- Stream mode (ffmpeg pipe) and download mode (temporary local file)
- Multi-QR detection with OpenCV detectAndDecodeMulti
- Deduplication using decoded text or image hash
- CSV + JSON export with frame and timestamp details
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import cv2
import numpy as np
import yt_dlp
from yt_dlp.utils import DownloadError

try:
    import zxingcpp
except ImportError:  # pragma: no cover - optional acceleration backend
    zxingcpp = None


EXIT_OK = 0
EXIT_METADATA_ERROR = 2
EXIT_ANALYSIS_ERROR = 3
EXIT_DEPENDENCY_ERROR = 4


_THREAD_STATE = threading.local()


@dataclass
class QRResult:
    """Serializable result for one unique QR detection."""

    timestamp_seconds: float
    timestamp_hhmmss_ms: str
    qr_content: Optional[str]
    image_file: str
    frame_number: int


def get_video_info(
    url: str,
    *,
    cookies_file: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,
    js_runtimes: Optional[dict[str, dict[str, str]]] = None,
    remote_components: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Fetch video metadata and available formats with yt-dlp, without downloading.
    """
    ydl_opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "skip_download": True,
        "extract_flat": False,
        "retries": 3,
        "fragment_retries": 3,
        "socket_timeout": 20,
    }
    apply_yt_dlp_auth_options(
        ydl_opts,
        cookies_file=cookies_file,
        cookies_from_browser=cookies_from_browser,
    )
    apply_yt_dlp_runtime_options(
        ydl_opts,
        js_runtimes=js_runtimes,
        remote_components=remote_components,
    )
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except DownloadError as exc:
        raise RuntimeError(f"Unable to retrieve video info: {exc}") from exc
    except Exception as exc:  # pragma: no cover - safety net
        raise RuntimeError(f"Unexpected metadata error: {exc}") from exc

    if not info:
        raise RuntimeError("No metadata returned by yt-dlp.")
    return info


def parse_cookies_from_browser_spec(
    spec: Optional[str],
) -> Optional[tuple[str, Optional[str], Optional[str], Optional[str]]]:
    """Parse yt-dlp browser cookie spec into the Python API tuple format."""
    if not spec:
        return None

    match = re.fullmatch(
        r"""(?x)
        (?P<name>[^+:]+)
        (?:\s*\+\s*(?P<keyring>[^:]+))?
        (?:\s*:\s*(?!:)(?P<profile>.+?))?
        (?:\s*::\s*(?P<container>.+))?
        """,
        spec.strip(),
    )
    if match is None:
        raise ValueError(f"Invalid --cookies-from-browser value: {spec!r}")

    browser_name, keyring, profile, container = match.group("name", "keyring", "profile", "container")
    return (
        browser_name.lower(),
        profile or None,
        keyring.upper() if keyring else None,
        container or None,
    )


def apply_yt_dlp_auth_options(
    ydl_opts: dict[str, Any],
    *,
    cookies_file: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,
) -> dict[str, Any]:
    """Attach cookie-related authentication options to yt-dlp opts."""
    if cookies_file:
        ydl_opts["cookiefile"] = str(Path(cookies_file).expanduser())
    if cookies_from_browser:
        ydl_opts["cookiesfrombrowser"] = parse_cookies_from_browser_spec(cookies_from_browser)
    return ydl_opts


def parse_js_runtime_specs(specs: Optional[list[str]]) -> dict[str, dict[str, str]]:
    """Parse yt-dlp JS runtime specs like ['deno', 'node:C:\\path\\node.exe'].""" 
    runtimes: dict[str, dict[str, str]] = {}
    for raw_spec in specs or []:
        spec = str(raw_spec).strip()
        if not spec:
            continue
        name, sep, path_value = spec.partition(":")
        name = name.strip().lower()
        if not name:
            raise ValueError(f"Invalid --js-runtime value: {raw_spec!r}")
        config: dict[str, str] = {}
        if sep and path_value.strip():
            config["path"] = path_value.strip()
        runtimes[name] = config
    return runtimes


def auto_detect_js_runtimes() -> dict[str, dict[str, str]]:
    """Auto-detect supported yt-dlp JS runtimes available in PATH."""
    detected: dict[str, dict[str, str]] = {}
    for runtime_name in ("deno", "node", "quickjs", "bun"):
        runtime_path = shutil.which(runtime_name)
        if runtime_path:
            detected[runtime_name] = {"path": runtime_path}
    return detected


def apply_yt_dlp_runtime_options(
    ydl_opts: dict[str, Any],
    *,
    js_runtimes: Optional[dict[str, dict[str, str]]] = None,
    remote_components: Optional[list[str]] = None,
) -> dict[str, Any]:
    """Attach yt-dlp JS runtime and remote component options."""
    if js_runtimes:
        ydl_opts["js_runtimes"] = js_runtimes
    if remote_components:
        ydl_opts["remote_components"] = list(dict.fromkeys(remote_components))
    elif js_runtimes:
        ydl_opts["remote_components"] = ["ejs:github"]
    return ydl_opts


def select_best_video_format(
    info: dict[str, Any], max_height: Optional[int] = None
) -> Optional[dict[str, Any]]:
    """
    Pick the best available format that contains a video stream.
    """
    formats = info.get("formats") or []
    candidates: list[dict[str, Any]] = []
    for fmt in formats:
        if fmt.get("vcodec") in (None, "none"):
            continue
        if not fmt.get("url"):
            continue
        if fmt.get("has_drm") is True:
            continue
        candidates.append(fmt)

    if not candidates:
        return None

    filtered = candidates
    if max_height is not None and max_height > 0:
        constrained = [fmt for fmt in candidates if int(fmt.get("height") or 0) <= max_height]
        if constrained:
            filtered = constrained

    def sort_key(fmt: dict[str, Any]) -> tuple[float, float, float, float]:
        height = float(fmt.get("height") or 0.0)
        width = float(fmt.get("width") or 0.0)
        fps = float(fmt.get("fps") or 0.0)
        bitrate = float(fmt.get("tbr") or fmt.get("vbr") or 0.0)
        return (height, width, fps, bitrate)

    return max(filtered, key=sort_key)


def build_download_format_selector(
    selected_format: dict[str, Any], max_height: Optional[int] = None
) -> str:
    """
    Build a resilient yt-dlp format selector for temporary local downloads.
    """
    exact_id = str(selected_format.get("format_id") or "").strip()
    selectors: list[str] = []

    if exact_id:
        selectors.append(exact_id)

    if max_height is not None and max_height > 0:
        selectors.append(f"bestvideo*[height<={max_height}]")
        selectors.append(f"best*[height<={max_height}]")

    selectors.append("bestvideo*")
    selectors.append("best*")

    # Preserve order while removing duplicates.
    unique_selectors = list(dict.fromkeys(item for item in selectors if item))
    return "/".join(unique_selectors)


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
    """
    Build ffmpeg command that emits raw BGR frames to stdout.
    """
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
    """Parse fps value from either numeric or ratio string."""
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
    Probe width/height/fps for a stream using ffprobe.
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
    width = float(stream0.get("width") or 0.0)
    height = float(stream0.get("height") or 0.0)
    fps = parse_fps(stream0.get("r_frame_rate"))
    return {"width": width, "height": height, "fps": fps}


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


def zxing_position_to_quad(barcode: Any) -> Optional[np.ndarray]:
    """Convert a zxing-cpp barcode position to a (4, 2) float32 quad."""
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


def is_plausible_zxing_candidate(barcode: Any, frame_shape: tuple[int, int]) -> bool:
    """
    Filter weak zxing-cpp error results before confirming them with OpenCV.

    This keeps the turbo prescan fast while avoiding obvious false positives
    from decoder error objects that do not resemble a QR geometry.
    """
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
    if zxingcpp is None:
        return []
    return zxingcpp.read_barcodes(
        image,
        formats=zxingcpp.BarcodeFormat.QRCode,
        try_rotate=True,
        try_downscale=False,
        try_invert=True,
        return_errors=True,
    )


def select_best_plausible_zxing_quad(
    results: list[Any], frame_shape: tuple[int, int]
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
    return any(bool(getattr(barcode, "valid", False)) for barcode in results)


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
            upscaled_roi = cv2.resize(roi, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            frames_to_try.append(upscaled_roi)

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
        probes.append(cv2.resize(roi, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC))
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


def prescan_detect_score(gray_frame: np.ndarray) -> int:
    """
    Fast-but-recall-oriented QR score for turbo pass 1.

    Returns:
    - 0: no plausible QR signal
    - 1: weak but confirmed signal
    - 2: strong signal (valid decode during prescan)
    """
    if zxingcpp is not None:
        results = prescan_zxing_scan(gray_frame)
        if prescan_valid_decode_found(results):
            return 2
        weak_quad = select_best_plausible_zxing_quad(results, gray_frame.shape[:2])
        if weak_quad is not None and opencv_prescan_confirm_candidate(gray_frame, weak_quad):
            return 1

        boosted = get_thread_clahe().apply(gray_frame)
        results = prescan_zxing_scan(boosted)
        if prescan_valid_decode_found(results):
            return 2
        weak_quad = select_best_plausible_zxing_quad(results, boosted.shape[:2])
        if weak_quad is not None and opencv_prescan_confirm_candidate(boosted, weak_quad):
            return 1

        height, width = gray_frame.shape[:2]
        if max(width, height) < 1280:
            upscaled = cv2.resize(gray_frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
            results = prescan_zxing_scan(upscaled)
            if prescan_valid_decode_found(results):
                return 2
            weak_quad = select_best_plausible_zxing_quad(results, upscaled.shape[:2])
            if weak_quad is not None and opencv_prescan_confirm_candidate(upscaled, weak_quad):
                return 1
        return 0

    if opencv_prescan_detect_presence(gray_frame):
        return 1

    boosted = get_thread_clahe().apply(gray_frame)
    if opencv_prescan_detect_presence(boosted):
        return 1

    height, width = gray_frame.shape[:2]
    if max(width, height) < 1280:
        upscaled = cv2.resize(gray_frame, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
        if opencv_prescan_detect_presence(upscaled):
            return 1

    return 0


def compute_scaled_dimensions(
    source_width: int, source_height: int, max_width: int
) -> tuple[int, int]:
    """
    Compute downscaled dimensions preserving aspect ratio.
    """
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
    """
    Build and merge candidate time windows around detected timestamps.
    """
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


def build_precise_windows_from_hit_spans(
    hit_spans: list[tuple[float, float]],
    prescan_fps: float,
    max_padding_seconds: float,
    merge_gap_seconds: float,
    max_end_seconds: float,
) -> list[tuple[float, float]]:
    """
    Build tighter precise-pass windows from prescan hit spans.

    This preserves the exact timestamp search area around each accepted cluster
    while avoiding large symmetric windows that make phase 2 unnecessarily slow.
    """
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
    """
    Keep only temporally consistent prescan signals.

    A single strong signal (score=2) is accepted.
    Weak signals (score=1) must appear at least twice in a short interval.
    Returning one representative timestamp per accepted cluster also reduces
    the number of windows sent to phase 2.
    """
    if not hit_events:
        return [], 0

    events = sorted((float(ts), int(score)) for ts, score in hit_events if score > 0)
    if not events:
        return [], 0

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
        ts_center = sum(ts * max(score, 1) for ts, score in cluster_events) / float(total_weight)
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
) -> tuple[list[tuple[float, float]], list[tuple[float, float]], int, int, int]:
    """
    Fast pass:
    - runs detectMulti (no decode) on downscaled frames
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
    pending: dict[concurrent.futures.Future[int], float] = {}

    def drain_completed(wait_for_all: bool = False, block_until_done: bool = False) -> None:
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
            ts_done = pending.pop(future)
            completed_calls += 1
            try:
                score = int(future.result())
                if score > 0:
                    raw_hit_frames += 1
                    hit_events.append((ts_done, score))
            except Exception as exc:
                print(f"[WARN] Prescan worker failed: {exc}", file=sys.stderr, flush=True)

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
                                    f"done={completed_calls} | t={short_timestamp(ts)} | "
                                    f"raw_hits={raw_hit_frames}",
                                    flush=True,
                                )
                                last_progress = now
                            continue

                        if gray_input:
                            frame_gray = np.frombuffer(chunk, dtype=np.uint8).reshape((height, width))
                        else:
                            frame_bgr = np.frombuffer(chunk, dtype=np.uint8).reshape((height, width, 3))
                            frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

                        run_detection = True
                        if motion_gating_enabled:
                            thumb = cv2.resize(frame_gray, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
                            thumb_quantized = np.right_shift(thumb, 4)
                            if prev_thumb_quantized is not None and np.array_equal(
                                thumb_quantized, prev_thumb_quantized
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
                                hot_diff_threshold = max(6, int(round(motion_threshold * 4.0)))
                                hot_ratio = float(np.count_nonzero(diff >= hot_diff_threshold)) / float(diff.size)
                                peak_diff = int(np.max(diff))
                                extended_skip_limit = (
                                    max_skip_frames * 2
                                    if hot_ratio < 0.002 and peak_diff < hot_diff_threshold
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
                            future = executor.submit(prescan_detect_score, frame_gray)
                            pending[future] = ts
                            while len(pending) >= max_inflight:
                                drain_completed(wait_for_all=False, block_until_done=True)
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
        print(f"[WARN] ffmpeg exited with status {process.returncode}", file=sys.stderr, flush=True)

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


def ensure_uint8_image(image: Any) -> Optional[np.ndarray]:
    """Normalize image dtype/shape to uint8 for saving/hash operations."""
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
    """Normalize OpenCV points output to a list of (4,2) float32 arrays."""
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
    """
    Order four points as top-left, top-right, bottom-right, bottom-left.
    """
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).ravel()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def sanitize_candidate_quad(frame_shape: tuple[int, ...], quad: Any) -> Optional[np.ndarray]:
    """Reject implausible QR quads that would lead to bogus huge allocations."""
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
    """Perspective-warp QR area from the 4-corner polygon."""
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


def extract_candidates_from_frame(
    frame: np.ndarray, detector: Optional[cv2.QRCodeDetector] = None
) -> list[tuple[Optional[str], np.ndarray, Optional[np.ndarray]]]:
    """
    Return QR candidates as (decoded_text_or_none, points_4x2, straight_qr_or_none).
    """
    detector = detector or get_thread_qr_detector()
    candidates: list[tuple[Optional[str], np.ndarray, Optional[np.ndarray]]] = []

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

    # Fallback to single-code API.
    try:
        single_text, single_points, single_straight = detector.detectAndDecode(frame)
    except cv2.error:
        single_text, single_points, single_straight = "", None, None

    single_points_list = normalize_points(single_points)
    if single_points_list:
        text = single_text.strip() if isinstance(single_text, str) else ""
        candidates.append((text or None, single_points_list[0], ensure_uint8_image(single_straight)))
        return candidates

    # Last fallback: detection without decoding, still useful for undecodable QR.
    try:
        ok_detect, detect_points = detector.detectMulti(frame)
    except cv2.error:
        ok_detect, detect_points = False, None

    if ok_detect:
        for quad in normalize_points(detect_points):
            candidates.append((None, quad, None))
    return candidates


def extract_candidates_global_rescue(
    frame: np.ndarray, detector: Optional[cv2.QRCodeDetector] = None
) -> list[tuple[Optional[str], np.ndarray, Optional[np.ndarray]]]:
    """
    Global full-frame fallback for hard QR codes:
    - upscale full frame for small QRs
    - run a contrast-enhanced variant for low-contrast/overlay cases
    """
    detector = detector or get_thread_qr_detector()
    h, w = frame.shape[:2]
    if h <= 0 or w <= 0:
        return []

    up_factor = 1.6
    target_w = int(round(w * up_factor))
    target_h = int(round(h * up_factor))
    if target_w < 32 or target_h < 32:
        return []

    frame_up = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    candidates = extract_candidates_from_frame(frame_up, detector)

    if not candidates:
        gray = cv2.cvtColor(frame_up, cv2.COLOR_BGR2GRAY)
        boosted = get_thread_clahe().apply(gray)

        decoded_info: list[Any] = []
        points_multi: Any = None
        straight_multi: Any = None
        try:
            multi_output = detector.detectAndDecodeMulti(boosted)
        except cv2.error:
            multi_output = None
        if isinstance(multi_output, tuple):
            if len(multi_output) == 4:
                _, decoded_info, points_multi, straight_multi = multi_output
            elif len(multi_output) == 3:
                decoded_info, points_multi, straight_multi = multi_output
            points_list = normalize_points(points_multi)
            for idx, quad in enumerate(points_list):
                text: Optional[str] = None
                if decoded_info and idx < len(decoded_info):
                    item = decoded_info[idx]
                    if isinstance(item, bytes):
                        item = item.decode("utf-8", errors="ignore")
                    if isinstance(item, str):
                        text = item.strip() or None
                straight = straight_item_at(straight_multi, idx)
                candidates.append((text, quad, straight))
            if not candidates:
                try:
                    ok_detect, detect_points = detector.detectMulti(boosted)
                except cv2.error:
                    ok_detect, detect_points = False, None
                if ok_detect:
                    for quad in normalize_points(detect_points):
                        candidates.append((None, quad, None))

    if not candidates:
        return []

    mapped: list[tuple[Optional[str], np.ndarray, Optional[np.ndarray]]] = []
    for content, points, straight in candidates:
        pts = np.asarray(points, dtype=np.float32) / float(up_factor)
        mapped.append((content, pts, straight))
    return mapped


def detect_frame_candidates(
    frame: np.ndarray, detector: Optional[cv2.QRCodeDetector] = None
) -> list[tuple[Optional[str], np.ndarray, Optional[np.ndarray]]]:
    """Run the full precise detection pipeline on one frame."""
    candidates = extract_candidates_from_frame(frame, detector)
    if not candidates:
        candidates = extract_candidates_global_rescue(frame, detector)
    return candidates


def make_timestamp(seconds: float) -> tuple[str, str]:
    """
    Return:
    - human timestamp hh:mm:ss.ms
    - filename-safe timestamp hh-mm-ss-ms
    """
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


def dedupe_key(content: Optional[str], straight_image: Optional[np.ndarray]) -> Optional[str]:
    """
    Build deduplication key:
    - Prefer decoded content if available
    - Otherwise hash the straight QR image
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


def save_qr_result(
    images_dir: Path,
    qr_image: Optional[np.ndarray],
    timestamp_seconds: float,
    frame_number: int,
    content: Optional[str],
    sequence_number: int,
) -> QRResult:
    """
    Save one QR PNG and return structured metadata for export.
    """
    human_ts, safe_ts = make_timestamp(timestamp_seconds)
    image_name = f"qr_{sequence_number:04d}_{safe_ts}.png"
    image_path = images_dir / image_name

    image = ensure_uint8_image(qr_image)
    if image is None:
        # Last-resort placeholder so every saved result has a PNG.
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
    )


def print_progress(frames_analyzed: int, timestamp_seconds: float, duration_seconds: float) -> None:
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
    candidates: list[tuple[Optional[str], np.ndarray, Optional[np.ndarray]]],
    images_dir: Path,
    results: list[QRResult],
    seen_last_timestamp: dict[str, float],
    min_interval: float,
    announced_links: Optional[set[str]] = None,
) -> None:
    """Dedupe, save, and append candidates already detected on one frame."""
    if not candidates:
        return

    seen_in_frame: set[str] = set()

    for _idx, (content, points, straight) in enumerate(candidates):
        safe_points = sanitize_candidate_quad(frame.shape, points)
        if safe_points is None:
            continue

        qr_image = ensure_uint8_image(straight)
        if qr_image is None:
            qr_image = warp_qr_from_points(frame, safe_points)
        if qr_image is None:
            qr_image = crop_qr_bbox(frame, safe_points)

        key = dedupe_key(content, qr_image)
        if key is None:
            geom_hash = hashlib.sha1(np.asarray(safe_points, dtype=np.float32).tobytes()).hexdigest()
            key = f"points:{geom_hash}"

        if key in seen_in_frame:
            continue
        seen_in_frame.add(key)

        last_seen = seen_last_timestamp.get(key)
        if last_seen is not None and (timestamp_seconds - last_seen) < min_interval:
            continue

        result = save_qr_result(
            images_dir=images_dir,
            qr_image=qr_image,
            timestamp_seconds=timestamp_seconds,
            frame_number=frame_number,
            content=content,
            sequence_number=len(results) + 1,
        )
        results.append(result)
        seen_last_timestamp[key] = timestamp_seconds

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
    seen_last_timestamp: dict[str, float],
    min_interval: float,
    announced_links: Optional[set[str]] = None,
) -> None:
    """
    Detect QRs on one frame, dedupe, save, and append unique results.
    """
    candidates = detect_frame_candidates(frame, detector)
    finalize_frame_candidates(
        frame=frame,
        frame_number=frame_number,
        timestamp_seconds=timestamp_seconds,
        candidates=candidates,
        images_dir=images_dir,
        results=results,
        seen_last_timestamp=seen_last_timestamp,
        min_interval=min_interval,
        announced_links=announced_links,
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
    seen_last_timestamp: Optional[dict[str, float]] = None,
    workers: int = 1,
    announced_links: Optional[set[str]] = None,
) -> tuple[list[QRResult], int]:
    """
    Scan frames emitted by ffmpeg rawvideo pipe.
    """
    if width <= 0 or height <= 0:
        raise ValueError("Invalid frame dimensions for pipe scanning.")
    if fps_for_timestamps <= 0:
        raise ValueError("Invalid fps_for_timestamps for pipe scanning.")

    frame_size = width * height * 3
    if results is None:
        results = []
    if seen_last_timestamp is None:
        seen_last_timestamp = {}
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
                concurrent.futures.Future[list[tuple[Optional[str], np.ndarray, Optional[np.ndarray]]]],
                tuple[int, int, np.ndarray],
            ] = {}
            ready: dict[
                int,
                tuple[int, np.ndarray, Optional[list[tuple[Optional[str], np.ndarray, Optional[np.ndarray]]]], Optional[Exception]],
            ] = {}

            def drain_completed(wait_for_all: bool = False, block_until_done: bool = False) -> None:
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
                        ready[seq] = (frame_index_done, frame_done, future.result(), None)
                    except Exception as exc:
                        ready[seq] = (frame_index_done, frame_done, None, exc)

                while next_commit_seq in ready:
                    frame_index_done, frame_done, candidates_done, error_done = ready.pop(next_commit_seq)
                    timestamp_done = timestamp_offset_seconds + (frame_index_done / fps_for_timestamps)
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
                            seen_last_timestamp=seen_last_timestamp,
                            min_interval=min_interval,
                            announced_links=announced_links,
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
                        print_progress(frames_analyzed, timestamp_done, duration_seconds)
                        last_progress = now
                    next_commit_seq += 1

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                while True:
                    if process.stdout is None:
                        break
                    chunk = process.stdout.read(frame_size)
                    if not chunk or len(chunk) < frame_size:
                        break

                    try:
                        frame = np.frombuffer(chunk, dtype=np.uint8).reshape((height, width, 3)).copy()
                    except Exception as exc:
                        frame_errors += 1
                        if frame_errors <= 5:
                            frame_number = frame_number_offset + frame_index + 1
                            print(f"[WARN] Skipping frame {frame_number}: {exc}", file=sys.stderr, flush=True)
                        frame_index += 1
                        continue

                    frames_analyzed += 1
                    future = executor.submit(detect_frame_candidates, frame)
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

                timestamp_seconds = timestamp_offset_seconds + (frame_index / fps_for_timestamps)
                frame_number = frame_number_offset + frame_index + 1
                frames_analyzed += 1

                try:
                    frame = np.frombuffer(chunk, dtype=np.uint8).reshape((height, width, 3))
                    process_frame_detections(
                        frame=frame,
                        frame_number=frame_number,
                        timestamp_seconds=timestamp_seconds,
                        detector=detector,
                        images_dir=images_dir,
                        results=results,
                        seen_last_timestamp=seen_last_timestamp,
                        min_interval=min_interval,
                        announced_links=announced_links,
                    )
                except Exception as exc:
                    frame_errors += 1
                    if frame_errors <= 5:
                        print(f"[WARN] Skipping frame {frame_number}: {exc}", file=sys.stderr, flush=True)

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
        print(f"[WARN] ffmpeg exited with status {process.returncode}", file=sys.stderr, flush=True)
    if frame_errors > 0:
        print(f"[WARN] {frame_errors} frame(s) failed and were skipped.", file=sys.stderr, flush=True)

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
) -> tuple[list[QRResult], int]:
    """
    Analyze only candidate windows from a local file by decoding each window
    through ffmpeg. This is faster than random seeking with VideoCapture on
    sparse windows and preserves exact global timestamps.
    """
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
    seen_last_timestamp: dict[str, float] = {}
    total_frames_analyzed = 0
    total_window_seconds = sum(max(0.0, end - start) for start, end in normalized_windows)
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
            seen_last_timestamp=seen_last_timestamp,
            workers=workers,
            announced_links=announced_links,
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
) -> tuple[list[QRResult], int]:
    """
    Scan a local video with cv2.VideoCapture.
    """
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
    seen_last_timestamp: dict[str, float] = {}
    frame_errors = 0
    frames_analyzed = 0
    last_progress = time.monotonic()
    max_workers = max(1, int(workers))
    use_parallel = max_workers > 1
    max_inflight = max_workers * 4

    def finalize_one_frame(
        frame_index: int,
        frame: np.ndarray,
        candidates: list[tuple[Optional[str], np.ndarray, Optional[np.ndarray]]],
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
                seen_last_timestamp=seen_last_timestamp,
                min_interval=min_interval,
                announced_links=announced_links,
            )
        except Exception as exc:
            frame_errors += 1
            if frame_errors <= 5:
                print(f"[WARN] Skipping frame {frame_number}: {exc}", file=sys.stderr, flush=True)

        now = time.monotonic()
        if now - last_progress >= 2.0:
            print_progress(frames_analyzed, timestamp_seconds, duration_seconds)
            last_progress = now

    def process_one_frame(frame_index: int, frame: np.ndarray, region_start_frame: int = 0) -> None:
        nonlocal frame_errors

        if not full_scan and (frame_index - region_start_frame) % frame_step != 0:
            return

        try:
            candidates = detect_frame_candidates(frame, detector)
            finalize_one_frame(frame_index, frame, candidates)
        except Exception as exc:
            frame_errors += 1
            if frame_errors <= 5:
                print(f"[WARN] Skipping frame {frame_index + 1}: {exc}", file=sys.stderr, flush=True)

    def scan_frames_parallel(frame_iter: Any) -> None:
        nonlocal frame_errors

        next_seq = 0
        next_commit_seq = 0
        pending: dict[concurrent.futures.Future[Any], tuple[int, int, np.ndarray]] = {}
        ready: dict[int, tuple[int, np.ndarray, Optional[list[tuple[Optional[str], np.ndarray, Optional[np.ndarray]]]], Optional[Exception]]] = {}

        def drain_completed(wait_for_all: bool = False, block_until_done: bool = False) -> None:
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
                frame_index_done, frame_done, candidates_done, error_done = ready.pop(next_commit_seq)
                if error_done is not None:
                    frame_errors += 1
                    if frame_errors <= 5:
                        print(
                            f"[WARN] Skipping frame {frame_index_done + 1}: {error_done}",
                            file=sys.stderr,
                            flush=True,
                        )
                else:
                    finalize_one_frame(frame_index_done, frame_done, candidates_done or [])
                next_commit_seq += 1

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for frame_index, region_start_frame, frame in frame_iter:
                if not full_scan and (frame_index - region_start_frame) % frame_step != 0:
                    continue

                future = executor.submit(detect_frame_candidates, frame)
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
                    process_one_frame(frame_index, frame, region_start_frame=region_start_frame)
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
                    process_one_frame(frame_index, frame, region_start_frame=region_start_frame)
    finally:
        capture.release()

    if frames_analyzed == 0:
        raise RuntimeError("No frame analyzed from local video.")
    if frame_errors > 0:
        print(f"[WARN] {frame_errors} frame(s) failed and were skipped.", file=sys.stderr, flush=True)

    return results, frames_analyzed


def download_video_temp(
    url: str,
    format_selector: str,
    temp_dir: Path,
    *,
    cookies_file: Optional[str] = None,
    cookies_from_browser: Optional[str] = None,
    js_runtimes: Optional[dict[str, dict[str, str]]] = None,
    remote_components: Optional[list[str]] = None,
) -> Path:
    """Download a temporary local file with a resilient yt-dlp format selector."""
    outtmpl = str(temp_dir / "source.%(ext)s")
    ydl_opts: dict[str, Any] = {
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "retries": 3,
        "fragment_retries": 3,
        "socket_timeout": 20,
        "format": format_selector,
        "outtmpl": outtmpl,
    }
    apply_yt_dlp_auth_options(
        ydl_opts,
        cookies_file=cookies_file,
        cookies_from_browser=cookies_from_browser,
    )
    apply_yt_dlp_runtime_options(
        ydl_opts,
        js_runtimes=js_runtimes,
        remote_components=remote_components,
    )

    info: Optional[dict[str, Any]] = None
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            prepared = Path(ydl.prepare_filename(info))
    except DownloadError as exc:
        raise RuntimeError(f"Temporary download failed: {exc}") from exc
    except Exception as exc:
        raise RuntimeError(f"Unexpected download error: {exc}") from exc

    candidates: list[Path] = []
    if info:
        requested = info.get("requested_downloads") or []
        for item in requested:
            file_path = item.get("filepath") or item.get("filename")
            if file_path:
                candidates.append(Path(file_path))
    candidates.append(prepared)
    candidates.extend(temp_dir.glob("source.*"))

    for path in candidates:
        if path.exists() and path.is_file():
            return path

    raise RuntimeError("yt-dlp reported success but no local video file was found.")


def get_local_video_metadata(video_path: Path) -> dict[str, Any]:
    """Probe a local video file for title, dimensions, fps, and duration."""
    if not video_path.exists() or not video_path.is_file():
        raise RuntimeError(f"Local video file not found: {video_path}")

    probe = ffprobe_stream_info(str(video_path))
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open local video: {video_path}")

    try:
        width = int(probe.get("width", 0)) or int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(probe.get("height", 0)) or int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        fps = float(probe.get("fps", 0.0)) or float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
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


def write_reports(results: list[QRResult], output_dir: Path) -> tuple[Path, Path]:
    """Write CSV + JSON reports."""
    csv_path = output_dir / "qr_results.csv"
    json_path = output_dir / "qr_results.json"

    fieldnames = [
        "timestamp_seconds",
        "timestamp_hhmmss_ms",
        "qr_content",
        "image_file",
        "frame_number",
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            row = asdict(result)
            row["timestamp_seconds"] = f"{result.timestamp_seconds:.6f}"
            row["qr_content"] = result.qr_content if result.qr_content is not None else "null"
            writer.writerow(row)

    json_payload: list[dict[str, Any]] = []
    for result in results:
        json_payload.append(
            {
                "timestamp_seconds": result.timestamp_seconds,
                "timestamp_hhmmss_ms": result.timestamp_hhmmss_ms,
                "qr_content": result.qr_content,
                "image_file": result.image_file,
                "frame_number": result.frame_number,
            }
        )

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(json_payload, handle, ensure_ascii=False, indent=2)

    return csv_path, json_path


def parse_args() -> argparse.Namespace:
    """CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze a YouTube video URL or a local video file and extract unique QR codes with timestamps."
    )
    parser.add_argument("url", nargs="?", help="Published YouTube URL to scan.")
    parser.add_argument(
        "--video-file",
        default=None,
        help="Path to a local video file to scan without using YouTube or yt-dlp.",
    )
    parser.add_argument(
        "--fps-scan",
        type=float,
        default=10.0,
        help="Frame rate used for scan sampling (ignored by --full-scan).",
    )
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="Force dense scan (analyze all frames).",
    )
    parser.add_argument(
        "--output",
        default="qr_results",
        help="Output directory for images and reports.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep a copy of temporary downloaded video in output directory (download mode).",
    )
    parser.add_argument(
        "--min-interval",
        type=float,
        default=1.0,
        help="Minimum time interval (seconds) for duplicate suppression.",
    )
    parser.add_argument(
        "--prefer-download",
        action="store_true",
        help="Prefer local download mode before stream mode.",
    )
    parser.add_argument(
        "--cookies",
        default=None,
        help="Path to a Netscape-format cookies.txt file for yt-dlp authentication.",
    )
    parser.add_argument(
        "--cookies-from-browser",
        default=None,
        help=(
            "Load YouTube cookies from a browser, e.g. 'chrome', 'edge', "
            "'firefox', or 'chrome:Default'."
        ),
    )
    parser.add_argument(
        "--js-runtime",
        action="append",
        default=None,
        help=(
            "Enable a yt-dlp JavaScript runtime, e.g. 'deno', 'node', or "
            "'node:C:\\\\Program Files\\\\nodejs\\\\node.exe'. Can be repeated."
        ),
    )
    parser.add_argument(
        "--remote-component",
        action="append",
        default=None,
        help=(
            "Allow yt-dlp remote components such as 'ejs:github'. "
            "Can be repeated."
        ),
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=0,
        help=(
            "Maximum source video height to request from YouTube (e.g. 1080). "
            "Use 0 to disable limit."
        ),
    )
    parser.add_argument(
        "--turbo-precise",
        action="store_true",
        help=(
            "Two-pass mode: fast prescan on downscaled frames, then precise decode "
            "only on candidate windows."
        ),
    )
    parser.add_argument(
        "--turbo-prescan-fps",
        type=float,
        default=24.0,
        help="Frame rate for turbo prescan detect-only pass.",
    )
    parser.add_argument(
        "--turbo-scale-width",
        type=int,
        default=960,
        help="Max frame width used during turbo prescan (downscaled for speed).",
    )
    parser.add_argument(
        "--turbo-window",
        type=float,
        default=0.75,
        help="Padding (seconds) around each prescan detection for precise pass.",
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
        help="Motion gating threshold (mean abs diff on thumbnails) for turbo prescan.",
    )
    parser.add_argument(
        "--turbo-max-skip-frames",
        type=int,
        default=2,
        help="Maximum consecutive low-motion frames skipped during turbo prescan.",
    )
    parser.add_argument(
        "--turbo-prescan-color",
        action="store_true",
        help="Force color frames for turbo prescan (default uses faster grayscale).",
    )
    parser.add_argument(
        "--turbo-workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Worker threads used by turbo pass 1.",
    )
    parser.add_argument(
        "--turbo-start-guard",
        type=float,
        default=8.0,
        help=(
            "Always include the first N seconds in precise pass "
            "(safety net for intro QR codes)."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    local_input_video = Path(args.video_file).expanduser().resolve() if args.video_file else None

    if args.url and local_input_video is not None:
        print("[ERROR] Use either a YouTube URL or --video-file, not both.", file=sys.stderr)
        return EXIT_ANALYSIS_ERROR
    if not args.url and local_input_video is None:
        print("[ERROR] Provide a YouTube URL or --video-file.", file=sys.stderr)
        return EXIT_ANALYSIS_ERROR

    if args.fps_scan <= 0:
        print("[ERROR] --fps-scan must be > 0.", file=sys.stderr)
        return EXIT_ANALYSIS_ERROR
    if args.min_interval < 0:
        print("[ERROR] --min-interval must be >= 0.", file=sys.stderr)
        return EXIT_ANALYSIS_ERROR
    if args.max_height < 0:
        print("[ERROR] --max-height must be >= 0.", file=sys.stderr)
        return EXIT_ANALYSIS_ERROR
    if args.turbo_prescan_fps <= 0:
        print("[ERROR] --turbo-prescan-fps must be > 0.", file=sys.stderr)
        return EXIT_ANALYSIS_ERROR
    if args.turbo_scale_width <= 0:
        print("[ERROR] --turbo-scale-width must be > 0.", file=sys.stderr)
        return EXIT_ANALYSIS_ERROR
    if args.turbo_window < 0:
        print("[ERROR] --turbo-window must be >= 0.", file=sys.stderr)
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
    if args.turbo_workers <= 0:
        print("[ERROR] --turbo-workers must be > 0.", file=sys.stderr)
        return EXIT_ANALYSIS_ERROR
    if args.turbo_start_guard < 0:
        print("[ERROR] --turbo-start-guard must be >= 0.", file=sys.stderr)
        return EXIT_ANALYSIS_ERROR

    try:
        yt_js_runtimes = (
            parse_js_runtime_specs(args.js_runtime)
            if args.js_runtime
            else auto_detect_js_runtimes()
        )
    except ValueError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return EXIT_ANALYSIS_ERROR

    yt_remote_components = list(dict.fromkeys(args.remote_component or []))
    if shutil.which("ffmpeg") is None:
        print("[ERROR] ffmpeg was not found in PATH.", file=sys.stderr)
        return EXIT_DEPENDENCY_ERROR

    output_dir = Path(args.output).expanduser().resolve()
    images_dir = output_dir / "qr_images"
    images_dir.mkdir(parents=True, exist_ok=True)

    if local_input_video is not None:
        try:
            local_meta = get_local_video_metadata(local_input_video)
        except Exception as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return EXIT_METADATA_ERROR

        title = str(local_meta.get("title") or local_input_video.stem)
        duration_seconds = float(local_meta.get("duration") or 0.0)
        width = int(local_meta.get("width") or 0)
        height = int(local_meta.get("height") or 0)
        source_fps = float(local_meta.get("fps") or 0.0)
        resolution_text = f"{width}x{height}" if width > 0 and height > 0 else "unknown"

        print(f"[INFO] Local video file: {local_input_video}", flush=True)
        print(f"[INFO] Titre: {title}", flush=True)
        print(f"[INFO] Resolution source: {resolution_text}", flush=True)

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
        turbo_analysis_started_at: Optional[float] = None
        turbo_elapsed_seconds: Optional[float] = None

        try:
            if args.turbo_precise:
                turbo_analysis_started_at = time.monotonic()
                print("[INFO] Mode: turbo-precise (local file only)", flush=True)
                prescan_skip_seconds = max(0.0, float(args.turbo_start_guard))
                if duration_seconds > 0:
                    prescan_skip_seconds = min(prescan_skip_seconds, duration_seconds)

                prescan_width = width
                prescan_height = height
                if prescan_width <= 0 or prescan_height <= 0:
                    local_probe = ffprobe_stream_info(str(local_input_video))
                    prescan_width = prescan_width or int(local_probe.get("width", 0))
                    prescan_height = prescan_height or int(local_probe.get("height", 0))

                if prescan_width > 0 and prescan_height > 0:
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

                    prescan_fps = max(args.turbo_prescan_fps, args.fps_scan)
                    prescan_cmd = build_ffmpeg_command(
                        input_source=str(local_input_video),
                        fps_scan=prescan_fps,
                        full_scan=False,
                        scale_width=scale_width,
                        pixel_format="bgr24" if args.turbo_prescan_color else "gray",
                    )
                    print(
                        f"[INFO] Turbo pass 1/2: prescan detect-only | source=local-file | "
                        f"fps={prescan_fps:.1f} | size={scan_width}x{scan_height} | pixfmt="
                        f"{'bgr24' if args.turbo_prescan_color else 'gray'} | "
                        f"backend={turbo_prescan_backend_name()} | workers={args.turbo_workers} | "
                        f"skip_first={prescan_skip_seconds:.2f}s",
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
                        window_padding_seconds=args.turbo_window,
                        merge_gap_seconds=args.turbo_merge_gap,
                        gray_input=not args.turbo_prescan_color,
                        motion_threshold=args.turbo_motion_threshold,
                        max_skip_frames=args.turbo_max_skip_frames,
                        workers=args.turbo_workers,
                        skip_detection_until_seconds=prescan_skip_seconds,
                    )
                    print(
                        f"[INFO] Turbo pass 1 done | frames={turbo_prescan_frames} | "
                        f"detect_calls={turbo_prescan_detect_calls} | confirmed_clusters={turbo_prescan_hits} | "
                        f"candidate_windows={len(turbo_windows)}",
                        flush=True,
                    )
                else:
                    print(
                        "[WARN] Unable to resolve dimensions for turbo pass 1; prescan skipped.",
                        file=sys.stderr,
                        flush=True,
                    )

                guard_end = prescan_skip_seconds
                precise_windows = build_precise_windows_from_hit_spans(
                    hit_spans=turbo_hit_spans,
                    prescan_fps=max(args.turbo_prescan_fps, args.fps_scan),
                    max_padding_seconds=args.turbo_window,
                    merge_gap_seconds=args.turbo_merge_gap,
                    max_end_seconds=duration_seconds,
                )
                if precise_windows:
                    turbo_windows = precise_windows
                if guard_end > 0:
                    turbo_windows.append((0.0, guard_end))

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
                    turbo_windows = merged_windows

                if turbo_windows:
                    precise_fps = source_fps or 30.0
                    print(
                        f"[INFO] Turbo pass 2/2: precise segment scan | fps={precise_fps:.3f} | "
                        f"resolution={resolution_text} | windows={len(turbo_windows)}",
                        flush=True,
                    )
                    try:
                        results, turbo_precise_frames = scan_video_windows_from_file_with_ffmpeg(
                            video_path=local_input_video,
                            windows=turbo_windows,
                            width=width,
                            height=height,
                            fps_for_timestamps=precise_fps,
                            detector=detector,
                            images_dir=images_dir,
                            min_interval=args.min_interval,
                            duration_seconds=duration_seconds,
                            workers=args.turbo_workers,
                            announced_links=announced_links,
                        )
                    except Exception as precise_exc:
                        print(
                            f"[WARN] Turbo segment scan failed, falling back to VideoCapture: {precise_exc}",
                            file=sys.stderr,
                            flush=True,
                        )
                        results, turbo_precise_frames = scan_local_video(
                            video_path=local_input_video,
                            fps_scan=max(args.fps_scan, 20.0),
                            full_scan=True,
                            detector=detector,
                            images_dir=images_dir,
                            min_interval=args.min_interval,
                            windows=turbo_windows,
                            workers=args.turbo_workers,
                            announced_links=announced_links,
                        )
                else:
                    print(
                        "[INFO] Turbo pass 2 skipped: no candidate windows found in prescan.",
                        flush=True,
                    )
                    results = []
                    turbo_precise_frames = 0

                frames_analyzed = turbo_prescan_frames + turbo_precise_frames
                mode_used = "turbo-precise(local-file)"
            else:
                print("[INFO] Mode: local-file", flush=True)
                results, frames_analyzed = scan_local_video(
                    video_path=local_input_video,
                    fps_scan=args.fps_scan,
                    full_scan=args.full_scan,
                    detector=detector,
                    images_dir=images_dir,
                    min_interval=args.min_interval,
                    workers=args.turbo_workers,
                    announced_links=announced_links,
                )
                mode_used = "local-file"
            if turbo_analysis_started_at is not None:
                turbo_elapsed_seconds = max(0.0, time.monotonic() - turbo_analysis_started_at)
        except Exception as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return EXIT_ANALYSIS_ERROR

        csv_path, json_path = write_reports(results, output_dir)

        print("\n=== RESUME ===", flush=True)
        print(f"Titre: {title}", flush=True)
        print(f"Resolution choisie: {resolution_text}", flush=True)
        print(f"Mode utilise: {mode_used}", flush=True)
        print(f"Frames analysees: {frames_analyzed}", flush=True)
        if args.turbo_precise:
            print(
                f"Turbo pass1 frames: {turbo_prescan_frames} | detect_calls: {turbo_prescan_detect_calls} | "
                f"confirmed clusters: {turbo_prescan_hits} | windows: {len(turbo_windows)}",
                flush=True,
            )
            print(f"Turbo pass2 frames: {turbo_precise_frames}", flush=True)
            if turbo_elapsed_seconds is not None:
                print(f"Turbo temps total: {format_elapsed(turbo_elapsed_seconds)}", flush=True)
        print(f"QR uniques trouves: {len(results)}", flush=True)
        summary_links = collect_unique_qr_links(results)
        if summary_links:
            print("Liens QR uniques:", flush=True)
            for link in summary_links:
                print(f"  {link}", flush=True)
        print(f"Images QR: {images_dir}", flush=True)
        print(f"CSV: {csv_path}", flush=True)
        print(f"JSON: {json_path}", flush=True)
        return EXIT_OK

    if yt_js_runtimes:
        runtime_desc = ", ".join(
            f"{name}={cfg.get('path', '') or 'PATH'}" for name, cfg in yt_js_runtimes.items()
        )
        print(f"[INFO] yt-dlp JS runtimes: {runtime_desc}", flush=True)
    else:
        print(
            "[WARN] No yt-dlp JavaScript runtime found in PATH. "
            "Install Deno or Node.js if YouTube metadata extraction fails.",
            file=sys.stderr,
            flush=True,
        )

    print("[INFO] Reading YouTube metadata...", flush=True)
    try:
        info = get_video_info(
            args.url,
            cookies_file=args.cookies,
            cookies_from_browser=args.cookies_from_browser,
            js_runtimes=yt_js_runtimes,
            remote_components=yt_remote_components,
        )
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return EXIT_METADATA_ERROR

    effective_max_height: Optional[int] = None
    if args.max_height > 0:
        effective_max_height = args.max_height
    elif args.turbo_precise:
        effective_max_height = 1080

    selected = select_best_video_format(info, max_height=effective_max_height)
    if not selected:
        print("[ERROR] No readable video format found.", file=sys.stderr)
        return EXIT_METADATA_ERROR

    title = str(info.get("title") or "Unknown title")
    duration_seconds = float(info.get("duration") or 0.0)
    format_id = str(selected.get("format_id") or "unknown")
    download_format_selector = build_download_format_selector(
        selected_format=selected,
        max_height=effective_max_height,
    )
    width = int(selected.get("width") or 0)
    height = int(selected.get("height") or 0)
    source_fps = float(selected.get("fps") or 0.0)
    stream_url = selected.get("url")
    resolution_text = f"{width}x{height}" if width > 0 and height > 0 else "unknown"

    print(f"[INFO] Titre: {title}", flush=True)
    max_h_text = str(effective_max_height) if effective_max_height is not None else "none"
    print(
        f"[INFO] Format choisi: id={format_id} | resolution={resolution_text} | max_height={max_h_text}",
        flush=True,
    )

    detector = cv2.QRCodeDetector()
    mode_order = ["download", "stream"] if args.prefer_download else ["stream", "download"]
    mode_used: Optional[str] = None
    frames_analyzed = 0
    results: list[QRResult] = []
    announced_links: set[str] = set()
    failures: list[str] = []
    kept_temp_video: Optional[Path] = None
    turbo_prescan_frames = 0
    turbo_prescan_hits = 0
    turbo_prescan_detect_calls = 0
    turbo_precise_frames = 0
    turbo_windows: list[tuple[float, float]] = []
    turbo_hit_spans: list[tuple[float, float]] = []
    turbo_analysis_started_at: Optional[float] = None
    turbo_elapsed_seconds: Optional[float] = None

    if args.turbo_precise:
        temp_dir = Path(tempfile.mkdtemp(prefix="yt_qr_"))
        local_video: Optional[Path] = None
        try:
            turbo_analysis_started_at = time.monotonic()
            print("[INFO] Mode: turbo-precise (pass 1 stream, pass 2 local file)", flush=True)
            prescan_skip_seconds = max(0.0, float(args.turbo_start_guard))
            if duration_seconds > 0:
                prescan_skip_seconds = min(prescan_skip_seconds, duration_seconds)

            prescan_source: Optional[str] = str(stream_url) if stream_url else None
            prescan_source_label = "stream"
            prescan_width = width
            prescan_height = height

            if prescan_source and (prescan_width <= 0 or prescan_height <= 0):
                probed = ffprobe_stream_info(prescan_source)
                prescan_width = prescan_width or int(probed.get("width", 0))
                prescan_height = prescan_height or int(probed.get("height", 0))

            if not prescan_source or prescan_width <= 0 or prescan_height <= 0:
                print(
                    "[WARN] Unable to resolve direct media stream for turbo pass 1; "
                    "falling back to temporary local file first.",
                    file=sys.stderr,
                    flush=True,
                )
                print(f"[INFO] Download selector: {download_format_selector}", flush=True)
                local_video = download_video_temp(
                    args.url,
                    format_selector=download_format_selector,
                    temp_dir=temp_dir,
                    cookies_file=args.cookies,
                    cookies_from_browser=args.cookies_from_browser,
                    js_runtimes=yt_js_runtimes,
                    remote_components=yt_remote_components,
                )
                print(f"[INFO] Local temp file: {local_video}", flush=True)

                local_probe = ffprobe_stream_info(str(local_video))
                prescan_source = str(local_video)
                prescan_source_label = "local-file"
                prescan_width = int(local_probe.get("width", 0)) or width
                prescan_height = int(local_probe.get("height", 0)) or height

            if prescan_source and prescan_width > 0 and prescan_height > 0:
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

                prescan_fps = max(args.turbo_prescan_fps, args.fps_scan)
                prescan_cmd = build_ffmpeg_command(
                    input_source=prescan_source,
                    fps_scan=prescan_fps,
                    full_scan=False,
                    scale_width=scale_width,
                    pixel_format="bgr24" if args.turbo_prescan_color else "gray",
                )
                print(
                    f"[INFO] Turbo pass 1/2: prescan detect-only | source={prescan_source_label} | "
                    f"fps={prescan_fps:.1f} | size={scan_width}x{scan_height} | pixfmt="
                    f"{'bgr24' if args.turbo_prescan_color else 'gray'} | "
                    f"backend={turbo_prescan_backend_name()} | workers={args.turbo_workers} | "
                    f"skip_first={prescan_skip_seconds:.2f}s",
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
                    window_padding_seconds=args.turbo_window,
                    merge_gap_seconds=args.turbo_merge_gap,
                    gray_input=not args.turbo_prescan_color,
                    motion_threshold=args.turbo_motion_threshold,
                    max_skip_frames=args.turbo_max_skip_frames,
                    workers=args.turbo_workers,
                    skip_detection_until_seconds=prescan_skip_seconds,
                )
                print(
                    f"[INFO] Turbo pass 1 done | frames={turbo_prescan_frames} | "
                    f"detect_calls={turbo_prescan_detect_calls} | confirmed_clusters={turbo_prescan_hits} | "
                    f"candidate_windows={len(turbo_windows)}",
                    flush=True,
                )
            else:
                print(
                    "[WARN] Unable to resolve dimensions for turbo pass 1; prescan skipped.",
                    file=sys.stderr,
                    flush=True,
                )

            # Safety net: always rescan beginning of the video in precise mode.
            guard_end = prescan_skip_seconds
            precise_windows = build_precise_windows_from_hit_spans(
                hit_spans=turbo_hit_spans,
                prescan_fps=max(args.turbo_prescan_fps, args.fps_scan),
                max_padding_seconds=args.turbo_window,
                merge_gap_seconds=args.turbo_merge_gap,
                max_end_seconds=duration_seconds,
            )
            if precise_windows:
                turbo_windows = precise_windows
            if guard_end > 0:
                turbo_windows.append((0.0, guard_end))

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
                turbo_windows = merged_windows

            if turbo_windows:
                if local_video is None:
                    print("[INFO] Downloading local video for turbo pass 2...", flush=True)
                    print(f"[INFO] Download selector: {download_format_selector}", flush=True)
                    local_video = download_video_temp(
                        args.url,
                        format_selector=download_format_selector,
                        temp_dir=temp_dir,
                        cookies_file=args.cookies,
                        cookies_from_browser=args.cookies_from_browser,
                        js_runtimes=yt_js_runtimes,
                        remote_components=yt_remote_components,
                    )
                    print(f"[INFO] Local temp file: {local_video}", flush=True)

                local_probe = ffprobe_stream_info(str(local_video))
                local_width = int(local_probe.get("width", 0)) or width
                local_height = int(local_probe.get("height", 0)) or height
                precise_fps = float(local_probe.get("fps", 0.0)) or source_fps or 30.0

                print(
                    f"[INFO] Turbo pass 2/2: precise segment scan | fps={precise_fps:.3f} | "
                    f"resolution={local_width}x{local_height} | windows={len(turbo_windows)}",
                    flush=True,
                )
                try:
                    results, turbo_precise_frames = scan_video_windows_from_file_with_ffmpeg(
                        video_path=local_video,
                        windows=turbo_windows,
                        width=local_width,
                        height=local_height,
                        fps_for_timestamps=precise_fps,
                        detector=detector,
                        images_dir=images_dir,
                        min_interval=args.min_interval,
                        duration_seconds=duration_seconds,
                        workers=args.turbo_workers,
                        announced_links=announced_links,
                    )
                except Exception as precise_exc:
                    print(
                        f"[WARN] Turbo segment scan failed, falling back to VideoCapture: {precise_exc}",
                        file=sys.stderr,
                        flush=True,
                    )
                    results, turbo_precise_frames = scan_local_video(
                        video_path=local_video,
                        fps_scan=max(args.fps_scan, 20.0),
                        full_scan=True,
                        detector=detector,
                        images_dir=images_dir,
                        min_interval=args.min_interval,
                        windows=turbo_windows,
                        workers=args.turbo_workers,
                        announced_links=announced_links,
                    )
            else:
                print(
                    "[INFO] Turbo pass 2 skipped: no candidate windows found in prescan.",
                    flush=True,
                )
                results = []
                turbo_precise_frames = 0
            frames_analyzed = turbo_prescan_frames + turbo_precise_frames
            mode_used = "turbo-precise(stream-prescan+download-precise)"
            if turbo_analysis_started_at is not None:
                turbo_elapsed_seconds = max(0.0, time.monotonic() - turbo_analysis_started_at)

            if args.keep_temp and local_video is not None:
                kept_temp_video = output_dir / f"temp_video{local_video.suffix or '.mp4'}"
                shutil.copy2(local_video, kept_temp_video)
        except Exception as exc:
            failures.append(f"turbo-precise: {exc}")
            print(f"[WARN] Turbo mode failed: {exc}", file=sys.stderr, flush=True)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    else:
        for mode in mode_order:
            try:
                if mode == "stream":
                    if not stream_url:
                        raise RuntimeError("Selected format does not expose a media URL for stream mode.")

                    if width <= 0 or height <= 0 or (args.full_scan and source_fps <= 0):
                        probed = ffprobe_stream_info(stream_url)
                        width = width or int(probed.get("width", 0))
                        height = height or int(probed.get("height", 0))
                        if source_fps <= 0:
                            source_fps = float(probed.get("fps", 0.0))

                    if width <= 0 or height <= 0:
                        raise RuntimeError("Unable to determine stream resolution.")
                    if args.full_scan and source_fps <= 0:
                        raise RuntimeError("Unable to determine source FPS for full stream scan.")

                    fps_for_timestamps = source_fps if args.full_scan else args.fps_scan
                    resolution_text = f"{width}x{height}"
                    ffmpeg_cmd = build_ffmpeg_command(stream_url, args.fps_scan, args.full_scan)
                    print(
                        f"[INFO] Mode: stream | resolution={resolution_text} | fps_ref={fps_for_timestamps:.3f}",
                        flush=True,
                    )
                    results, frames_analyzed = scan_frames_from_pipe(
                        ffmpeg_cmd=ffmpeg_cmd,
                        width=width,
                        height=height,
                        fps_for_timestamps=fps_for_timestamps,
                        detector=detector,
                        images_dir=images_dir,
                        min_interval=args.min_interval,
                        duration_seconds=duration_seconds,
                        announced_links=announced_links,
                    )
                    mode_used = mode
                    break

                temp_dir = Path(tempfile.mkdtemp(prefix="yt_qr_"))
                try:
                    print("[INFO] Mode: download (temporary local file)", flush=True)
                    print(f"[INFO] Download selector: {download_format_selector}", flush=True)
                    local_video = download_video_temp(
                        args.url,
                        format_selector=download_format_selector,
                        temp_dir=temp_dir,
                        cookies_file=args.cookies,
                        cookies_from_browser=args.cookies_from_browser,
                        js_runtimes=yt_js_runtimes,
                        remote_components=yt_remote_components,
                    )
                    print(f"[INFO] Local temp file: {local_video}", flush=True)
                    results, frames_analyzed = scan_local_video(
                        video_path=local_video,
                        fps_scan=args.fps_scan,
                        full_scan=args.full_scan,
                        detector=detector,
                        images_dir=images_dir,
                        min_interval=args.min_interval,
                        announced_links=announced_links,
                    )
                    if args.keep_temp:
                        kept_temp_video = output_dir / f"temp_video{local_video.suffix or '.mp4'}"
                        shutil.copy2(local_video, kept_temp_video)
                    mode_used = mode
                    break
                finally:
                    shutil.rmtree(temp_dir, ignore_errors=True)

            except Exception as exc:
                failures.append(f"{mode}: {exc}")
                print(f"[WARN] Mode {mode} failed: {exc}", file=sys.stderr, flush=True)

    if mode_used is None:
        print("[ERROR] All analysis modes failed.", file=sys.stderr)
        for item in failures:
            print(f"  - {item}", file=sys.stderr)
        return EXIT_ANALYSIS_ERROR

    csv_path, json_path = write_reports(results, output_dir)

    print("\n=== RESUME ===", flush=True)
    print(f"Titre: {title}", flush=True)
    print(f"Resolution choisie: {resolution_text}", flush=True)
    print(f"Mode utilise: {mode_used}", flush=True)
    print(f"Frames analysees: {frames_analyzed}", flush=True)
    if args.turbo_precise:
        print(
            f"Turbo pass1 frames: {turbo_prescan_frames} | detect_calls: {turbo_prescan_detect_calls} | "
            f"confirmed clusters: {turbo_prescan_hits} | windows: {len(turbo_windows)}",
            flush=True,
        )
        print(f"Turbo pass2 frames: {turbo_precise_frames}", flush=True)
        if turbo_elapsed_seconds is not None:
            print(f"Turbo temps total: {format_elapsed(turbo_elapsed_seconds)}", flush=True)
    print(f"QR uniques trouves: {len(results)}", flush=True)
    summary_links = collect_unique_qr_links(results)
    if summary_links:
        print("Liens QR uniques:", flush=True)
        for link in summary_links:
            print(f"  {link}", flush=True)
    print(f"Images QR: {images_dir}", flush=True)
    print(f"CSV: {csv_path}", flush=True)
    print(f"JSON: {json_path}", flush=True)
    if kept_temp_video is not None:
        print(f"Video temporaire conservee: {kept_temp_video}", flush=True)

    return EXIT_OK


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n[ERROR] Interrupted by user.", file=sys.stderr)
        raise SystemExit(130)
