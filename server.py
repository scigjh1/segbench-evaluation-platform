"""SegBench mask evaluation and regression server."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import statistics
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np
from runtime_bridge import runtime_data, compare_runs


ROOT = Path(__file__).resolve().parent
STATIC_DIR = ROOT / "static"
MAX_BODY_BYTES = 48 * 1024 * 1024


def read_mask(path: Path) -> np.ndarray:
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"无法读取掩膜：{path.name}")
    return np.where(mask >= 128, 255, 0).astype(np.uint8)


def decode_mask(data_url: str) -> np.ndarray:
    if not data_url or "," not in data_url:
        raise ValueError("掩膜数据无效")
    raw = base64.b64decode(data_url.split(",", 1)[1], validate=True)
    mask = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError("无法解析掩膜")
    return np.where(mask >= 128, 255, 0).astype(np.uint8)


def boundary(mask: np.ndarray) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)


def boundary_metrics(prediction: np.ndarray, target: np.ndarray, tolerance: int) -> tuple[float, float]:
    pred_boundary = boundary(prediction)
    target_boundary = boundary(target)
    kernel_size = tolerance * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    pred_dilated = cv2.dilate(pred_boundary, kernel)
    target_dilated = cv2.dilate(target_boundary, kernel)

    pred_count = cv2.countNonZero(pred_boundary)
    target_count = cv2.countNonZero(target_boundary)
    precision = cv2.countNonZero(cv2.bitwise_and(pred_boundary, target_dilated)) / pred_count if pred_count else 1.0
    recall = cv2.countNonZero(cv2.bitwise_and(target_boundary, pred_dilated)) / target_count if target_count else 1.0
    boundary_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    pred_binary = pred_boundary > 0
    target_binary = target_boundary > 0
    if not pred_binary.any() or not target_binary.any():
        return boundary_f1, 0.0 if np.array_equal(prediction, target) else float(max(target.shape))
    distance_to_target = cv2.distanceTransform((~target_binary).astype(np.uint8), cv2.DIST_L2, 5)
    distance_to_pred = cv2.distanceTransform((~pred_binary).astype(np.uint8), cv2.DIST_L2, 5)
    distances = np.concatenate([distance_to_target[pred_binary], distance_to_pred[target_binary]])
    hd95 = float(np.percentile(distances, 95)) if distances.size else 0.0
    return boundary_f1, hd95


def evaluate_pair(prediction: np.ndarray, target: np.ndarray, tolerance: int) -> dict[str, float]:
    if prediction.shape != target.shape:
        prediction = cv2.resize(prediction, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_NEAREST)
    pred = prediction > 0
    truth = target > 0
    tp = int(np.logical_and(pred, truth).sum())
    fp = int(np.logical_and(pred, ~truth).sum())
    fn = int(np.logical_and(~pred, truth).sum())
    intersection = tp
    union = int(np.logical_or(pred, truth).sum())
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 1.0
    iou = intersection / union if union else 1.0
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    boundary_f1, hd95 = boundary_metrics(prediction, target, tolerance)
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "boundaryF1": boundary_f1,
        "hd95": hd95,
    }


def summarize(rows: list[dict], timings_ms: list[float]) -> dict:
    metric_names = ["dice", "iou", "precision", "recall", "boundaryF1", "hd95"]
    summary = {
        metric: round(statistics.fmean(row[metric] for row in rows), 4)
        for metric in metric_names
    }
    ordered = sorted(timings_ms)
    p95_index = min(len(ordered) - 1, int(np.ceil(len(ordered) * 0.95)) - 1)
    summary.update(
        {
            "samples": len(rows),
            "meanLatencyMs": round(statistics.fmean(timings_ms), 3),
            "p95LatencyMs": round(ordered[p95_index], 3),
            "throughputFps": round(1000 / statistics.fmean(timings_ms), 1),
        }
    )
    return summary


def evaluate_dataset(pairs: list[tuple[str, np.ndarray, np.ndarray]], tolerance: int) -> dict:
    rows = []
    timings = []
    for name, prediction, target in pairs:
        started = time.perf_counter()
        metrics = evaluate_pair(prediction, target, tolerance)
        timings.append((time.perf_counter() - started) * 1000)
        rows.append({"name": name, **{key: round(value, 4) for key, value in metrics.items()}})
    return {"summary": summarize(rows, timings), "rows": rows}


def sample_pairs(group: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
    gt_dir = STATIC_DIR / "sample" / "gt"
    prediction_dir = STATIC_DIR / "sample" / group
    pairs = []
    for target_path in sorted(gt_dir.glob("*.png")):
        prediction_path = prediction_dir / target_path.name
        if prediction_path.is_file():
            pairs.append((target_path.stem, read_mask(prediction_path), read_mask(target_path)))
    if not pairs:
        raise ValueError("内置样例集尚未生成")
    return pairs


def sample_evaluation(tolerance: int) -> dict:
    baseline = evaluate_dataset(sample_pairs("baseline"), tolerance)
    candidate = evaluate_dataset(sample_pairs("candidate"), tolerance)
    delta = {
        metric: round(candidate["summary"][metric] - baseline["summary"][metric], 4)
        for metric in ["dice", "iou", "precision", "recall", "boundaryF1"]
    }
    delta["hd95"] = round(baseline["summary"]["hd95"] - candidate["summary"]["hd95"], 4)
    return {"baseline": baseline, "candidate": candidate, "delta": delta, "tolerance": tolerance}


def uploaded_evaluation(payload: dict) -> dict:
    tolerance = max(1, min(int(payload.get("tolerance", 2)), 10))
    pairs = []
    for item in payload.get("pairs", []):
        pairs.append((item["name"], decode_mask(item["prediction"]), decode_mask(item["target"])))
    if not pairs:
        raise ValueError("没有可评测的掩膜对")
    return evaluate_dataset(pairs, tolerance)


class SegBenchHandler(BaseHTTPRequestHandler):
    server_version = "SegBench/1.0"

    def send_json(self, data: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path.startswith('/api/runtime/'):
            try:
                self.send_json({'ok':True,'result':runtime_data(path.rsplit('/',1)[-1])})
            except Exception:
                self.send_json({'ok':False,'error':'Runtime unavailable; check SEG_SCOPE_RUNTIME_URL'},HTTPStatus.SERVICE_UNAVAILABLE)
            return
        if path == "/api/health":
            self.send_json({"status": "ok", "engine": f"OpenCV {cv2.__version__}"})
            return
        if path == "/api/sample-evaluation":
            tolerance = max(1, min(int(parse_qs(parsed.query).get("tolerance", ["2"])[0]), 10))
            try:
                self.send_json({"ok": True, "result": sample_evaluation(tolerance)})
            except ValueError as exc:
                self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return
        if path == "/README.md":
            content = (ROOT / "README.md").read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/markdown; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return
        if path == "/":
            path = "/gpu.html"
        requested = (STATIC_DIR / path.lstrip("/")).resolve()
        if STATIC_DIR.resolve() not in requested.parents or not requested.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content = requested.read_bytes()
        content_type, _ = mimetypes.guess_type(requested.name)
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def do_POST(self) -> None:  # noqa: N802
        if urlparse(self.path).path == '/api/regression':
            try:
                length=int(self.headers.get('Content-Length','0'))
                if not 0<length<4096:raise ValueError('Invalid regression request')
                payload=json.loads(self.rfile.read(length))
                runs={r['id']:r for r in runtime_data('runs')}
                result=compare_runs(runs[payload['baseline']],runs[payload['candidate']])
                self.send_json({'ok':True,'result':result})
            except (ValueError,KeyError) as exc:
                self.send_json({'ok':False,'error':str(exc)},HTTPStatus.BAD_REQUEST)
            except Exception:
                self.send_json({'ok':False,'error':'Runtime unavailable'},HTTPStatus.SERVICE_UNAVAILABLE)
            return
        if urlparse(self.path).path != "/api/evaluate":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > MAX_BODY_BYTES:
                raise ValueError("请求体为空或超过 48 MB")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            self.send_json({"ok": True, "result": uploaded_evaluation(payload)})
        except (ValueError, KeyError, json.JSONDecodeError, base64.binascii.Error) as exc:
            self.send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # pragma: no cover
            self.send_json({"ok": False, "error": f"评测失败：{exc}"}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def log_message(self, format: str, *args) -> None:
        print(f"[{self.log_date_time_string()}] {format % args}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4174)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), SegBenchHandler)
    print(f"SegBench running at http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
