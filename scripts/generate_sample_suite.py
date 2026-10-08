"""Generate a deterministic binary-mask regression suite."""

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1] / "static" / "sample"
SIZE = 512


def shift(mask: np.ndarray, dx: int, dy: int) -> np.ndarray:
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(mask, matrix, (SIZE, SIZE), flags=cv2.INTER_NEAREST, borderValue=0)


def perturb(mask: np.ndarray, rng: np.random.Generator, level: str) -> np.ndarray:
    if level == "baseline":
        moved = shift(mask, int(rng.integers(-13, 14)), int(rng.integers(-13, 14)))
        kernel_size = int(rng.choice([7, 9, 11]))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        moved = cv2.erode(moved, kernel, iterations=1) if rng.random() < 0.5 else cv2.dilate(moved, kernel, iterations=1)
        for _ in range(3):
            cx, cy = int(rng.integers(40, 472)), int(rng.integers(40, 472))
            cv2.circle(moved, (cx, cy), int(rng.integers(4, 13)), 255 if rng.random() < 0.45 else 0, -1)
        return moved
    moved = shift(mask, int(rng.integers(-4, 5)), int(rng.integers(-4, 5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(moved, cv2.MORPH_CLOSE, kernel, iterations=1)


def main() -> None:
    for group in ("gt", "baseline", "candidate"):
        (ROOT / group).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260923)
    for index in range(24):
        mask = np.zeros((SIZE, SIZE), np.uint8)
        if index % 3 == 0:
            center = (int(rng.integers(150, 362)), int(rng.integers(150, 362)))
            axes = (int(rng.integers(60, 130)), int(rng.integers(45, 110)))
            cv2.ellipse(mask, center, axes, int(rng.integers(0, 180)), 0, 360, 255, -1, cv2.LINE_AA)
        elif index % 3 == 1:
            x1, y1 = int(rng.integers(60, 190)), int(rng.integers(60, 190))
            x2, y2 = int(rng.integers(320, 455)), int(rng.integers(320, 455))
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            cv2.circle(mask, (x2, y1), int(rng.integers(28, 58)), 255, -1, cv2.LINE_AA)
        else:
            points = rng.integers(70, 442, size=(7, 2)).astype(np.int32)
            hull = cv2.convexHull(points)
            cv2.fillConvexPoly(mask, hull, 255, cv2.LINE_AA)

        name = f"sample_{index + 1:02d}.png"
        cv2.imwrite(str(ROOT / "gt" / name), mask)
        cv2.imwrite(str(ROOT / "baseline" / name), perturb(mask, rng, "baseline"))
        cv2.imwrite(str(ROOT / "candidate" / name), perturb(mask, rng, "candidate"))


if __name__ == "__main__":
    main()
