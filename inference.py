import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

# ---------------- CONFIG ---------------- #
BORDER = 16
BEAM_WIDTH = 40
VALID_OPTIONS = {1, 2, 3, 4, 5}

# ---------------- UTILS ---------------- #
def natural_patch_id(path: Path) -> int:
    return int(re.search(r"patch_(\d+)\.png$", path.name).group(1))


def load_patches(patches_dir: Path):
    paths = sorted(patches_dir.glob("patch_*.png"), key=natural_patch_id)
    patches = {}
    for p in paths:
        patches[natural_patch_id(p)] = np.asarray(Image.open(p).convert("RGB"))
    return patches


def infer_grid(n):
    r = int(math.isqrt(n))
    return (r, r) if r*r == n else (1, n)


# ---------------- FEATURES ---------------- #
def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


def sobel(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
    return np.sqrt(gx**2 + gy**2)


def norm(x):
    return (x - x.mean()) / (x.std() + 1e-6)


def edge_cost(a, b):
    a, b = norm(a), norm(b)
    return np.mean((a - b) ** 2)


# ---------------- CANDIDATES ---------------- #
@dataclass
class Candidate:
    pid: int
    rot: int
    img: np.ndarray
    top: np.ndarray
    bottom: np.ndarray
    left: np.ndarray
    right: np.ndarray


def build_candidates(patches):
    out = []
    for pid, img in patches.items():
        for r in range(4):
            rot = np.rot90(img, r)
            g = to_gray(rot)
            s = sobel(g)

            # combine pixel + gradient
            f = 0.7*g + 0.3*s

            out.append(Candidate(
                pid, r, rot,
                f[:BORDER, :],
                f[-BORDER:, :],
                f[:, :BORDER],
                f[:, -BORDER:]
            ))
    return out


# ---------------- BEAM SEARCH ---------------- #
@dataclass
class State:
    grid: list
    used: set
    score: float


def place_cost(state, cand, r, c, rows, cols):
    cost = 0
    grid = state.grid

    # left
    if c > 0 and grid[r][c-1] is not None:
        left = grid[r][c-1]
        cost += edge_cost(left.right, cand.left)

    # top
    if r > 0 and grid[r-1][c] is not None:
        top = grid[r-1][c]
        cost += edge_cost(top.bottom, cand.top)

    return cost


def beam_search(patches):
    rows, cols = infer_grid(len(patches))
    candidates = build_candidates(patches)

    # group by patch id
    by_id = {}
    for c in candidates:
        by_id.setdefault(c.pid, []).append(c)

    # start with patch_0 fixed
    start = [[None]*cols for _ in range(rows)]
    start[0][0] = by_id[0][0]

    states = [State(start, {0}, 0.0)]

    for pos in range(1, rows*cols):
        r, c = divmod(pos, cols)
        new_states = []

        for state in states:
            for pid, cand_list in by_id.items():
                if pid in state.used:
                    continue

                for cand in cand_list:
                    cost = place_cost(state, cand, r, c, rows, cols)

                    new_grid = [row[:] for row in state.grid]
                    new_grid[r][c] = cand

                    new_states.append(State(
                        new_grid,
                        state.used | {pid},
                        state.score + cost
                    ))

        # prune
        new_states.sort(key=lambda s: s.score)
        states = new_states[:BEAM_WIDTH]

    return states[0].grid


# ---------------- STITCH ---------------- #
def stitch(grid):
    rows, cols = len(grid), len(grid[0])
    h, w = grid[0][0].img.shape[:2]

    canvas = np.zeros((rows*h, cols*w, 3), dtype=np.uint8)

    for r in range(rows):
        for c in range(cols):
            canvas[r*h:(r+1)*h, c*w:(c+1)*w] = grid[r][c].img

    return canvas


# ---------------- VLM (UNCHANGED) ---------------- #
class DummyVLM:
    def answer(self, *_):
        return 5


def build_submission(test_csv, stitched_path):
    df = pd.read_csv(test_csv)
    id_col = "id" if "id" in df.columns else "question_id"

    vlm = DummyVLM()

    rows = []
    for _, row in df.iterrows():
        rows.append({
            "id": str(row[id_col]),
            "question_num": str(row[id_col]),
            "option": vlm.answer()
        })

    return pd.DataFrame(rows)


# ---------------- MAIN ---------------- #
def run(test_dir: Path):
    patches = load_patches(test_dir / "patches")

    grid = beam_search(patches)
    stitched = stitch(grid)

    out_path = Path("stitched_map.jpg")
    cv2.imwrite(str(out_path), cv2.cvtColor(stitched, cv2.COLOR_RGB2BGR))

    submission = build_submission(test_dir / "test.csv", out_path)
    submission.to_csv("submission.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", required=True)
    args = parser.parse_args()

    run(Path(args.test_dir))
