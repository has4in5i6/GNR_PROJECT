from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from PIL import Image

VALID_OPTIONS = {1, 2, 3, 4, 5}
MODEL_DIR     = Path("models") / "qwen2_5_vl_3b"

# Stitching tunables
SEAM_PX        = 32       # fallback seam strip width when overlap=0
COLOUR_W       = 1.0      # ms-NCC colour weight
GRAD_W         = 2.0      # ms-NCC gradient weight
SSIM_W         = 1.5      # SSIM weight
HIST_W         = 1.0      # gradient orientation histogram weight
MEAN_COLOUR_W  = 2.0      # v8.2: mean-colour distance weight
MAX_SWAP_ITER  = 30
SWAP_THRESH    = 0.00001

# Candidate overlap values to search (pixels)
OVERLAP_CANDIDATES = [4, 8, 12, 16, 20, 24, 32, 48, 64]


def _natural_id(path: Path) -> int:
    m = re.search(r"patch_(\d+)\.png$", path.name)
    if not m:
        raise ValueError(f"Unexpected patch filename: {path.name}")
    return int(m.group(1))


def load_patches(patches_dir: Path) -> dict[int, np.ndarray]:
    paths = sorted(patches_dir.glob("patch_*.png"), key=_natural_id)
    if not paths:
        raise FileNotFoundError(f"No patch_*.png found in {patches_dir}")
    out: dict[int, np.ndarray] = {}
    for p in paths:
        out[_natural_id(p)] = np.asarray(Image.open(p).convert("RGB"), dtype=np.uint8)
    if 0 not in out:
        raise FileNotFoundError("patch_0.png required as top-left anchor")
    return out


def infer_grid(n: int) -> tuple[int, int]:
    r = int(math.isqrt(n))
    if r * r == n:
        return r, r
    best, gap = (1, n), n - 1
    for rows in range(1, r + 1):
        if n % rows == 0:
            cols = n // rows
            if abs(cols - rows) < gap:
                best, gap = (rows, cols), abs(cols - rows)
    return best


# Seam strip extraction
def _seam_strip(img: np.ndarray, side: str, px: int) -> np.ndarray:
    f = img.astype(np.float32) / 255.0
    if side == "top":    return f[:px, :, :]
    if side == "bottom": return f[-px:, :, :]
    if side == "left":   return f[:, :px, :]
    return f[:, -px:, :]

# NCC
def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a -= a.mean(); b -= b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-8:
        return 1.0 - min(2.0, abs(a.mean() - b.mean()) * 2.0)
    return float(np.dot(a, b) / denom)


def _ms_ncc(a: np.ndarray, b: np.ndarray) -> float:
    scores = [_ncc(a, b)]
    for scale in [0.5, 0.25]:
        h = max(1, int(a.shape[0] * scale))
        w = max(1, int(a.shape[1] * scale))
        if a.ndim == 3:
            ar = cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)
            br = cv2.resize(b, (w, h), interpolation=cv2.INTER_AREA)
        else:
            ar = cv2.resize(a.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
            br = cv2.resize(b.astype(np.float32), (w, h), interpolation=cv2.INTER_AREA)
        scores.append(_ncc(ar, br))
    return float(np.mean(scores))


# Sobel gradient magnitude
def _sobel_mag(strip: np.ndarray) -> np.ndarray:
    u8   = (strip * 255).clip(0, 255).astype(np.uint8)
    gray = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy   = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return np.hypot(gx, gy) / 255.0

# SSIM on seam strips
def _ssim_strip(a: np.ndarray, b: np.ndarray) -> float:
    C1, C2 = 0.01**2, 0.03**2
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    mu_a, mu_b = a.mean(), b.mean()
    sig_a  = a.var()
    sig_b  = b.var()
    sig_ab = np.mean((a - mu_a) * (b - mu_b))
    num    = (2*mu_a*mu_b + C1) * (2*sig_ab + C2)
    den    = (mu_a**2 + mu_b**2 + C1) * (sig_a + sig_b + C2)
    return float(num / den) if den > 1e-10 else 0.0

# Gradient orientation histogram (HOG-lite)
def _grad_orient_hist(strip: np.ndarray, bins: int = 8) -> np.ndarray:
    u8   = (strip * 255).clip(0, 255).astype(np.uint8)
    gray = cv2.cvtColor(u8, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx   = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy   = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag  = np.hypot(gx, gy)
    ang  = np.arctan2(gy, gx) % (2 * np.pi)
    hist, _ = np.histogram(ang.ravel(), bins=bins, range=(0, 2*np.pi),
                           weights=mag.ravel())
    return hist / (hist.sum() + 1e-8)


def _bhattacharyya(ha: np.ndarray, hb: np.ndarray) -> float:
    return float(np.sum(np.sqrt(ha * hb)))


# v8.2: Mean-colour distance
def _mean_colour_dist(sa: np.ndarray, sb: np.ndarray) -> float:
    """
    Per-channel L1 distance between mean colours of two seam strips.
    Inputs: float32 [0,1] shape (H,W,3).  Returns value in [0,1].

    NCC subtracts the mean, so it is blind to absolute colour level —
    two water patches always look identical to NCC even if one is
    noticeably darker/greener than its true neighbour.
    This term is NOT normalised, so it correctly penalises such mismatches.
    """
    return float(np.mean(np.abs(sa.mean(axis=(0, 1)) - sb.mean(axis=(0, 1)))))


# Overlap detection
def _strip_ncc_at_overlap(img_a: np.ndarray, img_b: np.ndarray,
                           direction: str, overlap: int) -> float:
    if overlap == 0:
        return 0.0
    if direction == "h":
        sa = _seam_strip(img_a, "right", overlap)
        sb = _seam_strip(img_b, "left",  overlap)
    else:
        sa = _seam_strip(img_a, "bottom", overlap)
        sb = _seam_strip(img_b, "top",    overlap)
    return _ncc(sa, sb)


def estimate_overlap(patches: dict[int, np.ndarray],
                     grid: list[list[Optional["PD"]]],
                     rows: int, cols: int) -> tuple[int, int]:
    patch_h, patch_w = patches[0].shape[:2]
    max_ov = min(patch_w, patch_h) // 3
    candidates = [o for o in OVERLAP_CANDIDATES if o < max_ov]

    def best_overlap_for_pairs(pairs, direction):
        if not pairs:
            return 0
        scores = [(float(np.mean([_strip_ncc_at_overlap(a, b, direction, ov)
                                   for a, b in pairs])), ov)
                  for ov in candidates]
        return max(scores, key=lambda x: x[0])[1]

    h_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for r in range(rows):
        for c in range(cols - 1):
            if grid[r][c] and grid[r][c+1]:
                h_pairs.append((grid[r][c].img, grid[r][c+1].img))
            if len(h_pairs) >= 4: break
        if len(h_pairs) >= 4: break

    v_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for r in range(rows - 1):
        for c in range(cols):
            if grid[r][c] and grid[r+1][c]:
                v_pairs.append((grid[r][c].img, grid[r+1][c].img))
            if len(v_pairs) >= 4: break
        if len(v_pairs) >= 4: break

    return best_overlap_for_pairs(h_pairs, "h"), best_overlap_for_pairs(v_pairs, "v")


# Combined seam cost 
_OVERLAP_X: int = 0
_OVERLAP_Y: int = 0


def _effective_seam_px(direction: str) -> int:
    if direction == "h":
        return _OVERLAP_X if _OVERLAP_X > 0 else SEAM_PX
    return _OVERLAP_Y if _OVERLAP_Y > 0 else SEAM_PX


def seam_cost(img_a: np.ndarray, img_b: np.ndarray, direction: str) -> float:
    spx = _effective_seam_px(direction)

    if direction == "h":
        sa = _seam_strip(img_a, "right",  spx)
        sb = _seam_strip(img_b, "left",   spx)
    else:
        sa = _seam_strip(img_a, "bottom", spx)
        sb = _seam_strip(img_b, "top",    spx)

    ncc_c    = _ms_ncc(sa, sb)
    ncc_g    = _ms_ncc(_sobel_mag(sa), _sobel_mag(sb))
    ssim_c   = _ssim_strip(sa, sb)
    hist_s   = _bhattacharyya(_grad_orient_hist(sa), _grad_orient_hist(sb))
    mean_col = _mean_colour_dist(sa, sb)          # v8.2

    return (
        (1.0 - ncc_c)  * COLOUR_W      +
        (1.0 - ncc_g)  * GRAD_W        +
        (1.0 - ssim_c) * SSIM_W        +
        (1.0 - hist_s) * HIST_W        +
        mean_col       * MEAN_COLOUR_W  # v8.2
    )

# Patch descriptor dataclass
@dataclass
class PD:
    pid: int
    rot: int
    img: np.ndarray


def _make_pd(pid: int, img: np.ndarray, rot: int) -> PD:
    rotated = np.ascontiguousarray(np.rot90(img, rot)) if rot else img
    return PD(pid=pid, rot=rot, img=rotated)


def build_all(patches: dict[int, np.ndarray]) -> dict[tuple[int, int], PD]:
    out: dict[tuple[int, int], PD] = {}
    for pid, img in patches.items():
        for rot in ([0] if pid == 0 else [0, 1, 2, 3]):
            out[(pid, rot)] = _make_pd(pid, img, rot)
    return out



# Cost cache
class CostCache:
    def __init__(self):
        self._h: dict[tuple[int,int,int,int], float] = {}
        self._v: dict[tuple[int,int,int,int], float] = {}

    def h(self, L: PD, R: PD) -> float:
        key = (L.pid, L.rot, R.pid, R.rot)
        if key not in self._h:
            self._h[key] = seam_cost(L.img, R.img, "h")
        return self._h[key]

    def v(self, T: PD, B: PD) -> float:
        key = (T.pid, T.rot, B.pid, B.rot)
        if key not in self._v:
            self._v[key] = seam_cost(T.img, B.img, "v")
        return self._v[key]


_CACHE = CostCache()

def _h(L: PD, R: PD) -> float: return _CACHE.h(L, R)
def _v(T: PD, B: PD) -> float: return _CACHE.v(T, B)


# Rotation selection
def best_rot_ncc(
    pid: int,
    all_pd: dict[tuple[int, int], PD],
    left: Optional[PD], top: Optional[PD],
    right: Optional[PD] = None, bot: Optional[PD] = None,
) -> PD:
    rots = [0] if pid == 0 else [0, 1, 2, 3]

    def cost(d: PD) -> float:
        c, n = 0.0, 0
        if left  is not None: c += _h(left, d);  n += 1
        if right is not None: c += _h(d, right); n += 1
        if top   is not None: c += _v(top,  d);  n += 1
        if bot   is not None: c += _v(d,  bot);  n += 1
        return c / max(n, 1)

    return min((all_pd[(pid, r)] for r in rots if (pid, r) in all_pd), key=cost)

# Grid cost helpers
def _cell_cost(
    grid: list[list[Optional[PD]]], r: int, c: int, rows: int, cols: int,
) -> float:
    cell = grid[r][c]
    if cell is None: return 0.0
    cost = 0.0
    if c > 0      and grid[r][c-1] is not None: cost += _h(grid[r][c-1], cell)
    if c+1 < cols and grid[r][c+1] is not None: cost += _h(cell, grid[r][c+1])
    if r > 0      and grid[r-1][c] is not None: cost += _v(grid[r-1][c], cell)
    if r+1 < rows and grid[r+1][c] is not None: cost += _v(cell, grid[r+1][c])
    return cost


def total_grid_cost(
    grid: list[list[Optional[PD]]], rows: int, cols: int,
) -> float:
    cost = 0.0
    for r in range(rows):
        for c in range(cols):
            cell = grid[r][c]
            if cell is None: continue
            if c+1 < cols and grid[r][c+1] is not None:
                cost += _h(cell, grid[r][c+1])
            if r+1 < rows and grid[r+1][c] is not None:
                cost += _v(cell, grid[r+1][c])
    return cost


def _get_nbrs(
    grid: list[list[Optional[PD]]], r: int, c: int, rows: int, cols: int,
):
    return (
        grid[r][c-1]   if c > 0      else None,
        grid[r][c+1]   if c+1 < cols else None,
        grid[r-1][c]   if r > 0      else None,
        grid[r+1][c]   if r+1 < rows else None,
    )

# Phase 1 — NCC-based greedy placement
def _greedy_place_ncc(
    rows: int, cols: int,
    all_pd: dict[tuple[int, int], PD],
    all_pids: list[int],
    anchor: PD,
) -> list[list[Optional[PD]]]:
    grid: list[list[Optional[PD]]] = [[None] * cols for _ in range(rows)]
    grid[0][0] = anchor
    used = {anchor.pid}

    for r in range(rows):
        for c in range(cols):
            if r == 0 and c == 0:
                continue
            remaining = [p for p in all_pids if p not in used]
            left_d = grid[r][c-1] if c > 0 else None
            top_d  = grid[r-1][c] if r > 0 else None

            best_cost, best_pd = float("inf"), None
            for pid in remaining:
                d = best_rot_ncc(pid, all_pd, left_d, top_d)
                c_val, n = 0.0, 0
                if left_d is not None: c_val += _h(left_d, d); n += 1
                if top_d  is not None: c_val += _v(top_d,  d); n += 1
                c_val /= max(n, 1)
                if c_val < best_cost:
                    best_cost = c_val; best_pd = d

            grid[r][c] = best_pd
            used.add(best_pd.pid)

    return grid

# Phase 1.5 — Multi-pass NCC rotation refinement
def _ncc_rotation_pass(
    grid: list[list[Optional[PD]]],
    rows: int, cols: int,
    all_pd: dict[tuple[int, int], PD],
    n_passes: int = 3,
) -> list[list[Optional[PD]]]:
    for _ in range(n_passes):
        changed = False
        for r in range(rows):
            for c in range(cols):
                if r == 0 and c == 0:
                    continue
                cell = grid[r][c]
                if cell is None:
                    continue
                left, right, top, bot = _get_nbrs(grid, r, c, rows, cols)
                new = best_rot_ncc(cell.pid, all_pd, left, top, right, bot)
                if new.rot != cell.rot:
                    grid[r][c] = new
                    changed = True
        if not changed:
            break
    return grid


# Phase 2 — Swap search  (v8.2: + 2×2 block swap pass)
def _block_cost_2x2(
    grid: list[list[Optional[PD]]], r: int, c: int, rows: int, cols: int,
) -> float:
    """
    Sum of cell costs for a 2×2 block at (r,c), minus the 4 internal seams
    that would otherwise be double-counted.

    External seams (border of block vs rest of grid) are counted once each
    because each cell only counts seams with its own neighbours, and
    internal seams appear in two adjacent cells' costs.
    """
    positions = [(r, c), (r, c+1), (r+1, c), (r+1, c+1)]
    total = sum(_cell_cost(grid, br, bc, rows, cols) for br, bc in positions)

    # Subtract the 4 internal seams (double-counted above)
    # horizontal internals: (r,c)↔(r,c+1) and (r+1,c)↔(r+1,c+1)
    if grid[r][c] and grid[r][c+1]:
        total -= _h(grid[r][c], grid[r][c+1])
    if grid[r+1][c] and grid[r+1][c+1]:
        total -= _h(grid[r+1][c], grid[r+1][c+1])
    # vertical internals: (r,c)↔(r+1,c) and (r,c+1)↔(r+1,c+1)
    if grid[r][c] and grid[r+1][c]:
        total -= _v(grid[r][c], grid[r+1][c])
    if grid[r][c+1] and grid[r+1][c+1]:
        total -= _v(grid[r][c+1], grid[r+1][c+1])

    return total

def _block_cost_3x3(
    grid: list[list[Optional[PD]]], r: int, c: int, rows: int, cols: int,
) -> float:
    """
    Sum of cell costs for a 3×3 block at (r,c), minus the 12 internal
    seams that would otherwise be double-counted.

    Internal seams in a 3×3:
      Horizontal: 3 rows × 2 pairs = 6
      Vertical:   3 cols × 2 pairs = 6
      Total: 12
    """
    positions = [
        (r,   c), (r,   c+1), (r,   c+2),
        (r+1, c), (r+1, c+1), (r+1, c+2),
        (r+2, c), (r+2, c+1), (r+2, c+2),
    ]
    total = sum(_cell_cost(grid, br, bc, rows, cols) for br, bc in positions)

    # Subtract 6 horizontal internal seams
    for row in [r, r+1, r+2]:
        if grid[row][c]   and grid[row][c+1]: total -= _h(grid[row][c],   grid[row][c+1])
        if grid[row][c+1] and grid[row][c+2]: total -= _h(grid[row][c+1], grid[row][c+2])

    # Subtract 6 vertical internal seams
    for col in [c, c+1, c+2]:
        if grid[r][col]   and grid[r+1][col]: total -= _v(grid[r][col],   grid[r+1][col])
        if grid[r+1][col] and grid[r+2][col]: total -= _v(grid[r+1][col], grid[r+2][col])

    return total

def _try_block_swap_2x2(
    grid: list[list[Optional[PD]]],
    r1: int, c1: int,
    r2: int, c2: int,
    rows: int, cols: int,
    all_pd: dict[tuple[int, int], PD],
) -> bool:
    """
    Try swapping the 2×2 block at (r1,c1) with the 2×2 block at (r2,c2).
    Blocks must not overlap.  Anchor (0,0) is never moved.
    Returns True and commits if cost improves, else returns False.

    The internal arrangement of each block is preserved — only the two
    blocks exchange positions.  Each of the 4 patches gets its rotation
    re-optimised for its new location.
    """
    # Collect the 4 patch positions in each block
    pos_A = [(r1,   c1),   (r1,   c1+1),
             (r1+1, c1),   (r1+1, c1+1)]
    pos_B = [(r2,   c2),   (r2,   c2+1),
             (r2+1, c2),   (r2+1, c2+1)]

    # Anchor must not move
    if (0, 0) in pos_A or (0, 0) in pos_B:
        return False

    # Blocks must not overlap
    if set(pos_A) & set(pos_B):
        return False

    pds_A = [grid[r][c] for r, c in pos_A]
    pds_B = [grid[r][c] for r, c in pos_B]

    if any(p is None for p in pds_A + pds_B):
        return False

    before = _block_cost_2x2(grid, r1, c1, rows, cols) + \
             _block_cost_2x2(grid, r2, c2, rows, cols)

    # Subtract double-counted seam between the two blocks if they are adjacent
    # (shared border between block A and block B)
    def _sub_border_seam(rA, cA, rB, cB):
        nonlocal before
        if grid[rA][cA] and grid[rB][cB]:
            if rA == rB and abs(cA - cB) == 1:
                before -= _h(grid[rA][min(cA,cB)], grid[rA][max(cA,cB)])
            elif cA == cB and abs(rA - rB) == 1:
                before -= _v(grid[min(rA,rB)][cA], grid[max(rA,rB)][cA])

    for pa in pos_A:
        for pb in pos_B:
            _sub_border_seam(pa[0], pa[1], pb[0], pb[1])

    # --- Temporarily place block B's patches at block A's positions ---
    # and block A's patches at block B's positions, re-optimising rotation.
    # We blank out all 8 positions first so neighbour lookups are clean.
    saved_A = list(pds_A)
    saved_B = list(pds_B)

    for r, c in pos_A + pos_B:
        grid[r][c] = None

    # Place B's patches at A's positions (preserving 2×2 internal layout)
    new_A: list[PD] = []
    for i, (r, c) in enumerate(pos_A):
        left, _, top, _ = _get_nbrs(grid, r, c, rows, cols)
        new_pd = best_rot_ncc(pds_B[i].pid, all_pd, left, top)
        grid[r][c] = new_pd
        new_A.append(new_pd)

    # Place A's patches at B's positions
    new_B: list[PD] = []
    for i, (r, c) in enumerate(pos_B):
        left, _, top, _ = _get_nbrs(grid, r, c, rows, cols)
        new_pd = best_rot_ncc(pds_A[i].pid, all_pd, left, top)
        grid[r][c] = new_pd
        new_B.append(new_pd)

    after = _block_cost_2x2(grid, r1, c1, rows, cols) + \
            _block_cost_2x2(grid, r2, c2, rows, cols)

    # Subtract the same block-border seams from after
    after_adj = after
    for pa in pos_A:
        for pb in pos_B:
            rA, cA = pa; rB, cB = pb
            if grid[rA][cA] and grid[rB][cB]:
                if rA == rB and abs(cA - cB) == 1:
                    after_adj -= _h(grid[rA][min(cA,cB)], grid[rA][max(cA,cB)])
                elif cA == cB and abs(rA - rB) == 1:
                    after_adj -= _v(grid[min(rA,rB)][cA], grid[max(rA,rB)][cA])

    if after_adj < before - SWAP_THRESH:
        return True   # keep new placement

    # Revert
    for i, (r, c) in enumerate(pos_A):
        grid[r][c] = saved_A[i]
    for i, (r, c) in enumerate(pos_B):
        grid[r][c] = saved_B[i]
    return False

def _try_block_swap_3x3(
    grid: list[list[Optional[PD]]],
    r1: int, c1: int,
    r2: int, c2: int,
    rows: int, cols: int,
    all_pd: dict[tuple[int, int], PD],
) -> bool:
    """
    Try swapping the 3×3 block at (r1,c1) with the 3×3 block at (r2,c2).
    Blocks must not overlap.  Anchor (0,0) is never moved.
    """
    pos_A = [(r1+dr, c1+dc) for dr in range(3) for dc in range(3)]
    pos_B = [(r2+dr, c2+dc) for dr in range(3) for dc in range(3)]

    if (0, 0) in pos_A or (0, 0) in pos_B:
        return False
    if set(pos_A) & set(pos_B):
        return False

    pds_A = [grid[r][c] for r, c in pos_A]
    pds_B = [grid[r][c] for r, c in pos_B]
    if any(p is None for p in pds_A + pds_B):
        return False

    before = (_block_cost_3x3(grid, r1, c1, rows, cols) +
              _block_cost_3x3(grid, r2, c2, rows, cols))

    # Subtract double-counted border seams between the two blocks
    for pa in pos_A:
        for pb in pos_B:
            rA, cA = pa; rB, cB = pb
            if grid[rA][cA] and grid[rB][cB]:
                if rA == rB and abs(cA - cB) == 1:
                    before -= _h(grid[rA][min(cA,cB)], grid[rA][max(cA,cB)])
                elif cA == cB and abs(rA - rB) == 1:
                    before -= _v(grid[min(rA,rB)][cA], grid[max(rA,rB)][cA])

    saved_A = list(pds_A)
    saved_B = list(pds_B)

    for r, c in pos_A + pos_B:
        grid[r][c] = None

    new_A: list[PD] = []
    for i, (r, c) in enumerate(pos_A):
        left, _, top, _ = _get_nbrs(grid, r, c, rows, cols)
        new_pd = best_rot_ncc(pds_B[i].pid, all_pd, left, top)
        grid[r][c] = new_pd
        new_A.append(new_pd)

    new_B: list[PD] = []
    for i, (r, c) in enumerate(pos_B):
        left, _, top, _ = _get_nbrs(grid, r, c, rows, cols)
        new_pd = best_rot_ncc(pds_A[i].pid, all_pd, left, top)
        grid[r][c] = new_pd
        new_B.append(new_pd)

    after = (_block_cost_3x3(grid, r1, c1, rows, cols) +
             _block_cost_3x3(grid, r2, c2, rows, cols))

    # Subtract the same border seams from after
    for pa in pos_A:
        for pb in pos_B:
            rA, cA = pa; rB, cB = pb
            if grid[rA][cA] and grid[rB][cB]:
                if rA == rB and abs(cA - cB) == 1:
                    after -= _h(grid[rA][min(cA,cB)], grid[rA][max(cA,cB)])
                elif cA == cB and abs(rA - rB) == 1:
                    after -= _v(grid[min(rA,rB)][cA], grid[max(rA,rB)][cA])

    if after < before - SWAP_THRESH:
        return True   # keep new placement

    # Revert
    for i, (r, c) in enumerate(pos_A):
        grid[r][c] = saved_A[i]
    for i, (r, c) in enumerate(pos_B):
        grid[r][c] = saved_B[i]
    return False

def _swap_search(
    grid: list[list[Optional[PD]]],
    rows: int, cols: int,
    all_pd: dict[tuple[int, int], PD],
) -> list[list[Optional[PD]]]:
    positions = [(r, c) for r in range(rows) for c in range(cols)
                 if not (r == 0 and c == 0)]

    # All valid 2×2 block top-left corners
    block_origins = [(r, c) for r in range(rows - 1) for c in range(cols - 1)]

    block_origins_3x3 = [(r,c) for r in range(rows-2) for c in range(cols-2)]  # 3×3

    for iteration in range(MAX_SWAP_ITER):
        improved = False

        # ── Rotation-only pass 
        for (r, c) in positions:
            cell = grid[r][c]
            if cell is None: continue
            left, right, top, bot = _get_nbrs(grid, r, c, rows, cols)
            better = best_rot_ncc(cell.pid, all_pd, left, top, right, bot)
            if better.rot != cell.rot:
                grid[r][c] = better
                improved = True

        # ── Pairwise (single-patch) swap pass
        n_pos = len(positions)
        for i in range(n_pos):
            r1, c1 = positions[i]
            for j in range(i + 1, n_pos):
                r2, c2 = positions[j]
                pd1, pd2 = grid[r1][c1], grid[r2][c2]
                if pd1 is None or pd2 is None:
                    continue

                before = (_cell_cost(grid, r1, c1, rows, cols) +
                          _cell_cost(grid, r2, c2, rows, cols))

                adjacent = (abs(r1-r2) + abs(c1-c2) == 1)
                if adjacent:
                    if r1 == r2:
                        before -= _h(grid[r1][min(c1,c2)], grid[r1][max(c1,c2)])
                    else:
                        before -= _v(grid[min(r1,r2)][c1], grid[max(r1,r2)][c1])

                left1, right1, top1, bot1 = _get_nbrs(grid, r1, c1, rows, cols)
                left2, right2, top2, bot2 = _get_nbrs(grid, r2, c2, rows, cols)
                grid[r1][c1] = None; grid[r2][c2] = None

                new1 = best_rot_ncc(pd2.pid, all_pd, left1, top1, right1, bot1)
                new2 = best_rot_ncc(pd1.pid, all_pd, left2, top2, right2, bot2)
                grid[r1][c1] = new1; grid[r2][c2] = new2

                after = (_cell_cost(grid, r1, c1, rows, cols) +
                         _cell_cost(grid, r2, c2, rows, cols))
                if adjacent:
                    if r1 == r2:
                        after -= _h(grid[r1][min(c1,c2)], grid[r1][max(c1,c2)])
                    else:
                        after -= _v(grid[min(r1,r2)][c1], grid[max(r1,r2)][c1])

                if after < before - SWAP_THRESH:
                    improved = True
                else:
                    grid[r1][c1] = pd1; grid[r2][c2] = pd2

        # ── v8.2: 2×2 block swap pass
        # Try every pair of non-overlapping 2×2 blocks.
        n_blk = len(block_origins)
        for i in range(n_blk):
            r1, c1 = block_origins[i]
            for j in range(i + 1, n_blk):
                r2, c2 = block_origins[j]
                # Quick overlap check: blocks share a cell if their ranges overlap
                if abs(r1 - r2) < 2 and abs(c1 - c2) < 2:
                    continue   # blocks overlap, skip
                if _try_block_swap_2x2(grid, r1, c1, r2, c2, rows, cols, all_pd):
                    improved = True

        cost_now = total_grid_cost(grid, rows, cols)
        print(f"[swap] iter={iteration+1}  total_cost={cost_now:.4f}  improved={improved}")
        if not improved:
            break

        # ── 3×3 block swap pass 
        n_blk3 = len(block_origins_3x3)
        for i in range(n_blk3):
            r1, c1 = block_origins_3x3[i]
            for j in range(i + 1, n_blk3):
                r2, c2 = block_origins_3x3[j]
                # Blocks overlap if their 3×3 ranges share any cell
                if abs(r1 - r2) < 3 and abs(c1 - c2) < 3:
                    continue
                if _try_block_swap_3x3(grid, r1, c1, r2, c2, rows, cols, all_pd):
                    improved = True
                    
    return grid

# Phase 3 — Targeted repair 
def _targeted_repair(
    grid: list[list[Optional[PD]]],
    rows: int, cols: int,
    all_pd: dict[tuple[int, int], PD],
    top_k: int = 8,
) -> list[list[Optional[PD]]]:
    seam_costs: list[tuple[float, int, int, str]] = []
    for r in range(rows):
        for c in range(cols):
            if c+1 < cols and grid[r][c] and grid[r][c+1]:
                seam_costs.append((_h(grid[r][c], grid[r][c+1]), r, c, "h"))
            if r+1 < rows and grid[r][c] and grid[r+1][c]:
                seam_costs.append((_v(grid[r][c], grid[r+1][c]), r, c, "v"))

    seam_costs.sort(reverse=True)
    bad_cells: set[tuple[int, int]] = set()
    for _, r, c, direction in seam_costs[:top_k]:
        bad_cells.add((r, c))
        bad_cells.add((r, c+1) if direction == "h" else (r+1, c))
    bad_cells.discard((0, 0))

    if not bad_cells:
        return grid

    bad_list = list(bad_cells)
    print(f"[repair] Targeting {len(bad_list)} cells around top-{top_k} worst seams ...")

    for r1, c1 in bad_list:
        cell = grid[r1][c1]
        if cell is None: continue
        left, right, top, bot = _get_nbrs(grid, r1, c1, rows, cols)
        grid[r1][c1] = best_rot_ncc(cell.pid, all_pd, left, top, right, bot)

    for i in range(len(bad_list)):
        r1, c1 = bad_list[i]
        for j in range(i+1, len(bad_list)):
            r2, c2 = bad_list[j]
            pd1, pd2 = grid[r1][c1], grid[r2][c2]
            if pd1 is None or pd2 is None: continue

            before = (_cell_cost(grid, r1, c1, rows, cols) +
                      _cell_cost(grid, r2, c2, rows, cols))

            left1, right1, top1, bot1 = _get_nbrs(grid, r1, c1, rows, cols)
            left2, right2, top2, bot2 = _get_nbrs(grid, r2, c2, rows, cols)
            grid[r1][c1] = None; grid[r2][c2] = None

            new1 = best_rot_ncc(pd2.pid, all_pd, left1, top1, right1, bot1)
            new2 = best_rot_ncc(pd1.pid, all_pd, left2, top2, right2, bot2)
            grid[r1][c1] = new1; grid[r2][c2] = new2

            after = (_cell_cost(grid, r1, c1, rows, cols) +
                     _cell_cost(grid, r2, c2, rows, cols))
            if after >= before - SWAP_THRESH:
                grid[r1][c1] = pd1; grid[r2][c2] = pd2

    return grid


# Overlap-aware compositing  
def _composite(
    grid: list[list[Optional[PD]]],
    rows: int, cols: int,
    patch_h: int, patch_w: int,
    overlap_x: int, overlap_y: int,
) -> np.ndarray:
    step_x   = patch_w - overlap_x
    step_y   = patch_h - overlap_y
    canvas_w = step_x * cols + overlap_x
    canvas_h = step_y * rows + overlap_y
    canvas   = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for r in range(rows):
        for c in range(cols):
            cell = grid[r][c]
            if cell is None: continue
            img = cell.img
            if img.shape[0] != patch_h or img.shape[1] != patch_w:
                img = cv2.resize(img, (patch_w, patch_h))
            crop_left   = overlap_x if c > 0 else 0
            crop_top    = overlap_y if r > 0 else 0
            img_cropped = img[crop_top:, crop_left:, :]
            x0 = c * step_x
            y0 = r * step_y
            canvas[y0:y0+img_cropped.shape[0], x0:x0+img_cropped.shape[1]] = img_cropped

    return canvas


# Public API
def stitch_patches(patches: dict[int, np.ndarray]) -> np.ndarray:
    global _CACHE, _OVERLAP_X, _OVERLAP_Y

    _OVERLAP_X = 0
    _OVERLAP_Y = 0
    _CACHE = CostCache()

    rows, cols       = infer_grid(len(patches))
    patch_h, patch_w = patches[0].shape[:2]
    print(f"[stitch] grid={rows}x{cols}  patch={patch_w}x{patch_h}")

    all_pd   = build_all(patches)
    all_pids = list(patches.keys())
    anchor   = all_pd[(0, 0)]

    print("[stitch] Phase 1: NCC-based greedy placement ...")
    grid = _greedy_place_ncc(rows, cols, all_pd, all_pids, anchor)
    print(f"[stitch] after greedy: cost={total_grid_cost(grid, rows, cols):.4f}")

    print("[stitch] Phase 1.5: Multi-pass NCC rotation refinement ...")
    grid = _ncc_rotation_pass(grid, rows, cols, all_pd, n_passes=3)
    print(f"[stitch] after rotation pass: cost={total_grid_cost(grid, rows, cols):.4f}")

    print("[stitch] Phase 2: Swap optimisation (pairwise + 2×2 block) ...")
    grid = _swap_search(grid, rows, cols, all_pd)
    print(f"[stitch] after swap: cost={total_grid_cost(grid, rows, cols):.4f}")

    print("[stitch] Phase 3: Targeted repair of worst seams ...")
    for repair_round in range(3):
        cost_before = total_grid_cost(grid, rows, cols)
        grid = _targeted_repair(grid, rows, cols, all_pd, top_k=8)
        cost_after = total_grid_cost(grid, rows, cols)
        print(f"[repair] round={repair_round+1}  {cost_before:.4f} -> {cost_after:.4f}")
        if cost_after >= cost_before - SWAP_THRESH:
            break
        grid = _swap_search(grid, rows, cols, all_pd)

    print(f"[stitch] final cost={total_grid_cost(grid, rows, cols):.4f}")

    print("[stitch] Detecting patch overlap ...")
    ov_x, ov_y = estimate_overlap(patches, grid, rows, cols)
    print(f"[stitch] overlap detected: horizontal={ov_x}px  vertical={ov_y}px")

    _OVERLAP_X = ov_x
    _OVERLAP_Y = ov_y

    canvas = _composite(grid, rows, cols, patch_h, patch_w, ov_x, ov_y)
    return canvas


# VLM
class LocalVLM:
    def __init__(self, model_dir: Path) -> None:
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.process_vision_info = process_vision_info
        self.processor = AutoProcessor.from_pretrained(
            str(model_dir), local_files_only=True,
            min_pixels=256*28*28, max_pixels=1280*28*28,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(model_dir), local_files_only=True,
            torch_dtype=torch.float16, device_map="auto",
        )
        self.model.eval()

    def answer(self, stitched_map_path: Path, row: pd.Series) -> int:
        options  = [str(row[f"option_{i}"]) for i in range(1, 5)]
        opts_txt = "\n".join(f"{i+1}. {o}" for i, o in enumerate(options))
        prompt   = (
            "You are answering a multiple-choice question about a stitched geospatial map. "
            "Use only the image and the options. If the answer is not clearly supported, choose 5.\n\n"
            f"Question: {row['question']}\n\nOptions:\n{opts_txt}\n\n"
            "Return exactly one digit: 1, 2, 3, 4, or 5."
        )
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": stitched_map_path.resolve().as_uri()},
            {"type": "text",  "text": prompt},
        ]}]
        text = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        img_in, vid_in = self.process_vision_info(msgs)
        inputs = self.processor(
            text=[text], images=img_in, videos=vid_in,
            padding=True, return_tensors="pt",
        ).to(self.model.device)
        with self.torch.inference_mode():
            gen = self.model.generate(**inputs, max_new_tokens=4, do_sample=False)
        gen = [o[len(i):] for i, o in zip(inputs.input_ids, gen)]
        decoded = self.processor.batch_decode(
            gen, skip_special_tokens=True, clean_up_tokenization_spaces=False,
        )[0].strip()
        m = re.search(r"[1-5]", decoded)
        if not m:
            return 5
        ans = int(m.group(0))
        return ans if ans in VALID_OPTIONS else 5


# Submission + main
def build_submission(test_csv: Path, stitched_map_path: Path) -> pd.DataFrame:
    qs = pd.read_csv(test_csv)
    id_col  = "id" if "id" in qs.columns else "question_id"
    missing = {id_col, "question", "option_1", "option_2", "option_3", "option_4"} - set(qs.columns)
    if missing:
        raise ValueError(f"test.csv missing columns: {sorted(missing)}")

    vlm: Optional[LocalVLM] = None
    if MODEL_DIR.exists():
        try:
            vlm = LocalVLM(MODEL_DIR)
        except Exception as e:
            print(f"[warn] VLM load failed, using option 5: {e}")

    rows = []
    for _, row in qs.iterrows():
        qid    = str(row[id_col])
        answer = vlm.answer(stitched_map_path, row) if vlm else 5
        if answer not in VALID_OPTIONS:
            answer = 5
        rows.append({"id": qid, "question_num": qid, "option": answer})
    return pd.DataFrame(rows, columns=["id", "question_num", "option"])


def run(test_dir: Path) -> None:
    test_dir = test_dir.resolve()
    patches  = load_patches(test_dir / "patches")
    stitched = stitch_patches(patches)
    out = Path("stitched_map.jpg")
    cv2.imwrite(str(out), cv2.cvtColor(stitched, cv2.COLOR_RGB2BGR))
    print(f"[stitch] saved -> {out}")
    sub = build_submission(test_dir / "test.csv", out)
    sub.to_csv("submission.csv", index=False)
    print("[done] submission.csv written")


def main() -> None:
    p = argparse.ArgumentParser(description="Offline inference for Project 1")
    p.add_argument("--test_dir", required=True)
    run(Path(p.parse_args().test_dir))


if __name__ == "__main__":
    main()