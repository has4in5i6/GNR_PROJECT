import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image


VALID_OPTIONS = {1, 2, 3, 4, 5}
MODEL_DIR = Path("models") / "qwen2_5_vl_3b"
BORDER_MATCH_WIDTH = 16
MAX_CANDIDATE_COUNT = 32


def natural_patch_id(path: Path) -> int:
    match = re.search(r"patch_(\d+)\.png$", path.name)
    if not match:
        raise ValueError(f"Unexpected patch filename: {path.name}")
    return int(match.group(1))


@dataclass
class PatchCandidate:
    patch_id: int
    rotation: int
    image: np.ndarray
    gray: np.ndarray
    top_border: np.ndarray
    bottom_border: np.ndarray
    left_border: np.ndarray
    right_border: np.ndarray


def load_patches(patches_dir: Path) -> dict[int, np.ndarray]:
    patch_paths = sorted(patches_dir.glob("patch_*.png"), key=natural_patch_id)
    if not patch_paths:
        raise FileNotFoundError(f"No patch_*.png files found in {patches_dir}")

    patches = {}
    for path in patch_paths:
        idx = natural_patch_id(path)
        image = Image.open(path).convert("RGB")
        patches[idx] = np.asarray(image, dtype=np.uint8)

    if 0 not in patches:
        raise FileNotFoundError("patch_0.png is required as the top-left anchor")
    return patches


def infer_grid_shape(num_patches: int) -> tuple[int, int]:
    root = int(math.isqrt(num_patches))
    if root * root == num_patches:
        return root, root

    best_rows, best_cols = 1, num_patches
    best_gap = num_patches - 1
    for rows in range(1, root + 1):
        if num_patches % rows == 0:
            cols = num_patches // rows
            gap = abs(cols - rows)
            if gap < best_gap:
                best_rows, best_cols, best_gap = rows, cols, gap
    return best_rows, best_cols


def rotate_patch(image: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 0:
        return image
    return np.ascontiguousarray(np.rot90(image, rotation))


def gray_float(image: np.ndarray, size: int = 32) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    if gray.shape[0] != size or gray.shape[1] != size:
        gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.GaussianBlur(gray, (3, 3), 0)


def normalized_patch_cost(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("inf")
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a = (a - float(a.mean())) / (float(a.std()) + 1e-6)
    b = (b - float(b.mean())) / (float(b.std()) + 1e-6)
    return float(np.mean((a - b) ** 2))


def overlap_range(length: int) -> range:
    lo = max(6, int(length * 0.20))
    hi = min(length - 3, int(length * 0.80))
    return range(lo, hi + 1, 2)


def border_features(gray: np.ndarray, border_width: int = BORDER_MATCH_WIDTH) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        gray[:border_width, :],
        gray[-border_width:, :],
        gray[:, :border_width],
        gray[:, -border_width:],
    )


def border_match_cost(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        a = cv2.resize(a, (b.shape[1], b.shape[0]), interpolation=cv2.INTER_AREA)
    return normalized_patch_cost(a, b)


def prune_candidates(
    candidates: list[PatchCandidate],
    top_neighbor_gray: np.ndarray | None,
    left_neighbor_gray: np.ndarray | None,
) -> list[PatchCandidate]:
    if top_neighbor_gray is None and left_neighbor_gray is None:
        return candidates[:MAX_CANDIDATE_COUNT]

    scored_candidates: list[tuple[float, PatchCandidate]] = []
    for candidate in candidates:
        score = 0.0
        count = 0
        if left_neighbor_gray is not None:
            score += border_match_cost(left_neighbor_gray[:, -BORDER_MATCH_WIDTH :], candidate.left_border)
            count += 1
        if top_neighbor_gray is not None:
            score += border_match_cost(top_neighbor_gray[-BORDER_MATCH_WIDTH :, :], candidate.top_border)
            count += 1
        scored_candidates.append((score / max(count, 1), candidate))

    scored_candidates.sort(key=lambda item: item[0])
    return [candidate for _, candidate in scored_candidates[:MAX_CANDIDATE_COUNT]]


def horizontal_overlap_cost(left: np.ndarray, right: np.ndarray, original_width: int) -> tuple[float, int]:
    best_score = float("inf")
    best_overlap = left.shape[1] // 2
    for overlap in overlap_range(left.shape[1]):
        score = normalized_patch_cost(left[:, -overlap:], right[:, :overlap])
        if score < best_score:
            best_score = score
            best_overlap = overlap
    scaled_overlap = int(round(best_overlap * original_width / left.shape[1]))
    return best_score, scaled_overlap


def vertical_overlap_cost(top: np.ndarray, bottom: np.ndarray, original_height: int) -> tuple[float, int]:
    best_score = float("inf")
    best_overlap = top.shape[0] // 2
    for overlap in overlap_range(top.shape[0]):
        score = normalized_patch_cost(top[-overlap:, :], bottom[:overlap, :])
        if score < best_score:
            best_score = score
            best_overlap = overlap
    scaled_overlap = int(round(best_overlap * original_height / top.shape[0]))
    return best_score, scaled_overlap


def choose_next_patch(
    candidates: list[PatchCandidate],
    top_neighbor_gray: np.ndarray | None,
    left_neighbor_gray: np.ndarray | None,
    patch_h: int,
    patch_w: int,
) -> tuple[int, int, np.ndarray, int | None, int | None, float]:
    best = None
    best_score = float("inf")

    for candidate in prune_candidates(candidates, top_neighbor_gray, left_neighbor_gray):
        score = 0.0
        count = 0
        x_overlap = None
        y_overlap = None

        if left_neighbor_gray is not None:
            h_score, x_overlap = horizontal_overlap_cost(left_neighbor_gray, candidate.gray, patch_w)
            score += h_score
            count += 1
        if top_neighbor_gray is not None:
            v_score, y_overlap = vertical_overlap_cost(top_neighbor_gray, candidate.gray, patch_h)
            score += v_score
            count += 1

        if count and score / count < best_score:
            best_score = score / count
            best = (
                candidate.patch_id,
                candidate.rotation,
                candidate.image,
                x_overlap,
                y_overlap,
                best_score,
            )

    if best is None:
        raise RuntimeError("Could not choose next patch")
    return best


# def stitch_patches(patches: dict[int, np.ndarray]) -> np.ndarray:
#     rows, cols = infer_grid_shape(len(patches))
#     patch_h, patch_w = patches[0].shape[:2]

#     rotated_candidates: list[PatchCandidate] = []
#     for patch_id, image in patches.items():
#         if patch_id == 0:
#             continue
#         for rotation in range(4):
#             rotated = rotate_patch(image, rotation)
#             gray = gray_float(rotated)
#             top_border, bottom_border, left_border, right_border = border_features(gray)
#             rotated_candidates.append(
#                 PatchCandidate(
#                     patch_id=patch_id,
#                     rotation=rotation,
#                     image=rotated,
#                     gray=gray,
#                     top_border=top_border,
#                     bottom_border=bottom_border,
#                     left_border=left_border,
#                     right_border=right_border,
#                 )
#             )

#     grid: list[list[np.ndarray | None]] = [[None for _ in range(cols)] for _ in range(rows)]
#     grid_gray: list[list[np.ndarray | None]] = [[None for _ in range(cols)] for _ in range(rows)]
#     coords: list[list[tuple[int, int] | None]] = [[None for _ in range(cols)] for _ in range(rows)]
#     grid[0][0] = patches[0]
#     grid_gray[0][0] = gray_float(patches[0])
#     coords[0][0] = (0, 0)
#     used = {0}

#     while len(used) < len(patches):
#         candidates = [item for item in rotated_candidates if item.patch_id not in used]
#         best_cell = None
#         best_choice = None
#         best_score = float("inf")

#         for row in range(rows):
#             for col in range(cols):
#                 if grid[row][col] is not None:
#                     continue
#                 top_neighbor_gray = grid_gray[row - 1][col] if row > 0 else None
#                 left_neighbor_gray = grid_gray[row][col - 1] if col > 0 else None
#                 if top_neighbor_gray is None and left_neighbor_gray is None:
#                     continue

#                 choice = choose_next_patch(
#                     candidates, top_neighbor_gray, left_neighbor_gray, patch_h, patch_w
#                 )
#                 if choice[-1] < best_score:
#                     best_score = choice[-1]
#                     best_choice = choice
#                     best_cell = (row, col)

#         if best_cell is None or best_choice is None:
#             raise RuntimeError("Could not expand the stitch graph")

#         row, col = best_cell
#         patch_id, _, selected, x_overlap, y_overlap, _ = best_choice
#         top_neighbor = grid[row - 1][col] if row > 0 else None
#         left_neighbor = grid[row][col - 1] if col > 0 else None

#         grid[row][col] = selected
#         grid_gray[row][col] = gray_float(selected)

#         x = col * patch_w
#         y = row * patch_h
#         if left_neighbor is not None and x_overlap is not None:
#             left_coord = coords[row][col - 1]
#             if left_coord is not None:
#                 left_x, left_y = left_coord
#                 x = left_x + patch_w - x_overlap
#                 y = left_y
#         if top_neighbor is not None and y_overlap is not None:
#             top_coord = coords[row - 1][col]
#             if top_coord is not None:
#                 top_x, top_y = top_coord
#                 y = top_y + patch_h - y_overlap
#                 if left_neighbor is None:
#                     x = top_x
#                 else:
#                     x = int(round((x + top_x) / 2))
#         coords[row][col] = (x, y)
#         used.add(patch_id)

#     placed = []
#     for row in range(rows):
#         for col in range(cols):
#             image = grid[row][col]
#             coord = coords[row][col]
#             if image is None or coord is None:
#                 continue
#             placed.append((coord[0], coord[1], image))

#     min_x = min(x for x, _, _ in placed)
#     min_y = min(y for _, y, _ in placed)
#     max_x = max(x + patch_w for x, _, _ in placed)
#     max_y = max(y + patch_h for _, y, _ in placed)

#     acc = np.zeros((max_y - min_y, max_x - min_x, 3), dtype=np.float32)
#     weight = np.zeros((max_y - min_y, max_x - min_x, 1), dtype=np.float32)
#     for x, y, image in placed:
#         x0 = x - min_x
#         y0 = y - min_y
#         acc[y0 : y0 + patch_h, x0 : x0 + patch_w] += image.astype(np.float32)
#         weight[y0 : y0 + patch_h, x0 : x0 + patch_w] += 1.0

#     weight = np.maximum(weight, 1.0)
#     return np.clip(acc / weight, 0, 255).astype(np.uint8)

def stitch_patches(patches: dict[int, np.ndarray]) -> np.ndarray:
    """
    Uses OpenCV's high-level Stitcher class with SIFT features.
    This handles rotations, slight overlaps, and exposure blending automatically.
    """
    import cv2
    
    # 1. Prepare images in order
    # Note: We include patch_0 as it's the anchor
    imgs = [img for idx, img in sorted(patches.items())]
    
    # 2. Create the Stitcher
    # Mode 1 (SCANS) is optimized for flat surfaces like maps/orthophotos
    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    
    # 3. Perform Stitching
    # This detects features (SIFT), matches them, and blends the edges
    status, stitched = stitcher.stitch(imgs)

    if status != cv2.Stitcher_OK:
        print(f"[warn] Advanced stitching failed with status code {status}.")
        print("Falling back to basic tiled alignment...")
        return fallback_tile_stitch(patches)
        
    # 4. Post-processing: Remove any thin black borders created by the warping
    gray = cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(contours[0])
        stitched = stitched[y:y+h, x:x+w]

    return stitched

def fallback_tile_stitch(patches: dict[int, np.ndarray]) -> np.ndarray:
    """Simple fallback if feature matching fails: just sticks them in a grid."""
    rows, cols = infer_grid_shape(len(patches))
    patch_h, patch_w = patches[0].shape[:2]
    canvas = np.zeros((rows * patch_h, cols * patch_w, 3), dtype=np.uint8)
    
    for idx, img in patches.items():
        r, c = divmod(idx, cols)
        canvas[r*patch_h:(r+1)*patch_h, c*patch_w:(c+1)*patch_w] = img
    return canvas

class LocalVLM:
    def __init__(self, model_dir: Path) -> None:
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        self.process_vision_info = process_vision_info
        self.processor = AutoProcessor.from_pretrained(
            str(model_dir),
            local_files_only=True,
            min_pixels=256 * 28 * 28,
            max_pixels=1280 * 28 * 28,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            str(model_dir),
            local_files_only=True,
            torch_dtype=torch.float16,  # Optimized for T4 GPUs
            device_map="auto",
        )
        self.model.eval()

    def answer(self, stitched_map_path: Path, row: pd.Series) -> int:
        options = [
            str(row["option_1"]),
            str(row["option_2"]),
            str(row["option_3"]),
            str(row["option_4"]),
        ]
        options_text = "\n".join(f"{idx + 1}. {option}" for idx, option in enumerate(options))
        prompt = (
            "You are answering a multiple-choice question about a stitched geospatial map. "
            "Use only the image and the options. If the answer is not clearly supported, choose 5.\n\n"
            f"Question: {row['question']}\n\n"
            f"Options:\n{options_text}\n\n"
            "Return exactly one digit: 1, 2, 3, 4, or 5."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": stitched_map_path.resolve().as_uri()},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)

        with self.torch.inference_mode():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=4,
                do_sample=False,
            )
        generated = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated)
        ]
        decoded = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        match = re.search(r"[1-5]", decoded)
        if not match:
            return 5
        answer = int(match.group(0))
        return answer if answer in VALID_OPTIONS else 5


def build_submission(test_csv: Path, stitched_map_path: Path) -> pd.DataFrame:
    questions = pd.read_csv(test_csv)
    id_column = "id" if "id" in questions.columns else "question_id"
    required = {id_column, "question", "option_1", "option_2", "option_3", "option_4"}
    missing = required.difference(questions.columns)
    if missing:
        raise ValueError(f"test.csv is missing columns: {sorted(missing)}")

    vlm = None
    if MODEL_DIR.exists():
        try:
            vlm = LocalVLM(MODEL_DIR)
        except Exception as exc:
            print(f"[warn] Could not load local VLM, falling back to option 5: {exc}")

    rows = []
    for _, row in questions.iterrows():
        qid = str(row[id_column])
        answer = vlm.answer(stitched_map_path, row) if vlm is not None else 5
        if answer not in VALID_OPTIONS:
            answer = 5
        rows.append({"id": qid, "question_num": qid, "option": answer})
    return pd.DataFrame(rows, columns=["id", "question_num", "option"])


def run(test_dir: Path) -> None:
    test_dir = test_dir.resolve()
    patches_dir = test_dir / "patches"
    test_csv = test_dir / "test.csv"

    patches = load_patches(patches_dir)
    stitched = stitch_patches(patches)
    stitched_map_path = Path("stitched_map.jpg")
    cv2.imwrite(str(stitched_map_path), cv2.cvtColor(stitched, cv2.COLOR_RGB2BGR))

    submission = build_submission(test_csv, stitched_map_path)
    submission.to_csv("submission.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline inference for Project 1")
    parser.add_argument("--test_dir", required=True, help="Directory containing patches/ and test.csv")
    args = parser.parse_args()
    run(Path(args.test_dir))


if __name__ == "__main__":
    main()
