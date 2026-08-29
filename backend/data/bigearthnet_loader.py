"""
SatQuery AI - BigEarthNet Dataset Loader.

Parses BigEarthNet annotation format and links co-registered Sentinel-1 (SAR)
and Sentinel-2 (Optical) image patches for VQA-style training.

Supports two layouts:
  1. BigEarthNet-S2 (Sentinel-2 optical, 10-60m multi-spectral bands)
  2. BigEarthNet-S1 (Sentinel-1 SAR, VV/VH dual-pol backscatter)

The loader generates VQA training triplets (image_path, question, answer) that
are directly consumable by train_qwen_vl_lora.py.

Usage:
    from data.bigearthnet_loader import BigEarthNetLoader

    loader = BigEarthNetLoader(
        s2_root="/path/to/BigEarthNet-S2",
        s1_root="/path/to/BigEarthNet-S1",  # optional for fusion
        labels_file="/path/to/labels.json",
    )
    vqa_dataset = loader.build_vqa_dataset()
    loader.export_json("training_data.json")
"""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# BigEarthNet 19-class taxonomy (updated CLC labels)
# ---------------------------------------------------------------------------

BIGEARTHNET_19_CLASSES: List[str] = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grasslands and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland/shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
]

# VQA question templates for diverse training signal
_VQA_QUESTION_TEMPLATES: List[str] = [
    "What type of land cover is visible in this satellite image?",
    "Classify the land use categories in this image.",
    "What are the dominant surface features shown?",
    "Describe the land cover composition of this scene.",
    "Identify the primary land cover classes present.",
    "What geographical features can you identify?",
    "Is there any agricultural land visible? If so, what kind?",
    "Are there any water bodies in this satellite image?",
    "Is there urban or built-up area in this image?",
    "What percentage of this scene appears to be vegetated?",
]

_GROUNDING_QUESTION_TEMPLATES: List[str] = [
    "Where is the {class_name} located in this image?",
    "Point to the region showing {class_name}.",
    "Find the {class_name} area and provide its bounding box.",
]

_CHANGE_QUESTION_TEMPLATES: List[str] = [
    "What has changed between these two temporal acquisitions?",
    "Describe the differences between the earlier and later image.",
    "Has any land cover transition occurred? If so, what type?",
    "Compare the two images and identify areas of change.",
]


class BigEarthNetPatch:
    """Represents a single BigEarthNet patch with metadata and labels."""

    def __init__(
        self,
        patch_id: str,
        s2_path: Optional[str] = None,
        s1_path: Optional[str] = None,
        labels: Optional[List[str]] = None,
        split: str = "train",
    ):
        self.patch_id = patch_id
        self.s2_path = s2_path
        self.s1_path = s1_path
        self.labels = labels or []
        self.split = split

    @property
    def has_s2(self) -> bool:
        return self.s2_path is not None and os.path.exists(self.s2_path)

    @property
    def has_s1(self) -> bool:
        return self.s1_path is not None and os.path.exists(self.s1_path)

    @property
    def has_both(self) -> bool:
        return self.has_s2 and self.has_s1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "patch_id": self.patch_id,
            "s2_path": self.s2_path,
            "s1_path": self.s1_path,
            "labels": self.labels,
            "split": self.split,
        }


class BigEarthNetLoader:
    """
    Loads and indexes BigEarthNet patches from disk.

    Supports two directory structures:
      1. Flat: all patch directories under s2_root / s1_root
      2. Nested: patches grouped by Sentinel-2 tile ID

    Args:
        s2_root: Path to BigEarthNet-S2 directory (Sentinel-2 optical patches)
        s1_root: Path to BigEarthNet-S1 directory (Sentinel-1 SAR patches, optional)
        labels_file: Path to JSON file mapping patch_id -> list of label strings.
                     If None, labels are read from per-patch JSON files.
        split_file: Path to JSON with train/val/test splits (optional)
    """

    def __init__(
        self,
        s2_root: str,
        s1_root: Optional[str] = None,
        labels_file: Optional[str] = None,
        split_file: Optional[str] = None,
    ):
        self.s2_root = Path(s2_root) if s2_root else None
        self.s1_root = Path(s1_root) if s1_root else None
        self._labels_map: Dict[str, List[str]] = {}
        self._splits_map: Dict[str, str] = {}  # patch_id -> "train"/"val"/"test"
        self.patches: List[BigEarthNetPatch] = []

        if labels_file and os.path.exists(labels_file):
            self._load_labels_file(labels_file)

        if split_file and os.path.exists(split_file):
            self._load_split_file(split_file)

        self._index_patches()

    def _load_labels_file(self, path: str) -> None:
        """Load a JSON mapping from patch_id to list of label strings."""
        with open(path, "r") as f:
            data = json.load(f)
        if isinstance(data, dict):
            self._labels_map = data
        elif isinstance(data, list):
            # Handle list-of-dicts format: [{"patch_id": ..., "labels": [...]}, ...]
            for item in data:
                if isinstance(item, dict) and "patch_id" in item:
                    self._labels_map[item["patch_id"]] = item.get("labels", [])

    def _load_split_file(self, path: str) -> None:
        """Load train/val/test split assignments."""
        with open(path, "r") as f:
            data = json.load(f)
        # Supports {"train": [...], "val": [...], "test": [...]} format
        for split_name, patch_ids in data.items():
            if isinstance(patch_ids, list):
                for pid in patch_ids:
                    self._splits_map[pid] = split_name

    def _index_patches(self) -> None:
        """Scan S2 and S1 directories to build the patch index."""
        s2_patches: Dict[str, str] = {}
        s1_patches: Dict[str, str] = {}

        if self.s2_root and self.s2_root.exists():
            for entry in sorted(self.s2_root.iterdir()):
                if entry.is_dir():
                    # Each patch is a directory named with the patch_id
                    s2_patches[entry.name] = str(entry)

        if self.s1_root and self.s1_root.exists():
            for entry in sorted(self.s1_root.iterdir()):
                if entry.is_dir():
                    s1_patches[entry.name] = str(entry)

        # Build unified patch list (S2 as primary, S1 linked if available)
        all_ids = set(list(s2_patches.keys()) + list(s1_patches.keys()))

        for patch_id in sorted(all_ids):
            labels = self._labels_map.get(patch_id, [])

            # Try to read labels from per-patch JSON if not in global file
            if not labels:
                labels = self._read_patch_labels(s2_patches.get(patch_id))

            split = self._splits_map.get(patch_id, "train")

            patch = BigEarthNetPatch(
                patch_id=patch_id,
                s2_path=s2_patches.get(patch_id),
                s1_path=self._find_s1_pair(patch_id, s1_patches),
                labels=labels,
                split=split,
            )
            self.patches.append(patch)

    def _find_s1_pair(self, s2_patch_id: str, s1_patches: Dict[str, str]) -> Optional[str]:
        """
        Find the co-registered S1 patch for a given S2 patch.
        BigEarthNet uses a naming convention where S1 patches correspond
        to S2 patches via the same tile/date structure.
        """
        if s2_patch_id in s1_patches:
            return s1_patches[s2_patch_id]

        # Try matching by stripping the sensor prefix
        # S2 format: S2A_MSIL2A_..._T33UUP_...
        # S1 format: S1A_IW_GRDH_..._T33UUP_...
        for s1_id, s1_path in s1_patches.items():
            # Match by tile ID (e.g., T33UUP) embedded in the name
            s2_parts = s2_patch_id.split("_")
            s1_parts = s1_id.split("_")
            tile_matches = set(s2_parts) & set(s1_parts)
            if len(tile_matches) >= 2:  # At least tile ID + one date component
                return s1_path

        return None

    @staticmethod
    def _read_patch_labels(patch_dir: Optional[str]) -> List[str]:
        """Read labels from a per-patch JSON metadata file."""
        if not patch_dir:
            return []
        json_files = list(Path(patch_dir).glob("*_labels_metadata.json"))
        if not json_files:
            json_files = list(Path(patch_dir).glob("*.json"))
        if not json_files:
            return []
        try:
            with open(json_files[0], "r") as f:
                data = json.load(f)
            return data.get("labels", data.get("new_labels", []))
        except Exception:
            return []

    # -------------------------------------------------------------------
    # VQA Dataset Generation
    # -------------------------------------------------------------------

    def build_vqa_dataset(
        self,
        include_fusion: bool = True,
        include_grounding: bool = True,
        max_samples: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate VQA training triplets from indexed patches.

        Returns list of dicts:
            {
                "image": str (path to patch directory or band file),
                "question": str,
                "answer": str,
                "task_type": str ("vqa" | "fusion" | "grounding" | "change"),
                "patch_id": str,
                "labels": List[str],
            }
        """
        dataset: List[Dict[str, Any]] = []

        for patch in self.patches:
            if not patch.labels:
                continue

            # --- Standard VQA questions ---
            question = random.choice(_VQA_QUESTION_TEMPLATES)
            answer = self._format_labels_answer(patch.labels)

            image_path = self._get_primary_image_path(patch)
            if image_path:
                dataset.append({
                    "image": image_path,
                    "question": question,
                    "answer": answer,
                    "task_type": "vqa",
                    "patch_id": patch.patch_id,
                    "labels": patch.labels,
                })

            # --- Grounding questions ---
            if include_grounding and patch.labels:
                label = random.choice(patch.labels)
                template = random.choice(_GROUNDING_QUESTION_TEMPLATES)
                grounding_q = template.format(class_name=label.lower())
                dataset.append({
                    "image": image_path,
                    "question": grounding_q,
                    "answer": f"The {label.lower()} area is visible in the image. [Grounding requires spatial annotation]",
                    "task_type": "grounding",
                    "patch_id": patch.patch_id,
                    "labels": [label],
                })

            # --- Fusion questions (requires both S1 + S2) ---
            if include_fusion and patch.has_both:
                dataset.append({
                    "image": patch.s2_path,
                    "image_sar": patch.s1_path,
                    "question": "Using both optical and SAR data, what land cover types are present?",
                    "answer": f"Combining optical (Sentinel-2) and SAR (Sentinel-1) observations, the scene contains: {answer}",
                    "task_type": "fusion",
                    "patch_id": patch.patch_id,
                    "labels": patch.labels,
                })

        if max_samples and len(dataset) > max_samples:
            random.shuffle(dataset)
            dataset = dataset[:max_samples]

        return dataset

    def build_change_dataset(
        self,
        temporal_pairs: Optional[List[Tuple[str, str]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Generate change detection VQA pairs.

        If temporal_pairs is provided, it should be a list of (patch_id_t1, patch_id_t2)
        tuples representing known temporal pairs.
        Otherwise, randomly pairs patches from the same tile for synthetic training.
        """
        dataset: List[Dict[str, Any]] = []

        if temporal_pairs:
            patch_map = {p.patch_id: p for p in self.patches}
            for pid1, pid2 in temporal_pairs:
                p1 = patch_map.get(pid1)
                p2 = patch_map.get(pid2)
                if not p1 or not p2:
                    continue

                changed_classes = list(set(p2.labels) - set(p1.labels))
                lost_classes = list(set(p1.labels) - set(p2.labels))

                if changed_classes or lost_classes:
                    answer_parts = []
                    if changed_classes:
                        answer_parts.append(f"New: {', '.join(changed_classes)}")
                    if lost_classes:
                        answer_parts.append(f"Lost: {', '.join(lost_classes)}")
                    answer = "; ".join(answer_parts)
                else:
                    answer = "No significant land cover change detected between the two dates."

                question = random.choice(_CHANGE_QUESTION_TEMPLATES)
                dataset.append({
                    "image_t1": self._get_primary_image_path(p1),
                    "image_t2": self._get_primary_image_path(p2),
                    "question": question,
                    "answer": answer,
                    "task_type": "change",
                    "patch_id_t1": pid1,
                    "patch_id_t2": pid2,
                    "labels_t1": p1.labels,
                    "labels_t2": p2.labels,
                })

        return dataset

    # -------------------------------------------------------------------
    # Export
    # -------------------------------------------------------------------

    def export_json(self, output_path: str, max_samples: Optional[int] = None) -> str:
        """
        Export VQA dataset to JSON file, consumable by train_qwen_vl_lora.py.
        Returns the output path.
        """
        dataset = self.build_vqa_dataset(max_samples=max_samples)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(dataset, f, indent=2)
        print(f"Exported {len(dataset)} VQA samples to {output_path}")
        return output_path

    def get_split(self, split: str) -> List[BigEarthNetPatch]:
        """Return patches belonging to a specific split (train/val/test)."""
        return [p for p in self.patches if p.split == split]

    def summary(self) -> Dict[str, Any]:
        """Return a summary of the loaded dataset."""
        n_s2 = sum(1 for p in self.patches if p.has_s2)
        n_s1 = sum(1 for p in self.patches if p.has_s1)
        n_both = sum(1 for p in self.patches if p.has_both)
        n_labelled = sum(1 for p in self.patches if p.labels)
        splits = {}
        for p in self.patches:
            splits[p.split] = splits.get(p.split, 0) + 1

        return {
            "total_patches": len(self.patches),
            "s2_available": n_s2,
            "s1_available": n_s1,
            "paired_s1_s2": n_both,
            "labelled": n_labelled,
            "splits": splits,
            "label_classes": len(set(lbl for p in self.patches for lbl in p.labels)),
        }

    # -------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------

    @staticmethod
    def _get_primary_image_path(patch: BigEarthNetPatch) -> Optional[str]:
        """Get the primary image file path for a patch (first .tif found)."""
        if patch.s2_path and os.path.isdir(patch.s2_path):
            tifs = sorted(Path(patch.s2_path).glob("*.tif"))
            if tifs:
                return str(tifs[0])
            return patch.s2_path
        if patch.s1_path and os.path.isdir(patch.s1_path):
            tifs = sorted(Path(patch.s1_path).glob("*.tif"))
            if tifs:
                return str(tifs[0])
            return patch.s1_path
        return patch.s2_path or patch.s1_path

    @staticmethod
    def _format_labels_answer(labels: List[str]) -> str:
        """Format a list of labels into a natural-language answer."""
        if not labels:
            return "No land cover classes identified."
        if len(labels) == 1:
            return f"The primary land cover is {labels[0].lower()}."
        label_str = ", ".join(lbl.lower() for lbl in labels[:-1])
        return f"The scene contains {label_str}, and {labels[-1].lower()}."


# ---------------------------------------------------------------------------
# CLI for standalone usage
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="BigEarthNet Dataset Loader for SatQuery AI")
    parser.add_argument("--s2_root", type=str, required=True, help="Path to BigEarthNet-S2 directory")
    parser.add_argument("--s1_root", type=str, default=None, help="Path to BigEarthNet-S1 directory (optional)")
    parser.add_argument("--labels_file", type=str, default=None, help="Path to labels JSON file")
    parser.add_argument("--split_file", type=str, default=None, help="Path to splits JSON file")
    parser.add_argument("--output", type=str, default="./training_data.json", help="Output JSON path")
    parser.add_argument("--max_samples", type=int, default=None, help="Max training samples")

    args = parser.parse_args()

    loader = BigEarthNetLoader(
        s2_root=args.s2_root,
        s1_root=args.s1_root,
        labels_file=args.labels_file,
        split_file=args.split_file,
    )

    print("\n📊 Dataset Summary:")
    summary = loader.summary()
    for k, v in summary.items():
        print(f"  {k}: {v}")

    loader.export_json(args.output, max_samples=args.max_samples)
