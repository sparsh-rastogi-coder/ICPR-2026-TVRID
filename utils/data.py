import os
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import lightning as L
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
import torchvision.transforms.functional as TF # Needed for manual aug


@dataclass
class SequenceConfig:
    length: int = 1
    sampling: str = "even"


@dataclass
class TransformConfig:
    resize: int = 224 # <--- KEEP 224 for ViT/ConvNeXt Compatibility
    crop: int = 224
    rgb_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    rgb_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    depth_mean: float = 0.0
    depth_std: float = 0.25
    do_flip: bool = True 
    do_pad_crop: bool = True
    erasing_prob: float = 0.5


@dataclass
class DataConfig:
    root: str = "data/DB_extracted"
    train_csv: str = "data/DB_extracted/train_labels.csv"
    eval_csv: str = "data/DB_extracted/public_test_labels.csv"
    train_subdir: str = "train"
    eval_subdir: str = "test_public"
    modality: str = "rgb"
    val_mode: str = "train"
    sequence: SequenceConfig = field(default_factory=SequenceConfig)
    transforms: TransformConfig = field(default_factory=TransformConfig)
    batch_size: int = 8
    num_workers: int = 4
    shuffle: bool = True
    pin_memory: bool = True
    persistent_workers: bool = False
    mask_rgb_with_depth: bool = False
    depth_mask_threshold: float = 0.2


def build_transforms(cfg: TransformConfig) -> Tuple[T.Compose, T.Compose]:
    rgb_transform = T.Compose([
        T.Resize(cfg.resize),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=cfg.rgb_mean, std=cfg.rgb_std),
    ])
    depth_transform = T.Compose([
        T.Resize(cfg.resize),
        T.ConvertImageDtype(torch.float32),
        T.Normalize(mean=[cfg.depth_mean], std=[cfg.depth_std]),
    ])
    return rgb_transform, depth_transform


def _split_path_components(path: str) -> List[str]:
    return [p for p in path.replace("\\", os.sep).split(os.sep) if p]


class UnifiedReIDDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        root: str,
        modality: str = "rgb",
        mode: str = "train",
        sequence: Optional[SequenceConfig] = None,
        rgb_transform: Optional[T.Compose] = None,
        depth_transform: Optional[T.Compose] = None,
        train_subdir: str = "train",
        eval_subdir: str = "test_public",
        sampling_strategy: str = "even",
        mask_rgb_with_depth: bool = False,
        depth_mask_threshold: float = 0.2,
    ) -> None:
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.modality = modality
        self.mode = mode
        self.sequence = sequence or SequenceConfig()
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform
        self.base_dir = os.path.join(root, train_subdir if mode == "train" else eval_subdir)
        self.sampling_strategy = sampling_strategy
        self.mask_rgb_with_depth = mask_rgb_with_depth
        self.depth_mask_threshold = depth_mask_threshold

        self._validate_columns()
        
        # --- CRITICAL RESTORATION START ---
        # Logic to map Person IDs to Integer Labels (0..87)
        if mode == "train":
            unique_pids = sorted(self.df["person_id"].unique().astype(str))
            self.pid_to_label = {pid: i for i, pid in enumerate(unique_pids)}
            self.num_classes = len(unique_pids) # <--- Defines num_classes
            
            self._person_to_indices = self._index_by_person()
            self._negative_pool = self._build_negative_pool()
        # --- CRITICAL RESTORATION END ---

    def _validate_columns(self) -> None:
        if self.mode == "train":
            required = {"gallery_id", "person_id", "path"}
        else:
            required = {"gallery_id", "path"}
        # loose check
        pass 

    def _index_by_person(self) -> Dict[str, List[int]]:
        grouped = self.df.groupby("person_id").groups
        return {str(pid): list(idxs) for pid, idxs in grouped.items()}

    def _build_negative_pool(self) -> Dict[str, List[int]]:
        all_indices = list(range(len(self.df)))
        pool = {}
        for pid, idxs in self._person_to_indices.items():
            pool[pid] = [i for i in all_indices if i not in idxs]
        return pool

    def __len__(self) -> int:
        return len(self.df)

    def _resolve_passage_dir(self, relative_path: str) -> str:
        return os.path.join(self.base_dir, *_split_path_components(relative_path))

    def _list_stems(self, passage_dir: str) -> List[str]:
        files = os.listdir(passage_dir)
        rgb_stems = {f.split("_RGB")[0] for f in files if "_RGB" in f}
        depth_stems = {f.split("_depth")[0] for f in files if "_depth" in f}
        
        if self.modality == "rgb": stems = rgb_stems
        elif self.modality == "depth": stems = depth_stems
        else: stems = rgb_stems & depth_stems
        
        if not stems: return [] # Handle empty better
        return sorted(stems)

    def _select_stems(self, stems: Sequence[str]) -> List[str]:
        if not stems: return []
        n = len(stems)
        length = max(1, self.sequence.length)
        if length == 1:
            return [stems[n // 2]]
        idxs = np.linspace(0, n - 1, num=length, dtype=int).tolist()
        return [stems[i] for i in idxs]

    def _find_candidate(self, passage_dir: str, stem: str, suffixes: Iterable[str]) -> Optional[str]:
        for suffix in suffixes:
            candidate = os.path.join(passage_dir, f"{stem}{suffix}")
            if os.path.exists(candidate): return candidate
        return None

    def _load_depth_array(self, path: str) -> np.ndarray:
        depth = np.array(Image.open(path))
        if depth.ndim == 3: depth = depth[..., 0]
        depth = depth.astype(np.float32)
        if depth.max() > 1e3: depth = depth / 10000.0
        return depth

    def _load_frame(self, passage_dir: str, stem: str) -> Dict[str, torch.Tensor]:
        sample = {}
        # Depth Loading
        if self.modality in {"depth", "rgbd"}:
            dpath = self._find_candidate(passage_dir, stem, ["_depth.png", "_depth_depth.png", "_D.png"])
            if dpath:
                d = self._load_depth_array(dpath)
                sample["depth"] = torch.from_numpy(d).unsqueeze(0)

        # RGB Loading
        if self.modality in {"rgb", "rgbd"}:
            rpath = self._find_candidate(passage_dir, stem, ["_RGB.png", "_RGB_person.png"])
            if rpath:
                img = Image.open(rpath).convert("RGB")
                sample["rgb"] = torch.from_numpy(np.array(img)).permute(2, 0, 1)

        return sample

    def _apply_transform(self, tensor: torch.Tensor, transform: Optional[T.Compose], is_training: bool = False) -> torch.Tensor:
        if transform is None: return tensor
        
        def apply_base(t):
            if t.ndim == 3: return transform(t)
            return torch.stack([transform(f) for f in t])

        if not is_training:
            return apply_base(tensor)

        # Augmentation
        if random.random() < 0.5:
            tensor = TF.hflip(tensor)
            
        if True: # Random Crop
            _, h, w = tensor.shape[-3:]
            pad = 10
            tensor = TF.pad(tensor, pad)
            i, j, th, tw = T.RandomCrop.get_params(tensor, output_size=(h, w))
            tensor = TF.crop(tensor, i, j, th, tw)

        tensor = apply_base(tensor)

        # Random Erasing (Reduced scale to prevent full occlusion)
        if random.random() < 0.5:
            eraser = T.RandomErasing(p=1.0, scale=(0.02, 0.15)) 
            if tensor.ndim == 4:
                tensor = torch.stack([eraser(f) for f in tensor])
            else:
                tensor = eraser(tensor)
        return tensor

    def _load_sample(self, passage_path: str) -> Dict[str, torch.Tensor]:
        passage_dir = self._resolve_passage_dir(passage_path)
        stems = self._select_stems(self._list_stems(passage_dir))
        frames = [self._load_frame(passage_dir, stem) for stem in stems]

        stacked = {}
        for f in frames:
            for k, v in f.items():
                stacked.setdefault(k, []).append(v)

        sample = {}
        for k, v in stacked.items():
            t = torch.stack(v) if len(v) > 1 else v[0]
            is_train = (self.mode == "train")
            if k == "rgb":
                t = self._apply_transform(t, self.rgb_transform, is_train)
            elif k == "depth":
                t = self._apply_transform(t, self.depth_transform, is_train)
            sample[k] = t
        
        sample["path"] = passage_path
        return sample

    def _sample_positive_index(self, anchor_idx: int, person_id: str) -> int:
        candidates = [i for i in self._person_to_indices[str(person_id)] if i != anchor_idx]
        if not candidates: candidates = self._person_to_indices[str(person_id)]
        return random.choice(candidates)

    def _sample_negative_index(self, person_id: str) -> int:
        return random.choice(self._negative_pool[str(person_id)])

    def __getitem__(self, idx: int):
        if self.mode == "train":
            anchor_row = self.df.iloc[idx]
            pos_idx = self._sample_positive_index(idx, anchor_row["person_id"])
            neg_idx = self._sample_negative_index(anchor_row["person_id"])

            anchor = self._load_sample(anchor_row["path"])
            positive = self._load_sample(self.df.iloc[pos_idx]["path"])
            negative = self._load_sample(self.df.iloc[neg_idx]["path"])

            # --- CRITICAL: Add Label for Loss ---
            person_label = self.pid_to_label[str(anchor_row["person_id"])]

            return {
                "anchor": anchor,
                "positive": positive,
                "negative": negative,
                "label": torch.tensor(person_label, dtype=torch.long), # <--- NEEDED FOR ID LOSS
                "person_id": anchor_row["person_id"],
                "gallery_id": anchor_row["gallery_id"],
            }

        eval_row = self.df.iloc[idx]
        sample = self._load_sample(eval_row["path"])
        sample["gallery_id"] = eval_row["gallery_id"]
        return sample

class UnifiedReIDDataModule(L.LightningDataModule):
    # ... (Same as before, abbreviated for brevity) ...
    def __init__(self, cfg: DataConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.rgb_transform, self.depth_transform = build_transforms(cfg.transforms)
        self.train_set = None
        self.eval_set = None

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_set = UnifiedReIDDataset(
                csv_path=self.cfg.train_csv,
                root=self.cfg.root,
                modality=self.cfg.modality,
                mode="train", # triggers creation of self.num_classes
                sequence=self.cfg.sequence,
                rgb_transform=self.rgb_transform,
                depth_transform=self.depth_transform,
                train_subdir=self.cfg.train_subdir,
                eval_subdir=self.cfg.eval_subdir
            )
        if stage != "fit":
            self.eval_set = UnifiedReIDDataset(
                csv_path=self.cfg.eval_csv,
                root=self.cfg.root,
                modality=self.cfg.modality,
                mode=self.cfg.val_mode,
                sequence=self.cfg.sequence,
                rgb_transform=self.rgb_transform,
                depth_transform=self.depth_transform,
                train_subdir=self.cfg.train_subdir,
                eval_subdir=self.cfg.eval_subdir
            )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.cfg.batch_size, shuffle=self.cfg.shuffle, num_workers=self.cfg.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.eval_set, batch_size=self.cfg.batch_size, shuffle=False, num_workers=self.cfg.num_workers)