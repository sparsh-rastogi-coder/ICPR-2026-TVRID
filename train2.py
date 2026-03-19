import hydra
import torch
import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from utils.data import DataConfig, UnifiedReIDDataModule
# Import from models3 for ConvNext/ViT/OSNet
from utils.models3 import ReIDLightning 

torch.set_float32_matmul_precision("high")

TRACK_MODALITIES = {
    "rgb": {"data_modality": "rgb", "anchor": "rgb", "positive": "rgb", "negative": "rgb"},
    "depth": {"data_modality": "depth", "anchor": "depth", "positive": "depth", "negative": "depth"},
    "cross": {"data_modality": "rgbd", "anchor": "rgb", "positive": "depth", "negative": "depth"},
}

@hydra.main(config_path="config", config_name="train", version_base=None)
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed)

    track_cfg = TRACK_MODALITIES[cfg.track]

    # Instantiate Data Config
    data_cfg: DataConfig = instantiate(cfg.data)
    data_cfg.modality = track_cfg["data_modality"]

    # Setup Data Module
    dm = UnifiedReIDDataModule(data_cfg)
    dm.setup("fit")
    
    # --- FIX FOR "num_classes is not defined" ---
    # We dynamically get the count from the dataset
    num_classes = dm.train_set.num_classes
    print(f"Training with {num_classes} identities.")

    # Initialize Model
    model = ReIDLightning(
        embedding_size=cfg.model.embedding_size,
        num_classes=num_classes, # <--- Pass the variable here
        lr=cfg.model.lr,
        margin=cfg.model.margin,
        backbone_type=cfg.model.get("backbone_type", "osnet"), 
        anchor_modality=track_cfg["anchor"],
        positive_modality=track_cfg["positive"],
        negative_modality=track_cfg["negative"],
    )

    trainer_conf = OmegaConf.to_container(cfg.trainer, resolve=True)
    if "callbacks" in trainer_conf and trainer_conf["callbacks"]:
        trainer_conf["callbacks"] = [instantiate(cb) for cb in trainer_conf["callbacks"]]

    trainer = L.Trainer(**trainer_conf)
    trainer.fit(model=model, datamodule=dm)

if __name__ == "__main__":
    main()