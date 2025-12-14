import hydra
import lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from utils.data import DataConfig, UnifiedReIDDataModule
from utils.models import ReIDLightning


TRACK_MODALITIES = {
    "rgb": {"data_modality": "rgb", "anchor": "rgb", "positive": "rgb", "negative": "rgb"},
    "depth": {"data_modality": "depth", "anchor": "depth", "positive": "depth", "negative": "depth"},
    "cross": {"data_modality": "rgbd", "anchor": "rgb", "positive": "depth", "negative": "depth"},
}


@hydra.main(config_path="config", config_name="train", version_base=None)
def main(cfg: DictConfig):
    L.seed_everything(cfg.seed)

    track_cfg = TRACK_MODALITIES[cfg.track]

    data_cfg: DataConfig = instantiate(cfg.data)
    data_cfg.modality = track_cfg["data_modality"]

    dm = UnifiedReIDDataModule(data_cfg)
    dm.setup("fit")

    model = ReIDLightning(
        embedding_size=cfg.model.embedding_size,
        lr=cfg.model.lr,
        margin=cfg.model.margin,
        anchor_modality=track_cfg["anchor"],
        positive_modality=track_cfg["positive"],
        negative_modality=track_cfg["negative"],
        rgb_layers_not_frozen=cfg.model.rgb_layers_not_frozen,
    )

    trainer = L.Trainer(**cfg.trainer)
    trainer.fit(model=model, datamodule=dm)


if __name__ == "__main__":
    main()
