import torch
import torch.nn as nn
import torchvision.models as models
import lightning as L


class TripletLoss(nn.Module):
    def __init__(self, margin: float):
        super().__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        d_ap = (anchor - positive).pow(2).sum(dim=1)
        d_an = (anchor - negative).pow(2).sum(dim=1)
        return torch.relu(d_ap - d_an + self.margin).mean()


def _ensure_sequence(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        return x.unsqueeze(1)  # B,1,C,H,W
    if x.ndim == 5:
        return x
    raise ValueError(f"Unsupported input shape {x.shape}")


class DepthEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(1, 64, 3),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, stride=2),
        )
        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(64, 512, 3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
        )
        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2),
        )
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512 * 3 * 3, out_features=embedding_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ensure_sequence(x)  # B,S,1,H,W
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_3(x)
        x = self.conv_layer_3(x)
        x = self.encoder(x)
        x = x.view(B, S, -1).mean(dim=1)
        return x


class RGBEncoder(nn.Module):
    def __init__(self, embedding_size: int, layers_not_frozen: int = 4):
        super().__init__()
        resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        last_hidden = resnet50.fc.in_features
        self.feature_extractor = nn.Sequential(*list(resnet50.children())[:-1])
        self.embedding_layer = nn.Sequential(
            nn.Linear(last_hidden, 1024),
            nn.ReLU(),
            nn.Linear(1024, embedding_size),
            nn.ReLU(),
        )
        self._freeze_feature_extractor(layers_not_frozen)

    def _freeze_feature_extractor(self, layers_not_frozen: int):
        layers = list(self.feature_extractor.children())
        to_freeze = layers[:-layers_not_frozen] if layers_not_frozen > 0 else layers
        for layer in to_freeze:
            for p in layer.parameters():
                p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ensure_sequence(x)  # B,S,C,H,W
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        feats = self.feature_extractor(x)
        feats = feats.view(feats.size(0), -1)
        emb = self.embedding_layer(feats)
        emb = emb.view(B, S, -1).mean(dim=1)
        return emb


class ReIDLightning(L.LightningModule):
    def __init__(
        self,
        embedding_size: int = 128,
        lr: float = 1e-3,
        margin: float = 0.25,
        anchor_modality: str = "rgb",
        positive_modality: str = "rgb",
        negative_modality: str = "rgb",
        rgb_layers_not_frozen: int = 4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.depth_encoder = DepthEncoder(embedding_size)
        self.rgb_encoder = RGBEncoder(embedding_size, layers_not_frozen=rgb_layers_not_frozen)
        self.loss_fn = TripletLoss(margin)

    def encode(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        if modality == "rgb":
            return self.rgb_encoder(x)
        if modality == "depth":
            return self.depth_encoder(x)
        raise ValueError(f"Unknown modality {modality}")

    def training_step(self, batch, batch_idx):
        anchor = batch["anchor"]
        positive = batch["positive"]
        negative = batch["negative"]

        anchor_x = anchor[self.hparams.anchor_modality]
        positive_x = positive[self.hparams.positive_modality]
        negative_x = negative[self.hparams.negative_modality]

        anchor_out = self.encode(anchor_x, self.hparams.anchor_modality)
        positive_out = self.encode(positive_x, self.hparams.positive_modality)
        negative_out = self.encode(negative_x, self.hparams.negative_modality)

        loss = self.loss_fn(anchor_out, positive_out, negative_out)

        d_ap = (anchor_out - positive_out).pow(2).sum(1)
        d_an = (anchor_out - negative_out).pow(2).sum(1)
        correct = (d_ap < d_an).float().mean()

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/accuracy", correct, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if not {"anchor", "positive", "negative"} <= set(batch.keys()):
            return

        anchor_x = batch["anchor"][self.hparams.anchor_modality]
        positive_x = batch["positive"][self.hparams.positive_modality]
        negative_x = batch["negative"][self.hparams.negative_modality]

        anchor_out = self.encode(anchor_x, self.hparams.anchor_modality)
        positive_out = self.encode(positive_x, self.hparams.positive_modality)
        negative_out = self.encode(negative_x, self.hparams.negative_modality)

        loss = self.loss_fn(anchor_out, positive_out, negative_out)
        d_ap = (anchor_out - positive_out).pow(2).sum(1)
        d_an = (anchor_out - negative_out).pow(2).sum(1)
        correct = (d_ap < d_an).float().mean()

        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/accuracy", correct, prog_bar=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=self.hparams.lr)
        return optimizer
