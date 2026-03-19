import torch
import torch.nn as nn
import torchvision.models as models
import lightning as L

# Attempt to import torchreid
try:
    import torchreid
    HAS_TORCHREID = True
except ImportError:
    HAS_TORCHREID = False

def _ensure_sequence(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 4:
        return x.unsqueeze(1)  # B, 1, C, H, W
    if x.ndim == 5:
        return x
    raise ValueError(f"Unsupported input shape {x.shape}")

class ResNetEncoder(nn.Module):
    def __init__(self, embedding_size: int, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        resnet = models.resnet50(weights=weights)

        for name, child in resnet.named_children():
            if name in ["conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3"]:
                for param in child.parameters():
                    param.requires_grad = False

        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_dim = 2048
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, embedding_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ensure_sequence(x)
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        
        if C == 1:
            x = x.repeat(1, 3, 1, 1)

        x = self.backbone(x)     
        x = x.flatten(1)         
        emb = self.fc(x)         
        
        return emb.view(B, S, -1).mean(dim=1)

class ConvNextEncoder(nn.Module):
    def __init__(self, embedding_size: int, pretrained: bool = True):
        super().__init__()
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        self.model = models.convnext_tiny(weights=weights)
        self.feature_dim = 768

        for i, child in enumerate(self.model.features.children()):
            if i < 6: 
                for param in child.parameters():
                    param.requires_grad = False

        self.backbone = nn.Sequential(
            self.model.features,
            self.model.avgpool
        )
        
        self.fc = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(512, embedding_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ensure_sequence(x)
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        if C == 1: x = x.repeat(1, 3, 1, 1)

        x = self.backbone(x)
        emb = self.fc(x)
        return emb.view(B, S, -1).mean(dim=1)

class OSNetEncoder(nn.Module):
    def __init__(self, embedding_size: int, pretrained: bool = True):
        super().__init__()
        if not HAS_TORCHREID:
            raise ImportError("Please install torchreid: pip install torchreid")
        
        self.backbone = torchreid.models.build_model(
            name='osnet_x1_0', num_classes=1, pretrained=pretrained
        )
        self.feature_dim = self.backbone.feature_dim
        self.backbone.classifier = nn.Identity()
        
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ensure_sequence(x)
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        if C == 1: x = x.repeat(1, 3, 1, 1)
        
        feats = self.backbone(x)
        if feats.dim() > 2: feats = feats.view(feats.size(0), -1)
        emb = self.fc(feats)
        return emb.view(B, S, -1).mean(dim=1)

class ViTEncoder(nn.Module):
    def __init__(self, embedding_size: int):
        super().__init__()
        self.vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
        self.vit.heads = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _ensure_sequence(x)
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)
        if C == 1: x = x.repeat(1, 3, 1, 1)
        emb = self.vit(x)
        return emb.view(B, S, -1).mean(dim=1)

class ReIDLightning(L.LightningModule):
    def __init__(
        self,
        embedding_size: int = 128,
        num_classes: int = 88, # <--- FIXED: Added this argument
        lr: float = 1e-4,
        margin: float = 0.0, 
        backbone_type: str = "osnet", 
        anchor_modality: str = "rgb",
        positive_modality: str = "rgb",
        negative_modality: str = "rgb",
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # Select Backbone
        if backbone_type == "osnet":
            self.rgb_encoder = OSNetEncoder(embedding_size)
            self.depth_encoder = OSNetEncoder(embedding_size)
        elif backbone_type == "vit":
            self.rgb_encoder = ViTEncoder(embedding_size)
            self.depth_encoder = ViTEncoder(embedding_size)
        elif backbone_type == "convnext":
            self.rgb_encoder = ConvNextEncoder(embedding_size)
            self.depth_encoder = ConvNextEncoder(embedding_size)
        else:
            self.rgb_encoder = ResNetEncoder(embedding_size)
            self.depth_encoder = ResNetEncoder(embedding_size)

        # Classification Head (used in training, ignored in eval)
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(embedding_size), 
            nn.Linear(embedding_size, num_classes, bias=False)
        )
        self.id_loss = nn.CrossEntropyLoss()

    def encode(self, x: torch.Tensor, modality: str) -> torch.Tensor:
        # This method extracts features and ignores the classifier
        if modality == "rgb":
            return self.rgb_encoder(x)
        if modality == "depth":
            return self.depth_encoder(x)
        raise ValueError(f"Unknown modality {modality}")

    def training_step(self, batch, batch_idx):
        anchor_emb = self.encode(batch["anchor"][self.hparams.anchor_modality], self.hparams.anchor_modality)
        labels = batch["label"] 
        
        logits = self.classifier(anchor_emb) 
        loss = self.id_loss(logits, labels)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        self.log("train/loss", loss, prog_bar=True)
        self.log("train/acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        if "label" not in batch: return
        anchor_emb = self.encode(batch["anchor"][self.hparams.anchor_modality], self.hparams.anchor_modality)
        labels = batch["label"]
        
        logits = self.classifier(anchor_emb)
        loss = self.id_loss(logits, labels)
        
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()
        
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        self.log("val/acc", acc, prog_bar=True)

    def configure_optimizers(self):
        params = filter(lambda p: p.requires_grad, self.parameters())
        return torch.optim.AdamW(params, lr=self.hparams.lr, weight_decay=0.01)