"""
SegFormer-B2 + SE-Gated Stage-2 Slope Fusion for Landslide Segmentation
========================================================================
Results saved to: output/train_stage2_slope_B2/

"""

import csv
import json
import logging
import math
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from sklearn.model_selection import train_test_split
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from transformers import SegformerForSemanticSegmentation

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent
IMG_DIR  = ROOT / "data" / "IMG64"
SLP_DIR  = ROOT / "data" / "slope"
MASK_DIR = ROOT / "mask64"
OUT_DIR  = ROOT / "output" / "train_stage2_slope_B2"

# ── Config ─────────────────────────────────────────────────────────────────────
BACKBONE       = "nvidia/mit-b2"
IMG_SIZE       = 256
EPOCHS         = 12
BATCH_SIZE     = 16
LR             = 6e-5
WEIGHT_DECAY   = 0.01
WARMUP_EPOCHS  = 5
EARLY_STOP_PAT = 15
FOCAL_ALPHA    = 0.75
FOCAL_GAMMA    = 2.0
DICE_WEIGHT    = 0.5
N_TEST_SAMPLES = 30
SAVE_EVERY     = 5
NUM_WORKERS    = 4
SLOPE_MAX      = 60.0
RGB_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
RGB_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEVICE   = "cuda:1" if torch.cuda.is_available() else "cpu"
AUX_CH   = 1   # Slope only


def get_splits(seed=42):
    ids = sorted(re.search(r"IMG_(\d+)\.tif", f).group(1)
                 for f in os.listdir(IMG_DIR) if f.endswith(".tif"))
    train, rest = train_test_split(ids, test_size=0.2, random_state=seed)
    val, test   = train_test_split(rest, test_size=0.5, random_state=seed)
    return train, val, test


# ── Dataset ────────────────────────────────────────────────────────────────────
class LandslideDatasetSlope(Dataset):
    """Returns (img_t, aux_t, mask_t, id_)  where aux_t = Slope (1ch)."""
    def __init__(self, ids, augment=False):
        self.ids = ids; self.augment = augment

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        id_  = self.ids[idx]
        img  = Image.open(IMG_DIR  / f"IMG_{id_}.tif").convert("RGB")
        slp  = Image.open(SLP_DIR  / f"Slope_{id_}.tif")
        mask = Image.open(MASK_DIR / f"mask_{id_}.tif")

        img  = img.resize( (IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        slp  = slp.resize( (IMG_SIZE, IMG_SIZE), Image.BILINEAR)
        mask = mask.resize((IMG_SIZE, IMG_SIZE), Image.NEAREST)

        if self.augment:
            if np.random.rand() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                slp = slp.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if np.random.rand() > 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
                slp = slp.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            if np.random.rand() > 0.5:
                angle = np.random.choice([90, 180, 270])
                img = img.rotate(angle); slp = slp.rotate(angle); mask = mask.rotate(angle)
            if np.random.rand() > 0.5:
                img = ImageEnhance.Brightness(img).enhance(np.random.uniform(0.6, 1.4))
            if np.random.rand() > 0.5:
                img = ImageEnhance.Contrast(img).enhance(np.random.uniform(0.6, 1.4))
            if np.random.rand() > 0.5:
                img = ImageEnhance.Color(img).enhance(np.random.uniform(0.7, 1.3))

        img_np = np.array(img, dtype=np.float32) / 255.0
        img_t  = torch.from_numpy(((img_np - RGB_MEAN) / RGB_STD).transpose(2, 0, 1))

        slp_np = np.array(slp, dtype=np.float32)
        slp_np = np.where(slp_np > 127.0, 0.0, slp_np) / SLOPE_MAX
        aux_t  = torch.from_numpy(slp_np[np.newaxis].astype(np.float32))  # (1, H, W)

        mask_t = torch.from_numpy(np.array(mask)).long()
        return img_t, aux_t, mask_t, id_


# ── Architecture ───────────────────────────────────────────────────────────────
class AuxEncoder(nn.Module):

    def __init__(self, in_ch, stage2_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch,  32,         3, stride=2, padding=1), nn.BatchNorm2d(32),         nn.ReLU(inplace=True),
            nn.Conv2d(32,     64,         3, stride=2, padding=1), nn.BatchNorm2d(64),         nn.ReLU(inplace=True),
            nn.Conv2d(64,     stage2_ch,  3, stride=2, padding=1), nn.BatchNorm2d(stage2_ch),  nn.ReLU(inplace=True),
        )

    def forward(self, x): return self.net(x)


class SEGate(nn.Module):
    """
    Squeeze-and-Excitation gate.
    Uses globally-pooled auxiliary features to generate per-channel attention
    weights for the RGB Stage-2 feature map.
    """
    def __init__(self, rgb_ch, aux_ch, reduction=16):
        super().__init__()
        mid = max(rgb_ch // reduction, 4)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(aux_ch, mid), nn.ReLU(inplace=True),
            nn.Linear(mid, rgb_ch), nn.Sigmoid(),
        )

    def forward(self, f_rgb, f_aux):
        attn = self.gate(f_aux).view(f_rgb.shape[0], -1, 1, 1)  # (B, rgb_ch, 1, 1)
        return f_rgb * attn


class MultiModalStage2(nn.Module):
    """SegFormer-B2 with SE-gated auxiliary injection at encoder Stage-2."""
    def __init__(self, backbone=BACKBONE, aux_channels=AUX_CH):
        super().__init__()
        self.backbone = SegformerForSemanticSegmentation.from_pretrained(
            backbone, num_labels=2, ignore_mismatched_sizes=True)
        stage2_ch = self.backbone.config.hidden_sizes[1]  # 128 for B2
        self.aux_encoder = AuxEncoder(aux_channels, stage2_ch)
        self.se_gate     = SEGate(stage2_ch, stage2_ch)

    def forward(self, img, aux):
        # Run encoder, collect all hidden states
        enc_out = self.backbone.segformer(
            pixel_values=img, output_hidden_states=True, return_dict=True)
        hs = list(enc_out.hidden_states)  # 4 tensors, one per stage

        # Auxiliary features at Stage-2 resolution (32×32)
        f_aux = self.aux_encoder(aux)  # (B, 128, 32, 32)

        # Stage-2 hidden state — handle both spatial (B,C,H,W) and sequence (B,N,C) formats
        s2 = hs[1]
        if s2.ndim == 3:                          # (B, N, C) sequence format
            b, n, c = s2.shape
            h = w = int(n ** 0.5)
            s2_spatial = s2.reshape(b, h, w, c).permute(0, 3, 1, 2)
            reshape_back = True
        else:                                      # (B, C, H, W) spatial format
            s2_spatial = s2
            reshape_back = False

        # Apply SE gate
        s2_gated = self.se_gate(s2_spatial, f_aux)

        # Restore original format
        if reshape_back:
            b, c, h, w = s2_gated.shape
            hs[1] = s2_gated.permute(0, 2, 3, 1).reshape(b, h * w, c)
        else:
            hs[1] = s2_gated

        # Run decoder with modified hidden states
        return self.backbone.decode_head(tuple(hs))  # (B, 2, H/4, W/4)


# ── Loss, schedule, metrics (identical across all experiments) ─────────────────
class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA, dice_weight=DICE_WEIGHT, smooth=1.0):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma
        self.dice_weight = dice_weight; self.smooth = smooth

    def _focal(self, logits, targets):
        lp  = F.log_softmax(logits, dim=1)
        lpt = lp.gather(1, targets.unsqueeze(1)).squeeze(1)
        pt  = lpt.exp()
        at  = torch.where(targets==1, torch.full_like(pt,self.alpha), torch.full_like(pt,1-self.alpha))
        return (-at * (1-pt)**self.gamma * lpt).mean()

    def _dice(self, logits, targets):
        p = F.softmax(logits, dim=1)[:,1]; t = targets.float()
        return 1 - (2*(p*t).sum() + self.smooth) / (p.sum() + t.sum() + self.smooth)

    def forward(self, logits, targets):
        return (1-self.dice_weight)*self._focal(logits,targets) + self.dice_weight*self._dice(logits,targets)


def warmup_cosine_scheduler(optimizer, warmup, total):
    def f(ep):
        if ep < warmup: return (ep+1)/max(warmup,1)
        return 0.5*(1+math.cos(math.pi*(ep-warmup)/max(total-warmup,1)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def compute_metrics(preds, labels):
    tp=((preds==1)&(labels==1)).sum().item(); fp=((preds==1)&(labels==0)).sum().item()
    fn=((preds==0)&(labels==1)).sum().item(); tn=((preds==0)&(labels==0)).sum().item()
    return dict(iou=tp/(tp+fp+fn+1e-7), dice=2*tp/(2*tp+fp+fn+1e-7),
                precision=tp/(tp+fp+1e-7), recall=tp/(tp+fn+1e-7),
                pixel_acc=(tp+tn)/(tp+fp+fn+tn+1e-7))


# ── Train / val loop ───────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, scaler, train=True):
    model.train(train); total_loss = 0.0; all_preds, all_labels = [], []
    with torch.set_grad_enabled(train):
        for imgs, aux, masks, _ in loader:
            imgs, aux, masks = imgs.to(DEVICE), aux.to(DEVICE), masks.to(DEVICE)
            with autocast("cuda", enabled=(scaler is not None)):
                logits = model(imgs, aux)
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                loss   = criterion(logits, masks)
            if train:
                optimizer.zero_grad()
                if scaler is not None:
                    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
                else:
                    loss.backward(); optimizer.step()
            total_loss += loss.item() * imgs.size(0)
            all_preds.append(logits.argmax(dim=1).cpu()); all_labels.append(masks.cpu())
    m = compute_metrics(torch.cat(all_preds), torch.cat(all_labels))
    m["loss"] = total_loss / len(loader.dataset)
    return m


# ── Visualisation ──────────────────────────────────────────────────────────────
def _denorm(t): return np.clip(t.permute(1,2,0).cpu().numpy()*RGB_STD+RGB_MEAN, 0, 1)
def _overlay(img, mask, color, alpha=0.5):
    out=img.copy(); out[mask==1]=(1-alpha)*out[mask==1]+alpha*np.array(color); return np.clip(out,0,1)

def save_epoch_grid(images, gt_masks, pred_masks, ids, epoch, sample_dir, n=4):
    n=min(n,len(images)); fig,axes=plt.subplots(n,3,figsize=(9,3*n))
    if n==1: axes=axes[np.newaxis]
    for i in range(n):
        img=_denorm(images[i]); gt=gt_masks[i].cpu().numpy(); pred=pred_masks[i].cpu().numpy()
        axes[i,0].imshow(img); axes[i,0].set_title(f"Image {ids[i]}")
        axes[i,1].imshow(_overlay(img,gt,[1,0,0])); axes[i,1].set_title("GT")
        axes[i,2].imshow(_overlay(img,pred,[0,1,0])); axes[i,2].set_title("Predicted")
        for ax in axes[i]: ax.axis("off")
    plt.tight_layout(); fig.savefig(sample_dir/f"train_epoch_{epoch:03d}.png",dpi=100); plt.close(fig)

def tune_threshold(model, val_ids, logger):
    loader = DataLoader(LandslideDatasetSlope(val_ids), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    all_probs, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, aux, masks, _ in loader:
            logits = model(imgs.to(DEVICE), aux.to(DEVICE))
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            all_probs.append(F.softmax(logits,dim=1)[:,1].cpu()); all_labels.append(masks)
    probs=torch.cat(all_probs); labels=torch.cat(all_labels)
    best_t, best_iou = 0.5, 0.0
    for t in np.linspace(0.1, 0.9, 17):
        m = compute_metrics((probs >= t).long(), labels)
        if m["iou"] > best_iou: best_iou, best_t = m["iou"], float(t)
    logger.info(f"Threshold tuning: best t={best_t:.3f}  val IoU={best_iou:.4f}")
    return best_t

def save_test_overlays(model, test_ids, sample_dir, threshold, logger):
    out_dir = sample_dir/"test"; out_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(LandslideDatasetSlope(test_ids), batch_size=1, shuffle=False, num_workers=2)
    model.eval(); saved = 0
    for imgs, aux, masks, ids in loader:
        if saved >= N_TEST_SAMPLES: break
        with torch.no_grad():
            logits = model(imgs.to(DEVICE), aux.to(DEVICE))
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            pred = (F.softmax(logits,dim=1)[:,1] >= threshold).squeeze(0).cpu().numpy()
        img_np=_denorm(imgs[0]); gt_np=masks[0].numpy()
        fig,axes=plt.subplots(1,2,figsize=(7,3.5))
        axes[0].imshow(_overlay(img_np,gt_np,[1,0,0])); axes[0].set_title("GT")
        axes[1].imshow(_overlay(img_np,pred,[0,1,0])); axes[1].set_title("Predicted")
        for ax in axes: ax.axis("off")
        plt.suptitle(f"Test sample {ids[0]}  (t={threshold:.3f})", fontsize=10)
        plt.tight_layout(); fig.savefig(out_dir/f"sample_{ids[0]}.png",dpi=100); plt.close(fig); saved+=1
    logger.info(f"Saved {saved} test overlays → {out_dir}")

def plot_training_curves(log_csv, out_dir, logger):
    epochs,tr_iou,va_iou,tr_loss,va_loss=[],[],[],[],[]
    with open(log_csv) as f:
        for row in csv.DictReader(f):
            if row["phase"]=="train":
                epochs.append(int(row["epoch"])); tr_iou.append(float(row["iou"])); tr_loss.append(float(row["loss"]))
            else:
                va_iou.append(float(row["iou"])); va_loss.append(float(row["loss"]))
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10,4))
    ax1.plot(epochs,tr_iou,label="Train IoU"); ax1.plot(epochs,va_iou,label="Val IoU")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("IoU"); ax1.set_title("IoU"); ax1.legend(); ax1.grid(True)
    ax2.plot(epochs,tr_loss,label="Train Loss"); ax2.plot(epochs,va_loss,label="Val Loss")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Loss"); ax2.set_title("Loss"); ax2.legend(); ax2.grid(True)
    plt.tight_layout(); path=out_dir/"training_curves.png"; fig.savefig(path,dpi=120); plt.close(fig)
    logger.info(f"Saved training curves → {path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    EXP = "train_stage2_slope_B2"

    # Setup output dirs
    for d in ["checkpoints", "metrics", "logs", "samples"]:
        (OUT_DIR / d).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(EXP); logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); logger.addHandler(sh)
    fh = logging.FileHandler(OUT_DIR/"logs"/"training.log", mode="w"); fh.setFormatter(fmt); logger.addHandler(fh)

    logger.info("=" * 65)
    logger.info(f"Experiment  : {EXP}")
    logger.info(f"Backbone    : {BACKBONE}")
    logger.info("Aux band    : Slope  (1ch, /60°, Stage-2 SE injection)")
    logger.info(f"Loss        : Focal(α={FOCAL_ALPHA}, γ={FOCAL_GAMMA}) + Dice(λ={DICE_WEIGHT})")
    logger.info(f"Device      : {DEVICE}")
    logger.info("=" * 65)

    train_ids, val_ids, test_ids = get_splits()
    logger.info(f"Split  train={len(train_ids)}  val={len(val_ids)}  test={len(test_ids)}")

    train_loader = DataLoader(LandslideDatasetSlope(train_ids, augment=True),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(LandslideDatasetSlope(val_ids),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    logger.info(f"Building MultiModalStage2 ({BACKBONE}, Slope) …")
    model = MultiModalStage2().to(DEVICE)
    n_rgb = sum(p.numel() for p in model.backbone.parameters() if p.requires_grad)
    n_aux = sum(p.numel() for p in model.aux_encoder.parameters()) + sum(p.numel() for p in model.se_gate.parameters())
    logger.info(f"Params  RGB={n_rgb/1e6:.1f}M  AuxEncoder+SEGate={n_aux/1e3:.0f}K  Total={(n_rgb+n_aux)/1e6:.1f}M")

    criterion = FocalDiceLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = warmup_cosine_scheduler(optimizer, WARMUP_EPOCHS, EPOCHS)
    scaler    = GradScaler("cuda") if DEVICE == "cuda" else None

    log_csv    = OUT_DIR/"metrics"/"train_log.csv"
    fieldnames = ["epoch","phase","loss","iou","dice","precision","recall","pixel_acc"]
    with open(log_csv, "w", newline="") as f: csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    best_iou=0.0; no_improve=0; vis_batch=next(iter(val_loader))
    ckpt_path=OUT_DIR/"checkpoints"/"best_model.pth"; sample_dir=OUT_DIR/"samples"

    for epoch in range(1, EPOCHS + 1):
        tr = run_epoch(model, train_loader, criterion, optimizer, scaler, train=True)
        va = run_epoch(model, val_loader,   criterion, optimizer, scaler, train=False)
        scheduler.step()
        logger.info(f"Epoch {epoch:03d}  lr={scheduler.get_last_lr()[0]:.2e}  "
                    f"train loss={tr['loss']:.4f} iou={tr['iou']:.4f}  |  "
                    f"val   loss={va['loss']:.4f} iou={va['iou']:.4f} "
                    f"dice={va['dice']:.4f} prec={va['precision']:.4f} rec={va['recall']:.4f}")
        with open(log_csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            for phase, m in [("train",tr),("val",va)]:
                writer.writerow({"epoch":epoch,"phase":phase,**{k:f"{v:.6f}" for k,v in m.items()}})
        if va["iou"] > best_iou:
            best_iou=va["iou"]; no_improve=0
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"  ✓ Best model saved  (val IoU={best_iou:.4f})")
        else:
            no_improve += 1
            if no_improve >= EARLY_STOP_PAT:
                logger.info(f"  Early stopping at epoch {epoch}"); break
        if epoch % SAVE_EVERY == 0:
            imgs, aux, masks, ids = vis_batch
            with torch.no_grad():
                logits = model(imgs.to(DEVICE), aux.to(DEVICE))
                logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
                preds  = logits.argmax(dim=1).cpu()
            save_epoch_grid(imgs, masks, preds, ids, epoch, sample_dir)

    logger.info("\n── Post-training evaluation & visualisation ──")
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

    test_loader = DataLoader(LandslideDatasetSlope(test_ids), batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS)
    te = run_epoch(model, test_loader, criterion, optimizer, scaler, train=False)
    logger.info(f"Test (t=0.5)  iou={te['iou']:.4f} dice={te['dice']:.4f} "
                f"prec={te['precision']:.4f} rec={te['recall']:.4f} acc={te['pixel_acc']:.4f}")
    with open(OUT_DIR/"metrics"/"test_metrics.json","w") as f:
        json.dump({k:round(v,6) for k,v in te.items()}, f, indent=2)

    best_t = tune_threshold(model, val_ids, logger)
    with open(OUT_DIR/"metrics"/"best_threshold.json","w") as f:
        json.dump({"threshold":round(best_t,4)}, f, indent=2)

    all_probs, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for imgs, aux, masks, _ in test_loader:
            logits = model(imgs.to(DEVICE), aux.to(DEVICE))
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            all_probs.append(F.softmax(logits,dim=1)[:,1].cpu()); all_labels.append(masks)
    te_tuned = compute_metrics((torch.cat(all_probs)>=best_t).long(), torch.cat(all_labels))
    logger.info(f"Test (t={best_t:.3f}) iou={te_tuned['iou']:.4f} dice={te_tuned['dice']:.4f} "
                f"prec={te_tuned['precision']:.4f} rec={te_tuned['recall']:.4f} acc={te_tuned['pixel_acc']:.4f}")
    with open(OUT_DIR/"metrics"/"test_metrics_tuned.json","w") as f:
        json.dump({"threshold":round(best_t,4),**{k:round(v,6) for k,v in te_tuned.items()}}, f, indent=2)

    save_test_overlays(model, test_ids, sample_dir, best_t, logger)
    plot_training_curves(log_csv, OUT_DIR/"metrics", logger)
    logger.info(f"Done. All outputs saved to output/{OUT_DIR.name}/")


if __name__ == "__main__":
    main()

