# cue_net_with_source.py
# ResNet50 (frame encoder) + temporal conv + attention pooling + video-source conditioning (one-hot 0..6)

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score, classification_report, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms
from torch.amp import autocast, GradScaler

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")

# ===================== #
#       CONFIG          #
# ===================== #
BATCH_SIZE = 4
EPOCHS = 10
MAX_FRAMES = 80
LR = 1e-4
THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- EDIT THESE PATHS ----
CSV_DIR = r"D:\UK\00. 2024 QMUL\00. Course\Project\00. ViolenceDetectionProject\DATASET\Data split\Unimodal_with_videosource"
NPY_DIR = r"D:\UK\00. 2024 QMUL\00. Course\Project\00. ViolenceDetectionProject\DATASET\00. Actual Dataset\npy_segments_videosource"
SAVE_DIR = r"D:\UK\00. 2024 QMUL\00. Course\Project\00. ViolenceDetectionProject\Results\video_source\CUENet"
ckpt_path = os.path.join(SAVE_DIR, "checkpoint.pth")
# --------------------------
TRAIN_CSV = os.path.join(CSV_DIR, "train.csv")
TEST_CSV  = os.path.join(CSV_DIR, "test.csv")
os.makedirs(SAVE_DIR, exist_ok=True)

# ===================== #
#       DATASET         #
# ===================== #
class ViolenceDataset(Dataset):
    """
    Returns
      frames_t: [MAX_FRAMES, 3, 224, 224] float tensor (normalized)
      src_oh:   [7] one-hot source vector
      label:    scalar float tensor (0/1)
      seg_id:   string
    """
    def __init__(self, csv_path: str, npy_dir: str, max_frames: int = 80):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.npy_dir = npy_dir
        self.max_frames = max_frames

        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _to_chw_float(frame_np: np.ndarray) -> torch.Tensor:
        if frame_np.ndim != 3:
            raise ValueError(f"Frame must be 3D, got {frame_np.shape}")
        if frame_np.shape[0] == 3:  # CHW
            t = torch.from_numpy(frame_np).float()
        else:                       # HWC -> CHW
            t = torch.from_numpy(frame_np).permute(2, 0, 1).float()
        if t.max() > 1.5:
            t = t / 255.0
        return t

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        seg_id = row["Segment ID"]
        label = torch.tensor(row["Violence label(video)"], dtype=torch.float32)

        src_idx = int(row["Video Source Label"])
        src_oh = torch.zeros(7, dtype=torch.float32)
        if 0 <= src_idx < 7:
            src_oh[src_idx] = 1.0

        npy_path = os.path.join(self.npy_dir, seg_id + ".npy")
        frames = np.load(npy_path)
        if frames.ndim != 4:
            raise ValueError(f"{seg_id}: expected 4D array, got {frames.shape}")

        T = frames.shape[0]
        if T == 0:
            raise ValueError(f"{seg_id}: zero frames")

        if T >= self.max_frames:
            idxs = np.linspace(0, T - 1, num=self.max_frames, dtype=int)
        else:
            pad = [T - 1] * (self.max_frames - T)
            idxs = np.concatenate([np.arange(T, dtype=int), np.array(pad, dtype=int)], axis=0)
        frames = frames[idxs]

        tensor_frames = []
        for f in frames:
            t = self._to_chw_float(f)
            t = self.tf(t)
            tensor_frames.append(t)
        frames_t = torch.stack(tensor_frames, dim=0)  # [MAX_FRAMES, 3, 224, 224]

        return frames_t, src_oh, label, seg_id

# ===================== #
#        MODEL          #
# ===================== #
class CUENetWithSource(nn.Module):
    """
    ResNet50 backbone -> per-frame 2048-d embeddings
    Temporal conv + attention pooling to get video-level feature
    Concatenate source one-hot -> classifier to binary logit
    Freeze layers except layer4 for stability on small data
    """
    def __init__(self, num_sources: int = 7, t_hidden: int = 512, dropout: float = 0.3):
        super().__init__()
        # Backbone
        self.backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone.fc = nn.Identity()  # output 2048
        self.embed_dim = 2048

        # Freeze all, unfreeze last stage
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone.layer4.parameters():
            p.requires_grad = True

        for m in self.backbone.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                for p in m.parameters():
                    p.requires_grad = False

        # Temporal conv over features [B, T, C] -> [B, C, T]
        self.temporal = nn.Sequential(
            nn.Conv1d(self.embed_dim, t_hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(t_hidden, t_hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Attention pooling over time
        self.attn_score = nn.Sequential(
            nn.Conv1d(t_hidden, t_hidden // 2, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(t_hidden // 2, 1, kernel_size=1)  # [B, 1, T]
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(t_hidden + num_sources, 1)

    def forward(self, x: torch.Tensor, src_oh: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, 3, 224, 224]
        src_oh: [B, num_sources]
        return logits: [B]
        """
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W).contiguous()
        feats = self.backbone(x)                 # [B*T, 2048]
        feats = feats.view(B, T, self.embed_dim) # [B, T, 2048]

        feats = feats.permute(0, 2, 1).contiguous()  # [B, 2048, T]
        h = self.temporal(feats)                     # [B, t_hidden, T]

        # Attention weights over time
        scores = self.attn_score(h)                  # [B, 1, T]
        alpha = torch.softmax(scores, dim=-1)        # [B, 1, T]
        vid_feat = torch.sum(h * alpha, dim=-1)      # [B, t_hidden]

        fused = torch.cat([vid_feat, src_oh], dim=1) # [B, t_hidden + num_sources]
        fused = self.dropout(fused)
        logits = self.fc(fused).squeeze(1)           # [B]
        return logits

# ===================== #
#     TRAIN / EVAL      #
# ===================== #
def train_one_epoch(model, loader, device, criterion, optimizer, threshold, use_amp, scaler):
    model.train()
    total_loss, y_true, y_pred = 0.0, [], []

    for frames, src_oh, labels, _ in tqdm(loader, desc="Train", leave=False):
        frames = frames.to(device, non_blocking=True)
        src_oh  = src_oh.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(frames, src_oh)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > threshold).int()
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(preds.detach().cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(loader))
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    return avg_loss, macro_f1, micro_f1

@torch.no_grad()
def evaluate(model, loader, device, criterion, threshold, use_amp):
    model.eval()
    total_loss, y_true, y_pred, seg_ids = 0.0, [], [], []

    for frames, src_oh, labels, seg in tqdm(loader, desc="Test", leave=False):
        frames = frames.to(device, non_blocking=True)
        src_oh  = src_oh.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)

        with autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(frames, src_oh)
            loss   = criterion(logits, labels)

        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > threshold).int().cpu().numpy()
        y_true.extend(labels.detach().cpu().numpy().tolist())
        y_pred.extend(preds.tolist())
        seg_ids.extend(seg)

    avg_loss = total_loss / max(1, len(loader))
    report   = classification_report(y_true, y_pred, target_names=["Non-violent","Violent"],
                                     output_dict=True, zero_division=0)
    cm       = confusion_matrix(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    return avg_loss, macro_f1, micro_f1, report, cm, y_true, y_pred, seg_ids

# ===================== #
#         MAIN          #
# ===================== #
def main():
    print("[INFO] Training CUE-Net style model with video-source conditioning...")

    train_ds = ViolenceDataset(TRAIN_CSV, NPY_DIR, max_frames=MAX_FRAMES)
    test_ds  = ViolenceDataset(TEST_CSV,  NPY_DIR, max_frames=MAX_FRAMES)
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )
    test_loader  = DataLoader(
        test_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2
    )

    model = CUENetWithSource(num_sources=7, t_hidden=512, dropout=0.3).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    use_amp = (DEVICE.type == "cuda")
    scaler  = GradScaler() if use_amp else GradScaler(enabled=False)

    best_loss = float("inf")
    for epoch in range(EPOCHS):
        tr_loss, tr_f1_macro, tr_f1_micro = train_one_epoch(
            model, train_loader, DEVICE, criterion, optimizer, THRESHOLD, use_amp, scaler
        )
        print(f"Epoch {epoch+1}/{EPOCHS} | Train BCE: {tr_loss:.4f} | Macro F1: {tr_f1_macro:.4f} | Micro F1: {tr_f1_micro:.4f}")

        if tr_loss < best_loss:
            best_loss = tr_loss
            torch.save(model.state_dict(), ckpt_path)
            print("  [SAVE] Best checkpoint updated.")

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    test_loss, test_f1_macro, test_f1_micro, report, cm, y_true, y_pred, seg_ids = evaluate(
        model, test_loader, DEVICE, criterion, THRESHOLD, use_amp
    )

    print("\n[TEST] BCE Loss:", round(test_loss, 4))
    print("[TEST] Macro F1:", round(test_f1_macro, 4))
    print("[TEST] Micro F1:", round(test_f1_micro, 4))
    print("[TEST] Per-Class F1 Scores:")
    print(" - Non-violent F1:", round(report['Non-violent']['f1-score'], 4))
    print(" - Violent F1:", round(report['Violent']['f1-score'], 4))
    print("Confusion Matrix:\n", cm)

    pd.DataFrame({
        "Segment ID": seg_ids,
        "True": y_true,
        "Pred": y_pred
    }).to_csv(os.path.join(SAVE_DIR, "cue_net_with_source_test_predictions.csv"), index=False)

    pd.DataFrame(report).to_csv(os.path.join(SAVE_DIR, "cue_net_with_source_test_metrics.csv"))

    pd.DataFrame(cm, index=["True_Non-violent", "True_Violent"],
                 columns=["Pred_Non-violent", "Pred_Violent"]).to_csv(
        os.path.join(SAVE_DIR, "cue_net_with_source_confusion_matrix.csv")
    )

    with open(os.path.join(SAVE_DIR, "cue_net_with_source_test_bce_loss.txt"), "w") as f:
        f.write(f"BCE Loss: {test_loss:.4f}\n")
        f.write(f"Macro F1: {test_f1_macro:.4f}\n")
        f.write(f"Micro F1: {test_f1_micro:.4f}\n")
        f.write(f"Non-violent F1: {report['Non-violent']['f1-score']:.4f}\n")
        f.write(f"Violent F1: {report['Violent']['f1-score']:.4f}\n")

    print("[INFO] Results saved to:", SAVE_DIR)

if __name__ == "__main__":
    main()
