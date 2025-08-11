# resnet50_gru_with_source.py
# ResNet50 (frame encoder) + Bi-GRU (temporal) + video-source conditioning (one-hot 0..6)

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

# ===================== #
#        CONFIG         #
# ===================== #
BATCH_SIZE = 4
EPOCHS = 20
MAX_FRAMES = 80
LR = 1e-4
THRESHOLD = 0.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ---- EDIT THESE PATHS ----
CSV_DIR = r"D:\UK\00. 2024 QMUL\00. Course\Project\00. ViolenceDetectionProject\DATASET\Data split\Unimodal_with_videosource"
NPY_DIR = r"D:\UK\00. 2024 QMUL\00. Course\Project\00. ViolenceDetectionProject\DATASET\00. Actual Dataset\npy_segments_videosource"
SAVE_DIR = r"D:\UK\00. 2024 QMUL\00. Course\Project\00. ViolenceDetectionProject\Results\video_source\ResNet50_GRU_with_source"
# --------------------------
TRAIN_CSV = os.path.join(CSV_DIR, "train.csv")
TEST_CSV  = os.path.join(CSV_DIR, "test.csv")
os.makedirs(SAVE_DIR, exist_ok=True)


# ===================== #
#        DATASET        #
# ===================== #
class ViolenceDataset(Dataset):
    """
    Yields:
      frames_t: [MAX_FRAMES, 3, 224, 224]  (float, normalized)
      src_oh:   [7] one-hot video source
      label:    scalar float (0/1)
      seg_id:   string
    """
    def __init__(self, csv_path: str, npy_dir: str, max_frames: int = 80):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.npy_dir = npy_dir
        self.max_frames = max_frames
        self.tf = transforms.Compose([
            transforms.ToPILImage(),                         # accepts CHW [0..1] or HWC uint8
            transforms.Resize((224, 224)),
            transforms.ToTensor(),                           # -> CHW [0..1]
            transforms.Normalize(mean=(0.485, 0.456, 0.406), # ImageNet stats (ResNet pretrain)
                                 std=(0.229, 0.224, 0.225)),
        ])

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _to_chw_float(frame_np: np.ndarray) -> torch.Tensor:
        """Accept HWC or CHW; return CHW float in [0,1]."""
        if frame_np.ndim != 3:
            raise ValueError(f"Frame must be 3D, got {frame_np.shape}")
        if frame_np.shape[0] == 3:  # CHW
            t = torch.from_numpy(frame_np).float()
        else:                       # HWC -> CHW
            t = torch.from_numpy(frame_np).permute(2, 0, 1).float()
        if t.max() > 1.5:           # likely uint8 0..255
            t = t / 255.0
        return t

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        seg_id = row["Segment ID"]
        label = torch.tensor(row["Violence label(video)"], dtype=torch.float32)

        # One-hot video source (0..6)
        src_idx = int(row["Video Source Label"])
        src_oh = torch.zeros(7, dtype=torch.float32)
        if 0 <= src_idx < 7:
            src_oh[src_idx] = 1.0

        # Load frames from .npy (T,H,W,C) or (T,C,H,W)
        npy_path = os.path.join(self.npy_dir, seg_id + ".npy")
        frames = np.load(npy_path)
        if frames.ndim != 4:
            raise ValueError(f"{seg_id}: expected 4D array, got {frames.shape}")

        T = frames.shape[0]
        if T == 0:
            raise ValueError(f"{seg_id}: zero frames")

        # Uniform sample/pad to exactly MAX_FRAMES
        if T >= self.max_frames:
            idxs = np.linspace(0, T - 1, num=self.max_frames, dtype=int)
        else:
            pad = [T - 1] * (self.max_frames - T)
            idxs = np.concatenate([np.arange(T, dtype=int), np.array(pad, dtype=int)], axis=0)
        frames = frames[idxs]  # (MAX_FRAMES, ...)

        # Transform per-frame -> CHW 224x224 normalized
        tensor_frames = []
        for f in frames:
            t = self._to_chw_float(f)
            t = self.tf(t)
            tensor_frames.append(t)
        frames_t = torch.stack(tensor_frames, dim=0)  # [MAX_FRAMES, 3, 224, 224]

        return frames_t, src_oh, label, seg_id


# ===================== #
#         MODEL         #
# ===================== #
class ResNet50_GRU_WithSource(nn.Module):
    """
    Frame encoder: ResNet50 (avgpool features 2048)
    Temporal encoder: Bi-GRU -> 512
    Conditioning: concat 7-dim source one-hot to GRU pooled feature
    Classifier: MLP -> 1 logit
    """
    def __init__(self, hidden: int = 256, dropout: float = 0.3, unfreeze_layer4: bool = True):
        super().__init__()
        base = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Freeze everything first
        for p in base.parameters():
            p.requires_grad = False
        # Unfreeze last stage for fine-tuning if requested
        if unfreeze_layer4:
            for p in base.layer4.parameters():
                p.requires_grad = True

        # Remove classifier head -> outputs [B, 2048, 1, 1]
        self.cnn = nn.Sequential(*list(base.children())[:-1])

        # Temporal encoder
        self.gru = nn.GRU(input_size=2048, hidden_size=hidden, batch_first=True, bidirectional=True)

        # Classifier Head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2 + 7, 128),  # 512 + 7
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )

    def forward(self, frames: torch.Tensor, src_oh: torch.Tensor) -> torch.Tensor:
        """
        frames: [B, T, 3, 224, 224]
        src_oh: [B, 7]
        returns logits: [B]
        """
        B, T, C, H, W = frames.shape
        x = frames.view(B * T, C, H, W)

        # CNN features (keep grads for layer4 if unfrozen)
        feats = self.cnn(x)                 # [B*T, 2048, 1, 1]
        feats = feats.view(B, T, 2048)      # [B, T, 2048]

        # Bi-GRU over time
        out, _ = self.gru(feats)            # [B, T, 512]
        vid_feat = out.mean(dim=1)          # [B, 512] (temporal mean)

        # Concatenate source one-hot
        fused = torch.cat([vid_feat, src_oh], dim=1)  # [B, 519]
        fused = self.dropout(fused)
        logits = self.classifier(fused).squeeze(1)    # [B]
        return logits


# ===================== #
#     TRAIN / EVAL      #
# ===================== #
def train_one_epoch(model, loader, device, criterion, optimizer, threshold):
    model.train()
    total_loss = 0.0
    y_true, y_pred = [], []

    for frames, src_oh, labels, _ in tqdm(loader, desc="Train", leave=False):
        frames = frames.to(device, non_blocking=True)
        src_oh = src_oh.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(frames, src_oh)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > threshold).int()
        y_true.extend(labels.cpu().numpy().tolist())
        y_pred.extend(preds.cpu().numpy().tolist())

    avg_loss = total_loss / max(1, len(loader))
    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
    return avg_loss, macro_f1, micro_f1


@torch.no_grad()
def evaluate(model, loader, device, criterion, threshold):
    model.eval()
    total_loss = 0.0
    y_true, y_pred, seg_ids = [], [], []

    for frames, src_oh, labels, seg in tqdm(loader, desc="Test", leave=False):
        frames = frames.to(device, non_blocking=True)
        src_oh = src_oh.to(device, non_blocking=True)
        logits = model(frames, src_oh)
        loss = criterion(logits, labels.to(device))
        total_loss += loss.item()

        preds = (torch.sigmoid(logits) > threshold).int().cpu().numpy()
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(preds.tolist())
        seg_ids.extend(seg)

    avg_loss = total_loss / max(1, len(loader))
    report = classification_report(
        y_true, y_pred,
        target_names=["Non-violent", "Violent"],
        output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)

    return avg_loss, macro_f1, micro_f1, report, cm, y_true, y_pred, seg_ids


# ===================== #
#         MAIN          #
# ===================== #
def main():
    print("[INFO] Training ResNet50+GRU with video-source conditioning...")

    # Data
    train_ds = ViolenceDataset(TRAIN_CSV, NPY_DIR, max_frames=MAX_FRAMES)
    test_ds  = ViolenceDataset(TEST_CSV,  NPY_DIR, max_frames=MAX_FRAMES)

    # num_workers=0 to avoid Windows multiprocessing issues
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    # Model & optim
    model = ResNet50_GRU_WithSource(hidden=256, dropout=0.3, unfreeze_layer4=True).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    best_loss = float("inf")
    ckpt_path = os.path.join(SAVE_DIR, "resnet50_gru_with_source_best.pt")

    # Train
    for epoch in range(EPOCHS):
        tr_loss, tr_f1_macro, tr_f1_micro = train_one_epoch(
            model, train_loader, DEVICE, criterion, optimizer, THRESHOLD
        )
        print(f"Epoch {epoch+1}/{EPOCHS} | Train BCE: {tr_loss:.4f} | Macro F1: {tr_f1_macro:.4f} | Micro F1: {tr_f1_micro:.4f}")

        if tr_loss < best_loss:
            best_loss = tr_loss
            torch.save(model.state_dict(), ckpt_path)
            print("  [SAVE] Best checkpoint updated.")

    # Load best and evaluate on test set
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    test_loss, test_f1_macro, test_f1_micro, report, cm, y_true, y_pred, seg_ids = evaluate(
        model, test_loader, DEVICE, criterion, THRESHOLD
    )

    # Pretty print (your requested format)
    print("\n[TEST] BCE Loss:", round(test_loss, 4))
    print("[TEST] Macro F1:", round(test_f1_macro, 4))
    print("[TEST] Micro F1:", round(test_f1_micro, 4))
    print("[TEST] Per-Class F1 Scores:")
    print(" - Non-violent F1:", round(report['Non-violent']['f1-score'], 4))
    print(" - Violent F1:", round(report['Violent']['f1-score'], 4))
    print("Confusion Matrix:\n", cm)

    # Save artifacts
    pd.DataFrame({
        "Segment ID": seg_ids,
        "True": y_true,
        "Pred": y_pred
    }).to_csv(os.path.join(SAVE_DIR, "resnet50_gru_with_source_test_predictions.csv"), index=False)

    pd.DataFrame(report).to_csv(os.path.join(SAVE_DIR, "resnet50_gru_with_source_test_metrics.csv"))

    pd.DataFrame(cm, index=["True_Non-violent", "True_Violent"],
                    columns=["Pred_Non-violent", "Pred_Violent"]).to_csv(
        os.path.join(SAVE_DIR, "resnet50_gru_with_source_confusion_matrix.csv")
    )

    with open(os.path.join(SAVE_DIR, "resnet50_gru_with_source_test_bce_loss.txt"), "w") as f:
        f.write(f"BCE Loss: {test_loss:.4f}\n")
        f.write(f"Macro F1: {test_f1_macro:.4f}\n")
        f.write(f"Micro F1: {test_f1_micro:.4f}\n")
        f.write(f"Non-violent F1: {report['Non-violent']['f1-score']:.4f}\n")
        f.write(f"Violent F1: {report['Violent']['f1-score']:.4f}\n")

    print("[INFO] Results saved to:", SAVE_DIR)


if __name__ == "__main__":
    main()
