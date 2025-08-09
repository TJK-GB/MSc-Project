import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torchvision import transforms


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# === CONFIG ===
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
EPOCHS = 10
MAX_FRAMES = 80

# === PATHS ===
CSV_DIR = r"D:\\UK\\00. 2024 QMUL\\00. Course\\Project\\00. ViolenceDetectionProject\\DATASET\\00. Actual Dataset\\Data split\\Unimodal"
NPY_DIR = r"D:\\UK\\00. 2024 QMUL\\00. Course\\Project\\00. ViolenceDetectionProject\\DATASET\\00. Actual Dataset\\npy_segments_unimodal"
SAVE_PATH = r"D:\UK\00. 2024 QMUL\00. Course\Project\00. ViolenceDetectionProject\Results\with BCE\Again(CUE-NET)"
train_csv_path = os.path.join(CSV_DIR, "train.csv")
test_csv_path = os.path.join(CSV_DIR, "test.csv")

class ViolenceDataset(Dataset):
    def __init__(self, csv_path, npy_dir):
        self.df = pd.read_csv(csv_path)
        self.npy_dir = npy_dir
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        segment_id = row['Segment ID']
        label = row['Violence label(video)']
        frames = np.load(os.path.join(self.npy_dir, f"{segment_id}.npy"))[:MAX_FRAMES]
        frames = torch.stack([
            self.transform(torch.from_numpy(f).permute(2, 0, 1).float().div(255.0)) for f in frames
        ])
        return frames, torch.tensor(label, dtype=torch.float32)

class CueNet(nn.Module):
    def __init__(self):
        super().__init__()

        # === Local Branch: ResNet50 ===
        base = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.local_cnn = nn.Sequential(*list(base.children())[:-1])
        self.local_out_dim = 2048

        # === Global Branch: GRU ===
        self.temporal_gru = nn.GRU(
            input_size=self.local_out_dim,
            hidden_size=256,
            batch_first=True,
            bidirectional=True
        )
        self.global_out_dim = 512

        # === Project to common dimension (e.g., 512)
        self.local_proj = nn.Linear(self.local_out_dim, 512)
        self.global_proj = nn.Identity()  # GRU already gives 512

        # === Gating
        self.gate = nn.Sequential(
            nn.Linear(1024, 2),
            nn.Softmax(dim=1)
        )

        # === Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(1024, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):  # x: (B, T, C, H, W)
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        feats = self.local_cnn(x).view(B, T, -1)  # (B, T, 2048)

        local_feat = feats.mean(dim=1)            # (B, 2048)
        global_feat_seq, _ = self.temporal_gru(feats)
        global_feat = global_feat_seq.mean(dim=1) # (B, 512)

        # Project to same dim
        local_proj = self.local_proj(local_feat)  # (B, 512)
        global_proj = self.global_proj(global_feat)  # (B, 512)

        # Gate
        combined = torch.cat([local_proj, global_proj], dim=1)  # (B, 1024)
        gates = self.gate(combined)  # (B, 2)
        g1, g2 = gates[:, 0].unsqueeze(1), gates[:, 1].unsqueeze(1)

        fused_feat = g1 * local_proj + g2 * global_proj  # (B, 512)
        out = self.classifier(torch.cat([fused_feat, fused_feat], dim=1)).squeeze(1)
        return out

train_dataset = ViolenceDataset(train_csv_path, NPY_DIR)
test_dataset = ViolenceDataset(test_csv_path, NPY_DIR)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CueNet().to(device)

criterion = nn.BCEWithLogitsLoss()
optimiser = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='min', factor=0.5, patience=1)
scaler = GradScaler()
best_loss = float('inf')
early_stop_counter = 0
EARLY_STOPPING_PATIENCE = 4

for epoch in range(EPOCHS):
    model.train()
    y_true, y_pred = [], []
    total_loss = 0.0
    optimiser.zero_grad()

    for i, (frames, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
        frames, labels = frames.to(device), labels.to(device)

        with autocast(device_type='cuda'):
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss = loss / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        if (i + 1) % GRAD_ACCUM_STEPS == 0 or (i + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad()

        total_loss += loss.item() * GRAD_ACCUM_STEPS
        preds = (torch.sigmoid(outputs) > 0.5).int()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Macro F1: {f1_macro:.4f} | Micro F1: {f1_micro:.4f}")

    scheduler.step(total_loss)
    if total_loss < best_loss:
        best_loss = total_loss
        torch.save(model.state_dict(), os.path.join(SAVE_PATH, "cuenet_best.pt"))
        print("[INFO] Best model saved.")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= EARLY_STOPPING_PATIENCE:
            print("[INFO] Early stopping triggered.")
            break

model.eval()
y_true, y_pred, segment_ids = [], [], test_dataset.df['Segment ID'].tolist()

test_total_loss = 0.0
with torch.no_grad():
    for frames, labels in test_loader:
        frames, labels = frames.to(device), labels.to(device)
        outputs = model(frames)
        loss = criterion(outputs, labels)
        test_total_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).int()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

avg_test_loss = test_total_loss / len(test_loader)

print("\n[TEST] Classification Report:")
print(f"[TEST] BCE Loss: {avg_test_loss:.4f}")
report = classification_report(y_true, y_pred, target_names=["Non-violent", "Violent"], output_dict=True)
print(classification_report(y_true, y_pred, target_names=["Non-violent", "Violent"]))

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

results = pd.DataFrame({"Segment ID": segment_ids, "True": y_true, "Pred": y_pred})
results.to_csv(os.path.join(SAVE_PATH, "CUE-NET_test_predictions_final.csv"), index=False)
pd.DataFrame(report).to_csv(os.path.join(SAVE_PATH, "CUE-NET_test_metrics_final.csv"))
with open(os.path.join(SAVE_PATH, "CUE-NET_test_bce_loss.txt"), "w") as f:
    f.write(f"BCE Loss: {avg_test_loss:.4f}\n")
print("[INFO] Predictions and metrics saved.")
