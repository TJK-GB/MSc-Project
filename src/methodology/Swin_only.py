# === Paths ===
train_csv_path = r"D:\UK\00. 2024 QMUL\00. Course\Project\00. ViolenceDetectionProject\DATASET\00. Actual Dataset\Data split\Unimodal\train.csv"
test_csv_path = r"D:\UK\00. 2024 QMUL\00. Course\Project\00. ViolenceDetectionProject\DATASET\00. Actual Dataset\Data split\Unimodal\test.csv"
NPY_DIR = r"D:\UK\00. 2024 QMUL\00. Course\Project\00. ViolenceDetectionProject\DATASET\00. Actual Dataset\npy_segments_unimodal"
save_path = r"D:\UK\00. 2024 QMUL\00. Course\Project\00. ViolenceDetectionProject\Results\with BCE\Again(Swin Only)"

# === CONFIG ===
BATCH_SIZE = 4
MAX_FRAMES = 80
EPOCHS = 10
USE_WEIGHTED_LOSS = True


import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import swin_t, Swin_T_Weights
from torchvision.transforms import Resize
from torch.amp import autocast, GradScaler


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# === MODEL ===
class SwinVideoClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.swin = swin_t(weights=Swin_T_Weights.DEFAULT)
        self.swin.head = nn.Identity()
        for i in [0, 1]:
            for param in self.swin.features[i].parameters():
                param.requires_grad = False
        self.fc = nn.Linear(768, 1)

    def forward(self, x):  # [B, T, C, H, W]
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        features = self.swin(x).view(B, T, -1)  # [B, T, 768]
        pooled = features.mean(dim=1)          # [B, 768]
        out = self.fc(pooled)                  # [B, 1]
        return out.squeeze(1)

# === DATASET ===
class ViolenceDataset(Dataset):
    def __init__(self, csv_path, npy_dir):
        self.df = pd.read_csv(csv_path)
        self.npy_dir = npy_dir
        self.resize = Resize((224, 224))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        segment_id = row['Segment ID']
        label = row['Violence label(video)']
        frames = np.load(os.path.join(self.npy_dir, f"{segment_id}.npy"))[:MAX_FRAMES]
        frames = torch.stack([
            self.resize(torch.from_numpy(f).permute(2, 0, 1).float() / 255.0)
            for f in frames
        ])
        return frames, torch.tensor(label, dtype=torch.float32)

# === INIT ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset = ViolenceDataset(train_csv_path, NPY_DIR)
test_dataset = ViolenceDataset(test_csv_path, NPY_DIR)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
model = SwinVideoClassifier().to(device)

# Weighted BCE loss
pos = train_dataset.df['Violence label(video)'].sum()
neg = len(train_dataset) - pos
ratio = neg / pos
print(f"[INFO] Non-violent: {neg}, Violent: {pos}, Ratio: {ratio:.2f}")
criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([ratio]).to(device)) if USE_WEIGHTED_LOSS else nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=1, factor=0.5)
scaler = GradScaler()
best_f1, early_stop_counter = 0, 0
PATIENCE = 4
loss_log = []

# === TRAINING ===
for epoch in range(EPOCHS):
    model.train()
    y_true, y_pred, total_loss = [], [], 0

    for frames, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        frames, labels = frames.to(device), labels.to(device)
        with autocast(device_type='cuda'):
            outputs = model(frames)
            loss = criterion(outputs, labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).int()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

    macro_f1 = f1_score(y_true, y_pred, average='macro')
    micro_f1 = f1_score(y_true, y_pred, average='micro')

    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f} | Macro F1: {macro_f1:.4f} | Micro F1: {micro_f1:.4f}")
    loss_log.append({'Epoch': epoch+1, 'Train_BCE_Loss': total_loss, 'Train_Macro_F1': macro_f1, 'Train_Micro_F1': micro_f1})
    scheduler.step(macro_f1)

    if macro_f1 > best_f1:
        best_f1 = macro_f1
        torch.save(model.state_dict(), os.path.join(save_path, "swin_best.pt"))
        print("[INFO] Best model saved.")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= PATIENCE:
            print("[INFO] Early stopping.")
            break

# === TESTING ===
model.eval()
y_true, y_pred, test_loss = [], [], 0
segment_ids = test_dataset.df['Segment ID'].tolist()

with torch.no_grad():
    for frames, labels in test_loader:
        frames, labels = frames.to(device), labels.to(device)
        outputs = model(frames)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).int()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

# === METRICS ===
macro_f1_test = f1_score(y_true, y_pred, average='macro')
micro_f1_test = f1_score(y_true, y_pred, average='micro')
report = classification_report(y_true, y_pred, target_names=["Non-violent", "Violent"], output_dict=True, zero_division=0)
conf_matrix = confusion_matrix(y_true, y_pred)

print("\n[TEST] BCE Loss:", round(test_loss, 4))
print("[TEST] Macro F1:", round(macro_f1_test, 4))
print("[TEST] Micro F1:", round(micro_f1_test, 4))
print("[TEST] Per-Class F1 Scores:")
print(" - Non-violent F1:", round(report['Non-violent']['f1-score'], 4))
print(" - Violent F1:", round(report['Violent']['f1-score'], 4))
print("Confusion Matrix:\n", conf_matrix)

# === SAVE RESULTS ===
results_df = pd.DataFrame({"Segment ID": segment_ids, "True": y_true, "Pred": y_pred})
results_df.to_csv(os.path.join(save_path, "swin_test_predictions_final.csv"), index=False)
pd.DataFrame(report).to_csv(os.path.join(save_path, "swin_test_metrics_final.csv"))
pd.DataFrame(loss_log).to_csv(os.path.join(save_path, "swin_train_loss_log.csv"), index=False)
print("[INFO] All results saved.")