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

# === CONFIG ===
BATCH_SIZE = 4
GRAD_ACCUM_STEPS = 4
EPOCHS = 20
MAX_FRAMES = 80  
# === PATHS ===
CSV_DIR = r"D:\\UK\\00. 2024 QMUL\\00. Course\\Project\\00. ViolenceDetectionProject\\DATASET\\00. Actual Dataset\\Data split\\Unimodal"
NPY_DIR = r"D:\\UK\\00. 2024 QMUL\\00. Course\\Project\\00. ViolenceDetectionProject\\DATASET\\00. Actual Dataset\\npy_segments_unimodal"
train_csv_path = os.path.join(CSV_DIR, "train.csv")
test_csv_path = os.path.join(CSV_DIR, "test.csv")

# === MODEL ===
class ResNet50GRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)

        # Freeze all layers first
        for p in self.resnet.parameters():
            p.requires_grad = False
        # Unfreeze only layer4
        for p in self.resnet.layer4.parameters():
            p.requires_grad = True

        self.resnet.fc = nn.Identity()
        self.gru = nn.GRU(2048, 256, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(p=0.3)
        self.attn = nn.Linear(512, 1)
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        feats = self.resnet(x)
        feats = feats.view(B, T, -1)
        out, _ = self.gru(feats)
        weights = torch.softmax(self.attn(out), dim=1)  # (B, T, 1)
        out = torch.sum(weights * out, dim=1)           # (B, 512)
        out = self.dropout(out)
        return self.fc(out).squeeze(1)

# === DATASET ===
class ViolenceDataset(Dataset):
    def __init__(self, csv_path, npy_dir):
        self.df = pd.read_csv(csv_path)
        self.npy_dir = npy_dir
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
        frames = np.load(os.path.join(self.npy_dir, f"{segment_id}.npy"))
        frames = frames[:MAX_FRAMES]
        frames = torch.stack([
            self.transform(torch.from_numpy(f).permute(2, 0, 1).float().div(255.0)) for f in frames
        ])
        return frames, torch.tensor(label, dtype=torch.float32)

# === INIT ===
train_dataset = ViolenceDataset(train_csv_path, NPY_DIR)
test_dataset = ViolenceDataset(test_csv_path, NPY_DIR)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet50GRU().to(device)

criterion = nn.BCEWithLogitsLoss()  # binary cross-entropy
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)
scaler = GradScaler()
best_loss = float('inf')
early_stop_counter = 0
EARLY_STOPPING_PATIENCE = 4

# === TRAIN ===
for epoch in range(EPOCHS):
    model.train()
    y_true, y_pred = [], []
    total_loss = 0.0
    optimizer.zero_grad()

    for i, (frames, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
        frames, labels = frames.to(device), labels.to(device)

        with autocast(device_type='cuda'):
            outputs = model(frames)
            loss = criterion(outputs, labels)
            loss = loss / GRAD_ACCUM_STEPS

        scaler.scale(loss).backward()

        if (i + 1) % GRAD_ACCUM_STEPS == 0 or (i + 1) == len(train_loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

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
        torch.save(model.state_dict(), "model_best.pt")
        print("[INFO] Best model saved.")
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= EARLY_STOPPING_PATIENCE:
            print("[INFO] Early stopping triggered.")
            break

# === TEST ===
model.eval()
y_true, y_pred, segment_ids = [], [], test_dataset.df['Segment ID'].tolist()

with torch.no_grad():
    for frames, labels in test_loader:
        frames, labels = frames.to(device), labels.to(device)
        outputs = model(frames)
        preds = (torch.sigmoid(outputs) > 0.5).int()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print("\n[TEST] Classification Report:")
report = classification_report(y_true, y_pred, target_names=["Non-violent", "Violent"], output_dict=True)
print(classification_report(y_true, y_pred, target_names=["Non-violent", "Violent"]))

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

results = pd.DataFrame({"Segment ID": segment_ids, "True": y_true, "Pred": y_pred})
results.to_csv("test_predictions_final.csv", index=False)
pd.DataFrame(report).to_csv("test_metrics_final.csv")
print("[INFO] Predictions and metrics saved.")