import argparse
import torch
import torch.nn as nn

# === Models ===
from models.resnet50_gru import ViolenceModel as ResNet50GRU
from models.swin import ViolenceModel as SwinOnly
from models.swin_gru import ViolenceModel as SwinGRU

def get_model(name):
    """Return the selected model instance."""
    if name.lower() == "resnet50_gru":
        return ResNet50GRU(hidden=256, dropout=0.3)
    elif name.lower() == "swin":
        return SwinOnly(dropout=0.3)
    elif name.lower() == "swin_gru":
        return SwinGRU(hidden=256, dropout=0.3)
    else:
        raise ValueError(f"Unknown model name: {name}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Model type: resnet50_gru | swin | swin_gru")
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    # === Select model ===
    model = get_model(args.model)
    model = model.cuda() if torch.cuda.is_available() else model

    print(f"[INFO] Using model: {args.model}")
    print(model)

    # === Your existing train/test code here ===
    # Example:
    # train_loader, val_loader = ...
    # criterion = nn.BCEWithLogitsLoss()
    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    #
    # for epoch in range(args.epochs):
    #     train_one_epoch(model, train_loader, criterion, optimizer, epoch)
    #     validate(model, val_loader, criterion)

if __name__ == "__main__":
    main()
