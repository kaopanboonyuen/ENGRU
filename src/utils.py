import torch
import os

def save_model(model, path="gru_model.pth"):
    torch.save(model.state_dict(), path)
    print(f"✅ Model saved to {path}")

def load_model(model, path="gru_model.pth"):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"🔄 Model loaded from {path}")
    else:
        print("⚠️ No saved model found.")
    return model