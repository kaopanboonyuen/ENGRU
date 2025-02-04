import torch
import torch.optim as optim
import torch.nn as nn
from model import GRUModel
from dataset_loader import get_dataloader
from utils import save_model

# Hyperparameters
VOCAB_SIZE = 50
BATCH_SIZE = 32
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Data
train_loader = get_dataloader(BATCH_SIZE)

# Model, Loss, Optimizer
model = GRUModel(vocab_size=VOCAB_SIZE).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
for epoch in range(EPOCHS):
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)

        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {total_loss/len(train_loader):.4f}")

# Save Model
save_model(model)
print("✅ Training Complete!")