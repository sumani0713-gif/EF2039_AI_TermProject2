print("train.py started!")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from model import MLPRegressor
from preprocess import load_data


# Path to dataset
csv_path = "insurance.csv"

# Load data and scaler for inverse transformation
(X_train, X_test, y_train, y_test), scaler_y = load_data(csv_path)
input_dim = X_train.shape[1]

# Build model
model = MLPRegressor(input_dim)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Convert numpy arrays to PyTorch tensors
dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(y_train).float())
loader = DataLoader(dataset, batch_size=32, shuffle=True)

loss_history = []

# Training loop
for epoch in range(100):
    epoch_loss = 0
    for X, y in loader:
        optimizer.zero_grad()
        pred = model(X).squeeze()
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Average loss per epoch
    epoch_loss /= len(loader)
    loss_history.append(epoch_loss)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

# Save training loss curve
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.savefig("loss_curve.png")
print("loss_curve.png saved!")


# -----------------------------
#       Model Evaluation
# -----------------------------
model.eval()
X_test_t = torch.tensor(X_test).float()
y_pred_scaled = model(X_test_t).detach().numpy()

# Inverse scaling to obtain original value range
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_real = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

# RMSE calculation
rmse = ((y_pred - y_real) ** 2).mean() ** 0.5
print(f"Test RMSE: {rmse:.2f}")


# Save predicted vs actual scatter plot
plt.figure(figsize=(6, 6))
plt.scatter(y_real, y_pred, alpha=0.5)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Predicted vs Actual")
plt.savefig("pred_vs_actual.png")
print("pred_vs_actual.png saved!")
