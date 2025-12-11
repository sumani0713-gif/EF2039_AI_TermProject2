import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import pandas as pd

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
        epoch_loss += loss.detach().item()  # detaching to avoid unnecessary gradient computation

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

# Predicting for test data
X_test_t = torch.tensor(X_test).float()
y_pred_scaled = model(X_test_t).detach().numpy()

# Inverse scaling to obtain original value range
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_real = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

# RMSE calculation for test data
rmse_test = ((y_pred - y_real) ** 2).mean() ** 0.5
print(f"Test RMSE: {rmse_test:.2f}")  # Output RMSE

# MAE calculation for test data
mae_test = np.mean(np.abs(y_pred - y_real))  # Use numpy to calculate MAE
print(f"Test MAE: {mae_test:.2f}")  # Output MAE

# -----------------------------
#       Training Evaluation
# -----------------------------
# Predicting for training data
y_pred_train = model(torch.tensor(X_train).float()).detach().numpy()
y_real_train = scaler_y.inverse_transform(y_train.reshape(-1, 1)).ravel()

# RMSE and MAE calculation for training data
rmse_train = np.sqrt(np.mean((y_pred_train - y_real_train) ** 2))
mae_train = np.mean(np.abs(y_pred_train - y_real_train))

# -----------------------------
#       Performance Comparison (Train vs Test)
# -----------------------------
# Creating a comparison of model performance metrics
metrics = ['RMSE', 'MAE']
train_metrics = [rmse_train, mae_train]
test_metrics = [rmse_test, mae_test]

# Visualizing with a bar chart
x = np.arange(len(metrics))  # Position of bars
width = 0.35  # Width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, train_metrics, width, label='Train')
rects2 = ax.bar(x + width/2, test_metrics, width, label='Test')

# Labeling the bar chart
ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Model Performance Comparison (Train vs Test)')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()

# Save the bar chart
plt.tight_layout()
plt.savefig("model_performance_comparison.png")
plt.show()

# Printing the performance metrics table
performance_df = pd.DataFrame({
    'Metric': metrics,
    'Train': train_metrics,
    'Test': test_metrics
})

print(performance_df)

# Save predicted vs actual scatter plot
plt.figure(figsize=(6, 6))
plt.scatter(y_real, y_pred, alpha=0.5)
plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Predicted vs Actual")
plt.savefig("pred_vs_actual.png")
print("pred_vs_actual.png saved!")
