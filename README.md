# 1. Project Overview

This project aims to develop an end-to-end AI regression model that predicts medical insurance charges based on personal demographic and health-related attributes.
The objective is to experience the full AI development cycle, including:

dataset preprocessing

model architecture design

training

evaluation

visualization

GitHub-based distribution

The model is implemented from scratch without using any pre-trained weights.

# 2. Dataset

- The project uses insurance.csv, which contains the following features:

age, bmi, children, sex (categorical → one-hot encoded), smoker (categorical → one-hot encoded), region (categorical → one-hot encoded), charges (target variable), 

- Preprocessing Steps
mplemented in preprocess.py:
One-hot encoding for categorical variables
Standardization of input features (X) using StandardScaler
Standardization of the target value (y-scaling)

- Train/test split: 80% training / 20% test

Returns scaled training and test sets along with the y-scaler

# 3. Model Architecture

The regression model is a manually constructed Multi-Layer Perceptron (MLP):
'''
Input Layer  → Linear(input_dim → 64)
Hidden Layer → ReLU
Hidden Layer → Linear(64 → 32)
Hidden Layer → ReLU
Output Layer → Linear(32 → 1)
'''

Model Characteristics
- No pre-trained weights
- Lightweight and simple
- Effective for tabular regression tasks
- Fully implemented in model.py

# 4. Training Procedure

Training is implemented in train.py and uses the following setup:

- Training Configuration

Loss function: Mean Squared Error (MSELoss)
Optimizer: Adam (learning rate = 1e-4)
Batch size: 32
Epochs: 100

- Training Output & Visualization

During training, loss values are logged and the following plots are saved:
loss_curve.png → training loss visualization
pred_vs_actual.png → scatter plot of predictions vs actual charges
model_performance_comparison.png → comparison of train/test RMSE & MAE

# 5. Results
1. Training Loss Curve

Shows smooth decrease over epochs

Converges around ~0.15 (scaled target)

Indicates stable and effective training

2. Predicted vs Actual Plot

Demonstrates strong correlation between predictions and ground truth

Model captures nonlinear relationships in the dataset

Errors occur in extreme high-charge values

3. Performance Metrics

Using inverse-transformed predictions:

Test RMSE ≈ 4468

Test MAE ≈ 2647

Given the target range (0–60,000), these values indicate that the model learns overall trends, but still has room for improvement in prediction accuracy.

4. Train vs Test Comparison

Generated as model_performance_comparison.png:

Training error is relatively high

Test error is lower but still significant

Indicates a need for improved generalization and potential overfitting control

# 6. How to Run
Install dependencies
```
pip install -r requirements.txt
```
Run training
```
python train.py
```
or
```
python main.py
```
# 7. File Structure
```
EF2039_AI_TermProject2/
│── main.py
│── train.py
│── model.py
│── preprocess.py
│── insurance.csv
│── loss_curve.png
│── pred_vs_actual.png
│── model_performance_comparison.png
│── README.md
│── requirements.txt
```
# 8. Development Notes

The model was designed entirely from scratch (no pre-trained models used).

Followed the complete AI development workflow:
preprocessing → modeling → training → evaluation → visualization → distribution.

Code includes clear English comments and meaningful variable names.

Git commit history documents step-by-step development progress.

Performance evaluation includes RMSE, MAE, scatter plots, and train/test comparison metrics.

# 9. Author

Name: Sumin Yu
Student ID: 20240555
