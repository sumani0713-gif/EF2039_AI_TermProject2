# 1. Project Overview

This project aims to develop an end-to-end AI regression model that predicts medical insurance charges based on personal demographic and health-related attributes.
The goal is to experience the full AI development cycle including dataset preprocessing, model architecture design, training, evaluation, visualization, and GitHub-based distribution.

# 2. Dataset

The dataset (insurance.csv) contains the following features:

- age
- bmi
- children
- sex (categorical → one-hot encoded)
- smoker (categorical → one-hot encoded)
- region (categorical → one-hot encoded)
- charges (target variable)

Preprocessing steps include:

- One-hot encoding of categorical variables
- Standardization of numerical features
- Standardization of the target value (y-scaling)
- Train/test split (80/20)
- Implementation: preprocess.py

# 3. Model Architecture

The model is a simple Multi-Layer Perceptron (MLP) designed manually (no pre-trained weights).

```
Input Layer  →  Linear( input_dim → 64 )
Hidden Layer →  ReLU
Hidden Layer →  Linear(64 → 32)
Hidden Layer →  ReLU
Output Layer →  Linear(32 → 1)
```

All model details are implemented in:
model.py

# 4. Training Procedure

Training uses:

- Loss function: Mean Squared Error (MSE)
- Optimizer: Adam (learning rate = 1e-4)
- Batch size: 32
- Epochs: 100

The training loop stores loss values for visualization and saves:
- loss_curve.png
- pred_vs_actual.png

Training logic implemented in: 
- train.py

# 5. Results
1. Training Loss Curve

Shows a smooth decrease and convergence : loss_curve.png

2. Predicted vs Actual Plot
Demonstrates strong positive correlation between model predictions and ground truth : pred_vs_actual.png

3. RMSE (Root Mean Squared Error)
Approx. 4500, which is reasonable for this dataset where target charges range from 0 to 60,000.

# 6. How to Run
Requirements

Install dependencies:
```
pip install -r requirements.txt
````
Run training:
```
python main.py
```
or
```
python train.py
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
│── README.md
│── requirements.txt
```

# 8. Development Notes

- Model designed entirely from scratch (no pre-trained weights).
- Followed full AI development cycle: preprocessing → modeling → training → evaluation → distribution.
- Code includes clear English comments and meaningful variable names.
- Git commit history documents the development process.

# 9. Author

Name: Sumin Yu
Student ID: 20240555
