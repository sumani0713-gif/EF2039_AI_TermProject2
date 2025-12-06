import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(path):
    """
    Load dataset, preprocess features, and return train/test split.
    """
    df = pd.read_csv(path)

    # One-hot encode categorical columns
    df = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True)

    # Separate input features (X) and target (y)
    X = df.drop("charges", axis=1).values
    y = df["charges"].values.reshape(-1, 1)

    # Standardize input features
    scaler_x = StandardScaler()
    X_scaled = scaler_x.fit_transform(X)

    # Standardize target values for stable training
    scaler_y = StandardScaler()
    y_scaled = scaler_y.fit_transform(y).ravel()

    # Return train/test sets and target scaler
    return train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42), scaler_y
