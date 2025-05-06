
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

# Load sample dataset (replace with full NSL-KDD for better performance)
df = pd.read_csv("nsl_kdd_sample.csv")

# Encode labels for binary classification (0 = normal, 1 = anomaly)
df['binary_label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)

# Drop label column for training
X = df.drop(columns=['label', 'binary_label'])
y = df['binary_label']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Isolation Forest
model = IsolationForest(contamination=0.05)
model.fit(X_scaled)

# Save model and scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
