import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('creditcard.csv')

# Print available columns and the first few rows
print("Available columns:", data.columns)
print(data.head())

# Fill missing values
data.bfill(inplace=True)

# Check if required columns are present
if 'Amount' not in data.columns or 'Class' not in data.columns:
    raise KeyError("The dataset does not contain 'Amount' or 'Class' column.")

# Scale the 'Amount' feature
scaler = StandardScaler()
data[['Amount']] = scaler.fit_transform(data[['Amount']])

# Initialize and fit the Isolation Forest model
model = IsolationForest(n_estimators=100, max_samples='auto', contamination=0.01, random_state=42)
model.fit(data[['Amount']])

# Predict anomalies (1: normal, -1: anomaly)
data['anomaly'] = model.predict(data[['Amount']])

# Convert predictions from (-1, 1) to (0, 1) for easier analysis
data['anomaly'] = data['anomaly'].apply(lambda x: 1 if x == -1 else 0)

# Show a few examples of predicted anomalies
print("Predicted anomalies:")
print(data[data['anomaly'] == 1].head())

# Print classification report and confusion matrix
print("Classification Report:")
print(classification_report(data['Class'], data['anomaly']))
print("Confusion Matrix:")
print(confusion_matrix(data['Class'], data['anomaly']))

# Compute ROC AUC and plot ROC Curve
y_true = data['Class']
y_pred_prob = model.decision_function(data[['Amount']])
roc_auc = roc_auc_score(y_true, y_pred_prob)
fpr, tpr, _ = roc_curve(y_true, y_pred_prob)

# Compute Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_true, y_pred_prob)

# Plot ROC Curve and Precision-Recall Curve
plt.figure(figsize=(12, 6))

# ROC Curve
plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, marker='.')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

# Precision-Recall Curve
plt.subplot(1, 2, 2)
plt.plot(recall, precision, marker='.')
plt.title('Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')

plt.show()

print(f'ROC AUC Score: {roc_auc}')
