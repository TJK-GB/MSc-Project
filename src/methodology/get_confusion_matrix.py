import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import os


BASE_PATH = r"D:\UK\00. 2024 QMUL\00. Course\Project\00. ViolenceDetectionProject\DATASET\RESULTS"
# Load your CSV file
file_path = os.path.join(BASE_PATH, "SWIN+GRU", "swin+GRU_predictions_1.csv")  # Replace with your actual path
df = pd.read_csv(file_path)

# Extract true and predicted labels
y_true = df.iloc[:, 1]  # Second column: True labels
y_pred = df.iloc[:, 2]  # Third column: Predicted labels

# Generate confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Display confusion matrix as text
print("Confusion Matrix:")
print(cm)

# Optional: show classification metrics
print("\nClassification Report:")
print(classification_report(y_true, y_pred, digits=4))

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues', values_format='d')
plt.title("Confusion Matrix")
plt.show()
