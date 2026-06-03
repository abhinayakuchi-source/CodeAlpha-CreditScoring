import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load Dataset
df = pd.read_csv("credit_risk_dataset.csv")

print(df.head())

# Replace 'loan_status' with the actual target column name from your dataset
X = df.drop("loan_status", axis=1)
y = df["loan_status"]

# Convert text columns if any
X = pd.get_dummies(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Prediction
predictions = model.predict(X_test)

# Results
print("\nAccuracy:", accuracy_score(y_test, predictions))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClassification Report:")
print(classification_report(y_test, predictions))