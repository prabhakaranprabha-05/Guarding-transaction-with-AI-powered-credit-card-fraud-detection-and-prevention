import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv('cdd.csv')

# Print the columns of the DataFrame
print(df.columns)

# Check for the target column
if 'is_fraud' in df.columns:
    X = df.drop(['is_fraud'], axis=1)
    y = df['is_fraud']
else:
    target_column = 'Class'  # Replace with the actual target column name
    print(f"Assuming target variable is '{target_column}' instead.")
    X = df.drop([target_column], axis=1)
    y = df[target_column]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Classification report as dict
report = classification_report(y_test, y_pred, output_dict=True)

# Convert report to DataFrame
report_df = pd.DataFrame(report).transpose()

# Drop support column for bar plot clarity
metrics_df = report_df.drop(columns=['support'], errors='ignore')

# Plotting
metrics_df[['precision', 'recall', 'f1-score']].iloc[:-1].plot(kind='bar')
plt.title('Classification Report Metrics')
plt.ylabel('Score')
plt.xlabel('Class')
plt.xticks(rotation=0)
plt.ylim(0, 1)
plt.legend(loc='lower right')
plt.tight_layout()
plt.show()

# Save model
joblib.dump(model, 'fraud_detector.pkl')