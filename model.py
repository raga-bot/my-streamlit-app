# Diabetes Prediction with Machine Learning

# ✅ Step 1: Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression # Import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc # Import roc_curve and auc
from sklearn.preprocessing import StandardScaler
import joblib
from google.colab import files
import pickle

# ✅ Step 2: Load Data
# Upload the file
uploaded = files.upload()

# Assuming the file uploaded is named 'diabetes.csv'.
for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))
  file_name = fn

df = pd.read_csv(file_name)

# ✅ Step 3: Preprocessing
cols_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)
df.fillna(df.median(), inplace=True)

# Separate features (X) and target (y) BEFORE scaling
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ✅ Step 4: EDA - Deeper Analysis
# Distribution of features
df.hist(figsize=(15, 10))
plt.tight_layout()
plt.show()

# Box plots to check for outliers and distribution by outcome
plt.figure(figsize=(15, 10))
for i, col in enumerate(X.columns):
    plt.subplot(3, 3, i + 1)
    sns.boxplot(x=y, y=X[col])
    plt.title(f'Box Plot of {col} by Outcome')
plt.tight_layout()
plt.show()

# Check for class imbalance
print("\nDistribution of the target variable (Outcome):")
print(y.value_counts())
print(y.value_counts(normalize=True) * 100) # Percentage

# ✅ Step 5: Scale Numerical Features
# Initialize the StandardScaler
scaler = StandardScaler()

# Fit the scaler on the entire dataset (or X_train if you prefer scaling before splitting) and transform
# Note: In your provided code, you fitted on X and then split X_scaled.
# It's generally recommended to fit the scaler ONLY on the training data
# and then transform both training and testing data. Let's update this to be more standard.

# Split before scaling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Fit scaler on training data ONLY
scaler.fit(X_train)

# Transform both training and testing data
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert scaled arrays back to DataFrames (optional, but good practice to maintain column names)
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)


# ✅ Step 6: Model Training and Evaluation (Random Forest - with Hyperparameter Tuning)
print("\n--- Random Forest Model ---")
# Define the parameter grid for GridSearchCV
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Initialize the Random Forest Classifier
rf = RandomForestClassifier(random_state=42)

# Initialize GridSearchCV for Random Forest
grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid_rf, cv=5, n_jobs=-1, verbose=2, scoring='accuracy') # Added scoring

# Fit GridSearchCV to the training data (using scaled training data)
grid_search_rf.fit(X_train_scaled, y_train)

# Get the best Random Forest model
best_model_rf = grid_search_rf.best_estimator_

# Make predictions with the best Random Forest model (using scaled testing data)
y_pred_rf = best_model_rf.predict(X_test_scaled)
y_prob_rf = best_model_rf.predict_proba(X_test_scaled)[:, 1] # Get probabilities for ROC curve

# Evaluation for Random Forest
print("Confusion Matrix (Random Forest):\n", confusion_matrix(y_test, y_pred_rf))
print("Classification Report (Random Forest):\n", classification_report(y_test, y_pred_rf))
print("Accuracy (Random Forest):", accuracy_score(y_test, y_pred_rf))
print("Best Hyperparameters (Random Forest):", grid_search_rf.best_params_)

# Plot ROC curve for Random Forest
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
roc_auc_rf = auc(fpr_rf, tpr_rf)
plt.figure(figsize=(8, 6))
plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label=f'ROC curve (Random Forest) (area = {roc_auc_rf:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# ✅ Step 8: Model Training and Evaluation (Logistic Regression)
print("\n--- Logistic Regression Model ---")
# Initialize the Logistic Regression Classifier
log_reg = LogisticRegression(random_state=42, solver='liblinear') # Use liblinear solver for small datasets

# Define a simpler parameter grid for Logistic Regression (L1/L2 regularization and C)
param_grid_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100], # Inverse of regularization strength
    'penalty': ['l1', 'l2'] # Specify L1 or L2 regularization
}

# Initialize GridSearchCV for Logistic Regression
grid_search_lr = GridSearchCV(estimator=log_reg, param_grid=param_grid_lr, cv=5, n_jobs=-1, verbose=2, scoring='accuracy') # Added scoring

# Fit GridSearchCV to the training data (using scaled training data)
grid_search_lr.fit(X_train_scaled, y_train)

# Get the best Logistic Regression model
best_model_lr = grid_search_lr.best_estimator_

# Make predictions with the best Logistic Regression model (using scaled testing data)
y_pred_lr = best_model_lr.predict(X_test_scaled)
y_prob_lr = best_model_lr.predict_proba(X_test_scaled)[:, 1] # Get probabilities for ROC curve

# Evaluation for Logistic Regression
print("Confusion Matrix (Logistic Regression):\n", confusion_matrix(y_test, y_pred_lr))
print("Classification Report (Logistic Regression):\n", classification_report(y_test, y_pred_lr))
print("Accuracy (Logistic Regression):", accuracy_score(y_test, y_pred_lr))
print("Best Hyperparameters (Logistic Regression):", grid_search_lr.best_params_)

# Plot ROC curve for Logistic Regression
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)
plt.figure(figsize=(8, 6))
plt.plot(fpr_lr, tpr_lr, color='darkorange', lw=2, label=f'ROC curve (Logistic Regression) (area = {roc_auc_lr:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# ✅ Step 9: Feature Importance (using the best Random Forest model)
# We'll still show feature importance for the Random Forest as it's a tree-based model
print("\n--- Feature Importance (from Best Random Forest Model) ---")
importances_rf = best_model_rf.feature_importances_
features = X_train.columns # Use columns from the original X_train before scaling
plt.figure(figsize=(8, 6))
sns.barplot(x=importances_rf, y=features)
plt.title("Feature Importance (from Best Random Forest Model)")
plt.show()


# ✅ Step 10: Save Model (save the best performing model - you might want to decide based on accuracy or other metrics)
# For now, let's save the best Random Forest model and the scaler
with open('diabetes_model_rf.pkl', 'wb') as f:
    pickle.dump(best_model_rf, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\nModels and scaler trained and saved.")