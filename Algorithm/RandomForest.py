# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 15:40:31 2025

@author: papem
"""

# ==============================
# Libraries
# ==============================
import os, glob, time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, butter, filtfilt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ==============================
# Load data
# ==============================
folder_path = 'data/ALL_modified/'
file_pattern = os.path.join(folder_path, '*.txt')
required_columns = [f'A{i}' for i in range(1, 19)] + ['label']

all_data = []
errors = []

for file_path in glob.glob(file_pattern):
    try:
        df = pd.read_csv(file_path)
        if not all(col in df.columns for col in required_columns):
            errors.append(f"{file_path} skipped: missing columns")
            continue
        df = df[required_columns].dropna()
        if df.empty:
            errors.append(f"{file_path} skipped: empty after cleaning")
            continue
        df['source'] = os.path.basename(file_path)
        all_data.append(df)
    except Exception as e:
        errors.append(f"{file_path} skipped: {str(e)}")

if not all_data:
    print("No valid files found.")
    for err in errors:
        print(err)
    raise SystemExit

data = pd.concat(all_data, ignore_index=True)

# ==============================
# Filtering (optional)
# ==============================
def apply_filter(sequence, filter_type='lowpass'):
    filtered = sequence.copy()
    if filter_type == 'lowpass':
        b, a = butter(N=3, Wn=0.9, btype='low')
        for col in filtered.columns:
            try:
                filtered[col] = filtfilt(b, a, filtered[col])
            except Exception:
                pass
    elif filter_type == 'savgol':
        for col in filtered.columns:
            try:
                filtered[col] = savgol_filter(filtered[col], window_length=7, polyorder=3)
            except Exception:
                pass
    elif filter_type == 'ema':
        filtered = filtered.ewm(span=5, adjust=False).mean()
    else:
        print(f"Unknown filter type: {filter_type}")
    return filtered

# Comment / Uncomment to apply filtering
emg_columns = [f'A{i}' for i in range(1, 19)]
filtered_emg = apply_filter(data[emg_columns], filter_type='ema')
data_filtered = pd.concat([filtered_emg, data[['label', 'source']].reset_index(drop=True)], axis=1)

# If filtering is not applied, use raw data
data_filtered = data.copy()

# ==============================
# Segment gestures
# ==============================
def segment_gestures(data, min_length=3):
    sequences = []
    current_sequence = []
    current_label = None
    for _, row in data.iterrows():
        label = row['label']
        features = row.drop('label')
        if label == current_label:
            current_sequence.append(features)
        else:
            if current_sequence and len(current_sequence) >= min_length:
                sequences.append((current_label, pd.DataFrame(current_sequence)))
            current_label = label
            current_sequence = [features]
    if current_sequence and len(current_sequence) >= min_length:
        sequences.append((current_label, pd.DataFrame(current_sequence)))
    return sequences

gesture_sequences = segment_gestures(data_filtered)
gesture_sequences = [(label, seq) for label, seq in gesture_sequences if str(label).lower() != 'neutral']

print(f"\n Nombre de séquences après suppression de 'Neutral' : {len(gesture_sequences)}")

# ==============================
# Feature extraction (optional)
# ==============================
def extract_features_from_sequence(sequence):
    features = {}
    base_cols = [f'A{i}' for i in range(1, 17)]
    for col in base_cols:
        values = sequence[col]
        features[f'{col}_mean'] = values.mean()
        features[f'{col}_std'] = values.std()
        features[f'{col}_min'] = values.min()
        features[f'{col}_max'] = values.max()
    features['A17_mean'] = sequence['A17'].mean()
    features['A18_weighted'] = sequence['A18'].mean()
    return features

# Comment / Uncomment if using segmented sequences
feature_rows = []
for label, sequence in gesture_sequences:
    row = extract_features_from_sequence(sequence)
    row['label'] = label
    feature_rows.append(row)
features_df = pd.DataFrame(feature_rows).dropna()


# ==============================
# Train/test split
# ==============================
X = features_df.drop('label', axis=1)
y = features_df['label']

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in sss.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]


# ==============================
# Random Forest training (uncomment to tune)
# ==============================

# # Define the hyperparameter grid to test
# # Comment / Uncomment to try different configurations
# param_grid = {
#     'n_estimators': [100],
#     'max_depth': [None],
#     'min_samples_split': [2],
#     'min_samples_leaf': [2]
# }

# Reduced grid
param_grid = {
    'n_estimators': [100]
}

# Initialize RF model with a base configuration
rf = RandomForestClassifier(random_state=42)

# Start timing the training process
start_time = time.time()

# Grid Search with cross-validation to find best hyperparameters
grid_search = GridSearchCV(
    estimator=rf,         # base RF model
    param_grid=param_grid, # hyperparameter grid to search
    cv=5,                  # 5-fold cross-validation
    n_jobs=-1,             # use all available CPU cores
    scoring='accuracy',    # evaluation metric
    verbose=0              # verbosity level (0 = silent)
)

# Fit the model with training data
grid_search.fit(X_train, y_train)

# End timing
end_time = time.time()


# Retrieve the best model found
best_model = grid_search.best_estimator_
print("Best params:", grid_search.best_params_)

# Print training time
print(f"Training completed in {end_time - start_time:.6f} seconds")


# ==============================
# Evaluation
# ==============================
start_pred = time.time()
y_pred = best_model.predict(X_test)
end_pred = time.time()

# Average prediction time per sample (in milliseconds)
prediction_time = (end_pred - start_pred) / len(X_test) * 1000

# Display results
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2%}")
print(f"Average prediction time: {prediction_time:.3f} ms per gesture")
print("\nClassification report:")
print(classification_report(y_test, y_pred))

# Display a confusion matrix in percentage
def plot_confusion_matrix_percent(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    plt.figure(figsize=(8, 6))
    sns.set(font_scale=2.1)
    sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix (%)')
    plt.tight_layout()
    plt.show()

unique_labels = sorted(set(y))
plot_confusion_matrix_percent(y_test, y_pred, unique_labels)

# Display examples where the model made mistakes
errors = (y_test != y_pred)
X_test_errors = X_test[errors]
y_test_errors = y_test[errors]
y_pred_errors = y_pred[errors]

print(f"Number of errors: {len(y_test_errors)} out of {len(y_test)} samples")
for i in range(len(y_test_errors)):
    print(f"  - Sample {i+1}: True = {y_test_errors.iloc[i]}, Predicted = {y_pred_errors[i]}")

# Generate a table of classification errors
def get_classification_errors_table(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    error_table = []

    for i, true_label in enumerate(unique_labels):
        for j, pred_label in enumerate(unique_labels):
            if i != j and cm[i, j] > 0:
                total_true = cm[i].sum()
                error_rate = cm[i, j] / total_true * 100
                error_table.append({
                    'True class': true_label,
                    'Predicted class': pred_label,
                    'Number of errors': cm[i, j],
                    'Error percentage': round(error_rate, 2)
                })

    return pd.DataFrame(error_table)

# Display the error table
error_df = get_classification_errors_table(y_test, y_pred)
print("\nClassification errors table:")
print(error_df)

# ==============================
# # Save model (uncomment to save)
# ==============================
# joblib.dump(best_model, 'random_forest_model.pkl')
# print("Model saved as 'random_forest_model.pkl'")

