# -*- coding: utf-8 -*-
"""
Created on Thu Jul 02 12:20:37 2025

@author: papem
"""

# === Libraries for data manipulation ===
import os, glob, time
import pandas as pd
import numpy as np

# === Libraries for visualization ===
import seaborn as sns
import matplotlib.pyplot as plt

# === Signal processing and statistics ===
from collections import Counter
from scipy.signal import savgol_filter, butter, filtfilt

# === Machine learning models, hyperparameter tuning, and evaluation ===
from sklearn.ensemble import HistGradientBoostingClassifier, KNeighborsClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# === Path to the EMG text data files ===
folder_path = 'data/ALL_modified/'
file_pattern = os.path.join(folder_path, '*.txt')

# Expected columns: A1 to A18 + 'label'
required_columns = [f'A{i}' for i in range(1, 19)] + ['label']

all_data = []   # Will store valid DataFrames
errors = []     # Will store file loading errors

print("Loading files...")

# Loop through all .txt files in the directory
for file_path in glob.glob(file_pattern):
    try:
        df = pd.read_csv(file_path)

        # Check if all required columns are present
        if not all(col in df.columns for col in required_columns):
            errors.append(f"{file_path} ignored: missing columns")
            continue

        # Keep only required columns and drop empty rows
        df = df[required_columns].dropna()
        if df.empty:
            errors.append(f"{file_path} ignored: empty after cleaning")
            continue

        # Add source filename for traceability
        df['source'] = os.path.basename(file_path)
        all_data.append(df)
        print(f"{file_path} loaded ({len(df)} rows)")

    except Exception as e:
        errors.append(f"{file_path} ignored: {str(e)}")

# Stop the script if no valid files were loaded
if not all_data:
    print("No valid file found.")
    for err in errors:
        print(err)
    raise SystemExit

# Merge all DataFrames into a single one
data = pd.concat(all_data, ignore_index=True)
print(f"\nTotal combined rows: {len(data)}")


# === Apply filter to EMG signal columns to reduce noise ===
def apply_filter(sequence, filter_type='lowpass'):
    filtered = sequence.copy()

    if filter_type == 'lowpass':
        # Butterworth low-pass filter
        b, a = butter(N=3, Wn=0.5, btype='low')
        for col in filtered.columns:
            try:
                filtered[col] = filtfilt(b, a, filtered[col])
            except Exception:
                pass

    elif filter_type == 'savgol':
        # Savitzky-Golay smoothing filter
        for col in filtered.columns:
            try:
                filtered[col] = savgol_filter(filtered[col], window_length=5, polyorder=3)
            except Exception:
                pass

    elif filter_type == 'ema':
        # Exponential moving average
        filtered = filtered.ewm(span=5, adjust=False).mean()

    else:
        print(f"Unknown filter: {filter_type}")

    return filtered

# Apply filtering only to EMG columns (A1 to A18)
emg_columns = [f'A{i}' for i in range(1, 19)]
filtered_emg = apply_filter(data[emg_columns], filter_type='lowpass')

# Keep unfiltered 'label' and 'source' columns
data_filtered = pd.concat([filtered_emg, data[['label', 'source']].reset_index(drop=True)], axis=1)
print("Filtering completed.")


# === No filtering applied (Uncomment to use) ===
# emg_columns = [f'A{i}' for i in range(1, 19)]
# data_no_filtered = pd.concat([data[emg_columns], data[['label', 'source']].reset_index(drop=True)], axis=1)
# print("No filtering applied.")


# === Group consecutive rows with the same label into sequences ===
def segment_gestures(data, min_length=1):
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

# Remove all sequences labeled as "neutral"
gesture_sequences = [
    (label, seq) for label, seq in gesture_sequences
    if str(label).lower() != 'neutral'
]

print(f"\nNumber of sequences after removing 'Neutral': {len(gesture_sequences)}")

# Count remaining gesture labels
labels = [label for label, _ in gesture_sequences]
labels_count = Counter(labels)

print("\nGesture distribution:")
for label, count in labels_count.items():
    print(f"  {label} : {count} sequences")


# === Feature extraction from sequences ===
def extract_features_from_sequence(sequence):
    """
    Extracts statistical features from a sequence.

    Parameters:
    - sequence: DataFrame with columns A1 to A18

    Returns:
    - Dictionary of features
    """
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

feature_rows = []

for label, sequence in gesture_sequences:
    row = extract_features_from_sequence(sequence)
    row['label'] = label
    feature_rows.append(row)

features_df = pd.DataFrame(feature_rows).dropna()


# === Prepare data for model training ===
X = features_df.drop('label', axis=1)
y = features_df['label']

# Stratified split to maintain class balance
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in sss.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]



# === Confusion matrix (% values) ===
def plot_confusion_matrix_percent(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    plt.figure(figsize=(8, 6))
    sns.set(font_scale=2.1)
    sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predictions')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix (%)')
    plt.tight_layout()
    plt.show()


unique_labels = sorted(set(y)) 

   
# === Generate classification error table ===
def get_classification_errors_table(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    error_table = []

    for i, true_label in enumerate(unique_labels):
        for j, pred_label in enumerate(unique_labels):
            if i != j and cm[i, j] > 0:
                total_true = cm[i].sum()
                error_rate = cm[i, j] / total_true * 100
                error_table.append({
                    'True Class': true_label,
                    'Predicted Class': pred_label,
                    'Error Count': cm[i, j],
                    'Error Rate (%)': round(error_rate, 2)
                })

    return pd.DataFrame(error_table)


# %%

# === Random Forest Classifier ===

param_grid = {
    'n_estimators': [50],       # Number of trees
    'max_depth': [None],        # Unlimited depth
    'min_samples_split': [2],   # Minimum samples to split a node
    'min_samples_leaf': [1]     # Minimum samples per leaf
}

rf = RandomForestClassifier(random_state=42)

print("\nHyperparameter search started...")
start_time = time.time()

# Grid search with cross-validation
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring='accuracy',
    verbose=0
)

grid_search.fit(X_train, y_train)
end_time = time.time()

print(f"GridSearch completed in {end_time - start_time:.2f} seconds")
print("Best hyperparameters:", grid_search.best_params_)

# === Prediction and evaluation ===
start_pred = time.time()
y_pred = grid_search.best_estimator_.predict(X_test)
end_pred = time.time()

prediction_time = (end_pred - start_pred) / len(X_test) * 1000
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.2%}")
print(f"Avg prediction time: {prediction_time:.3f} ms per sample")
print("\nClassification report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
plot_confusion_matrix_percent(y_test, y_pred, unique_labels)

# Display misclassified samples
errors = (y_test != y_pred)
X_test_errors = X_test[errors]
y_test_errors = y_test[errors]
y_pred_errors = y_pred[errors]

print(f"Number of errors: {len(y_test_errors)} / {len(y_test)} samples")
for i in range(len(y_test_errors)):
    print(f"  - Sample {i+1}: True = {y_test_errors.iloc[i]}, Pred = {y_pred_errors[i]}")

# Generate classification error table
error_df = get_classification_errors_table(y_test, y_pred)
print("\nClassification error table:")
print(error_df)


# %%

# === Classifier (Uncomment to use) ===

# # === HistGradientBoostingClassifier ===

# param_grid = {
#     'learning_rate': [0.1],
#     'max_iter': [50],
#     'max_depth': [None],
#     'l2_regularization': [1]
# }

# hgb = HistGradientBoostingClassifier()

# print("\nHyperparameter search started...")
# start_time = time.time()

# grid_search = GridSearchCV(
#     estimator=hgb,
#     param_grid=param_grid,
#     cv=5,
#     n_jobs=-1,
#     scoring='accuracy',
#     verbose=0
# )

# grid_search.fit(X_train, y_train)
# end_time = time.time()

# print(f"GridSearch completed in {end_time - start_time:.2f} seconds")
# print("Best hyperparameters:", grid_search.best_params_)

# # === Prediction and evaluation ===
# start_pred = time.time()
# y_pred = grid_search.best_estimator_.predict(X_test)
# end_pred = time.time()

# prediction_time = (end_pred - start_pred) / len(X_test) * 1000
# accuracy = accuracy_score(y_test, y_pred)

# print(f"\nAccuracy: {accuracy:.2%}")
# print(f"Avg prediction time: {prediction_time:.3f} ms per sample")
# print("\nClassification report:")
# print(classification_report(y_test, y_pred))


# plot_confusion_matrix_percent(y_test, y_pred, unique_labels)


# # === Display incorrect predictions ===
# errors = (y_test != y_pred)
# X_test_errors = X_test[errors]
# y_test_errors = y_test[errors]
# y_pred_errors = y_pred[errors]

# print(f"Number of errors: {len(y_test_errors)} / {len(y_test)} samples")
# for i in range(len(y_test_errors)):
#     print(f"  - Sample {i+1}: True = {y_test_errors.iloc[i]}, Pred = {y_pred_errors[i]}")



# error_df = get_classification_errors_table(y_test, y_pred)
# print("\nClassification error table:")
# print(error_df)

# %%

# === Classifier (Uncomment to use) ===

# === K-Nearest Neighbors (KNN) Classifier ===

# param_grid = {
#     'weights': ['distance'],  # Weighting by distance
#     'p': [1]  # p=1 for Manhattan distance, p=2 for Euclidean
# }

# knn = KNeighborsClassifier(n_neighbors=5)  # Use 5 neighbors

# print("\nHyperparameter search started...")
# start_time = time.time()

# # Grid search with cross-validation
# grid_search = GridSearchCV(
#     estimator=knn,
#     param_grid=param_grid,
#     cv=5,
#     n_jobs=-1,
#     scoring='accuracy',
#     verbose=0
# )

# grid_search.fit(X_train, y_train)
# knn.fit(X_train, y_train)  # Optional: also fit the base model
# end_time = time.time()

# print(f"GridSearch completed in {end_time - start_time:.2f} seconds")
# print("Best hyperparameters:", grid_search.best_params_)

# # Ensure input format is contiguous for KNN
# X_test = np.ascontiguousarray(X_test)

# # === Prediction and evaluation ===
# start_pred = time.time()
# y_pred = grid_search.best_estimator_.predict(X_test)
# end_pred = time.time()

# prediction_time = (end_pred - start_pred) / len(X_test) * 1000
# accuracy = accuracy_score(y_test, y_pred)

# print(f"\nAccuracy: {accuracy:.2%}")
# print(f"Avg prediction time: {prediction_time:.3f} ms per sample")
# print("\nClassification report:")
# print(classification_report(y_test, y_pred))

# # Plot confusion matrix
# plot_confusion_matrix_percent(y_test, y_pred, unique_labels)

# # Display misclassified samples
# errors = (y_test != y_pred)
# X_test_errors = X_test[errors]
# y_test_errors = y_test[errors]
# y_pred_errors = y_pred[errors]

# print(f"Number of errors: {len(y_test_errors)} / {len(y_test)} samples")
# for i in range(len(y_test_errors)):
#     print(f"  - Sample {i+1}: True = {y_test_errors.iloc[i]}, Pred = {y_pred_errors[i]}")

# # Generate classification error table
# error_df = get_classification_errors_table(y_test, y_pred)
# print("\nClassification error table:")
# print(error_df)
