# -*- coding: utf-8 -*-
"""
Created on Wed Jul 16 16:01:20 2025

@author: madou fall
"""


# ==============================
# Libraries
# ==============================

# OS and file handling
import os, glob, time

# Data manipulation
import pandas as pd
import numpy as np

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Counting and basic data structures
from collections import Counter

# Signal processing
from scipy.signal import savgol_filter, butter, filtfilt

# Machine learning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Model saving/loading
import joblib


# ==============================
# Load data
# ==============================

# Folder containing the data files
folder_path = 'data/ALL/'

# Pattern to match all text files in the folder
file_pattern = os.path.join(folder_path, '*.txt')

# Columns we expect in each file
required_columns = [f'A{i}' for i in range(1, 19)] + ['label']

# Lists to store valid dataframes and any errors encountered
all_data = []
errors = []

# Loop through all matching files
for file_path in glob.glob(file_pattern):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Check if all required columns are present
        if not all(col in df.columns for col in required_columns):
            errors.append(f"{file_path} skipped: missing columns")
            continue
        
        # Keep only required columns and drop rows with missing values
        df = df[required_columns].dropna()
        
        # Skip if the dataframe is empty after cleaning
        if df.empty:
            errors.append(f"{file_path} skipped: empty after cleaning")
            continue
        
        # Add a column indicating the source file
        df['source'] = os.path.basename(file_path)
        
        # Append cleaned dataframe to the list
        all_data.append(df)
        
    except Exception as e:
        # Record any errors encountered while reading files
        errors.append(f"{file_path} skipped: {str(e)}")

# Stop execution if no valid files were found
if not all_data:
    print("No valid files found.")
    for err in errors:
        print(err)
    raise SystemExit

# Concatenate all valid dataframes into a single dataframe
data = pd.concat(all_data, ignore_index=True)

# ==============================
# Filtering (optional)
# ==============================


def apply_filter(sequence, filter_type='lowpass'):
    """
    Apply a smoothing/filtering method to a sequence of signals.
    
    Parameters:
        sequence (DataFrame): Signal columns to filter.
        filter_type (str): 'lowpass', 'savgol', or 'ema'.
        
    Returns:
        DataFrame: Filtered signals.
    """
    filtered = sequence.copy()
    
    if filter_type == 'lowpass':
        # Apply a low-pass Butterworth filter
        b, a = butter(N=3, Wn=0.9, btype='low')
        for col in filtered.columns:
            try:
                filtered[col] = filtfilt(b, a, filtered[col])
            except Exception:
                pass  # Ignore columns that can't be filtered
                
    elif filter_type == 'savgol':
        # Apply a Savitzky-Golay filter
        for col in filtered.columns:
            try:
                filtered[col] = savgol_filter(filtered[col], window_length=7, polyorder=3)
            except Exception:
                pass
                
    elif filter_type == 'ema':
        # Apply Exponential Moving Average smoothing
        filtered = filtered.ewm(span=5, adjust=False).mean()
        
    else:
        print(f"Unknown filter type: {filter_type}")
        
    return filtered

# Specify which columns contain EMG signals
emg_columns = [f'A{i}' for i in range(1, 19)]

# Apply chosen filter (here: EMA)
filtered_emg = apply_filter(data[emg_columns], filter_type='ema')

# Combine filtered signals with label and source columns
data_filtered = pd.concat([filtered_emg, data[['label', 'source']].reset_index(drop=True)], axis=1)

# If filtering is not applied, just use the raw data
data_filtered = data.copy()


# ==============================
# Segment gestures
# ==============================


def segment_gestures(data, min_length=3):
    """
    Segment continuous rows of the same label into separate gesture sequences.

    Parameters:
        data (DataFrame): The input dataset containing a 'label' column.
                          Each row corresponds to one time step of a gesture.
        min_length (int): Minimum number of rows required to keep a sequence.
                          (shorter sequences will be discarded)

    Returns:
        List of tuples: Each element is (label, sequence_dataframe)
                        - label: gesture type (e.g., "Swipe", "Tap")
                        - sequence_dataframe: DataFrame with all rows of that gesture
    """
    sequences = []           # Will hold the final segmented gestures
    current_sequence = []    # Temporary buffer for the current gesture
    current_label = None     # Tracks the active gesture label

    # --- Iterate through each row in the dataset ---
    for _, row in data.iterrows():
        label = row['label']             # Current gesture label
        features = row.drop('label')     # Sensor features without the label

        if label == current_label:
            # Continue adding rows to the current sequence
            current_sequence.append(features)
        else:
            # If the label changed, save the previous sequence (if it's long enough)
            if current_sequence and len(current_sequence) >= min_length:
                sequences.append((current_label, pd.DataFrame(current_sequence)))

            # Start a new sequence for the new label
            current_label = label
            current_sequence = [features]

    # --- Handle the last sequence after finishing the loop ---
    if current_sequence and len(current_sequence) >= min_length:
        sequences.append((current_label, pd.DataFrame(current_sequence)))

    return sequences


# --- Main Gesture Segmentation Pipeline ---

# Segment gestures from the filtered dataset
gesture_sequences = segment_gestures(data_filtered)

# Step 4: Remove "Neutral" gestures (they are not meaningful for classification)
gesture_sequences = [
    (label, seq) for label, seq in gesture_sequences
    if str(label).lower() != 'neutral'
]

print(f"\nNumber of sequences after removing 'Neutral': {len(gesture_sequences)}")

# Count how many sequences remain for each gesture type
labels = [label for label, _ in gesture_sequences]
labels_count = Counter(labels)

# Print the distribution of gesture sequences
print("\nGesture distribution:")
for label, count in labels_count.items():
    print(f" {label} : {count} sequences")



# ==============================
# Feature extraction
# ==============================


def extract_features_from_sequence(sequence):
    """
    Extract statistical features from a gesture sequence.

    Parameters:
        sequence (DataFrame): A single gesture sequence, where each column (A1–A18)
                              represents sensor readings or measurements.

    Returns:
        dict: A dictionary containing extracted statistical features.
    """
    features = {}
    base_cols = [f'A{i}' for i in range(1, 17)]  # Define columns A1–A16 for standard feature extraction

    # --- Compute basic statistics for A1–A16 ---
    # For each sensor column, we calculate:
    # - Mean: average value of the signal
    # - Std: how much the values vary (standard deviation)
    # - Min: the smallest observed value
    # - Max: the largest observed value
    for col in base_cols:
        values = sequence[col]
        features[f'{col}_mean'] = values.mean()
        features[f'{col}_std'] = values.std()
        features[f'{col}_min'] = values.min()
        features[f'{col}_max'] = values.max()

    # --- Add extra features from A17 and A18 ---
    # These may represent different types of signals, so we extract specific summaries:
    features['A17_mean'] = sequence['A17'].mean()         # Average signal of A17
    features['A18_weighted'] = sequence['A18'].mean()     # Average of A18 (can be used as a weighted signal)

    return features


# --- Main Feature Extraction Pipeline ---

feature_rows = []  # List to store feature dictionaries (one per gesture sequence)

# Iterate over all gesture sequences
# Each item in gesture_sequences contains: (label, sequence)
# - label: the class or type of gesture
# - sequence: the DataFrame of sensor values for that gesture
for label, sequence in gesture_sequences:
    row = extract_features_from_sequence(sequence)  # Extract features for this sequence
    row['label'] = label                            # Attach the gesture label
    feature_rows.append(row)                        # Add the result to our dataset

# Build the final feature DataFrame
# - Convert list of dicts into a DataFrame
# - Drop rows with missing values to avoid problems in later analysis or ML models
features_df = pd.DataFrame(feature_rows).dropna()

# ==============================
# Train/test split
# ==============================


# --- Separate features (X) and labels (y) ---
# X = all extracted features except the 'label' column
# y = the target variable (gesture labels)
X = features_df.drop('label', axis=1)
y = features_df['label']

# --- Create a stratified train/test split ---
# StratifiedShuffleSplit ensures:
# - Train and test sets keep the same class proportion as the full dataset
# - Random split is reproducible with random_state=42
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

# --- Perform the split ---
# We only use one split (train/test), but StratifiedShuffleSplit could generate multiple
for train_idx, test_idx in sss.split(X, y):
    # Select training and testing samples based on generated indices
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]



# ==============================
# K-Nearest Neighbors training (uncomment to tune)
# ==============================


# Example hyperparameter grid for KNeighborsClassifier:
# - n_neighbors: number of neighbors to consider for each prediction
# - weights: determines how neighbors influence the prediction
#      'uniform' = all neighbors contribute equally
#      'distance' = closer neighbors have more influence than farther ones
# - p: distance metric used to measure distance between points
#      (1 = Manhattan distance, sum of absolute differences)
#      (2 = Euclidean distance, straight-line distance)


# # Uncomment and modify to perform hyperparameter tuning with GridSearchCV
# param_grid = {
#     'n_neighbors': [10],
#     'weights': ['uniform'],
#     'p': [1],
# }


# Reduced grid (empty, using default HGB parameters)
param_grid = {
    'n_neighbors': [10],
}

# Initialize KNN model with a base configuration
knn = KNeighborsClassifier() 

# Start timing the training process
start_time = time.time()

# Grid Search with cross-validation to find best hyperparameters
grid_search = GridSearchCV(
    estimator=knn,         # base KNN model
    param_grid=param_grid, # hyperparameter grid to search
    cv=5,                  # 5-fold cross-validation
    n_jobs=-1,             # use all available CPU cores
    scoring='accuracy',    # evaluation metric
    verbose=0              # verbosity level (0 = silent)
)

# Fit the model using the training data
grid_search.fit(X_train, y_train)

# End timing
end_time = time.time()

# Retrieve the best model found during grid search
best_model = grid_search.best_estimator_
print("Best params:", grid_search.best_params_)

# Print training time
print(f"Training completed in {end_time - start_time:.6f} seconds")


# ==============================
# Evaluation
# ==============================

# Ensure X_test is in a contiguous NumPy array format
# This can help avoid potential issues with certain scikit-learn functions
X_test = np.ascontiguousarray(X_test)


# --- Measure prediction time ---
# Record start time, run predictions, record end time
start_pred = time.time()
y_pred = best_model.predict(X_test)
end_pred = time.time()

# Compute average prediction time per sample (in milliseconds)
prediction_time = (end_pred - start_pred) / len(X_test) * 1000

# --- Evaluate model accuracy ---
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.2%}")
print(f"Average prediction time: {prediction_time:.3f} ms per gesture")

# --- Print a detailed classification report ---
# This includes precision, recall, F1-score, and support per class
print("\nClassification report:")
print(classification_report(y_test, y_pred))


# --- Function to plot a normalized confusion matrix ---
def plot_confusion_matrix_percent(y_true, y_pred, labels):
    """
    Plot a confusion matrix normalized as percentages.

    Parameters:
        y_true (array-like): True class labels
        y_pred (array-like): Predicted class labels
        labels (list): Ordered list of class labels to display
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100  # Normalize by row

    plt.figure(figsize=(8, 6))
    sns.set(font_scale=2.1)
    sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap='Blues',
                xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix (%)')
    plt.tight_layout()
    plt.show()


# --- Plot confusion matrix for all gesture labels ---
unique_labels = sorted(set(y))
plot_confusion_matrix_percent(y_test, y_pred, unique_labels)


# --- Identify misclassified samples ---
errors = (y_test != y_pred)
X_test_errors = X_test[errors]   # Features of wrongly predicted samples
y_test_errors = y_test[errors]   # True labels
y_pred_errors = y_pred[errors]   # Model’s predictions

print(f"Number of errors: {len(y_test_errors)} out of {len(y_test)} samples")
for i in range(len(y_test_errors)):
    print(f"  - Sample {i+1}: True = {y_test_errors.iloc[i]}, Predicted = {y_pred_errors[i]}")


# --- Function to summarize classification errors ---
def get_classification_errors_table(y_true, y_pred):
    """
    Create a DataFrame summarizing misclassifications.

    For each true class, shows:
    - Which other class it was misclassified as
    - Number of such errors
    - Percentage of errors relative to that true class
    """
    cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
    error_table = []

    for i, true_label in enumerate(unique_labels):
        for j, pred_label in enumerate(unique_labels):
            if i != j and cm[i, j] > 0:  # Only record misclassifications
                total_true = cm[i].sum()
                error_rate = cm[i, j] / total_true * 100
                error_table.append({
                    'True class': true_label,
                    'Predicted class': pred_label,
                    'Number of errors': cm[i, j],
                    'Error percentage': round(error_rate, 2)
                })

    return pd.DataFrame(error_table)


# --- Display the classification error table ---
error_df = get_classification_errors_table(y_test, y_pred)
print("\nClassification errors table:")
print(error_df)


# ==============================
# Save model (uncomment to save)
# ==============================

# # Save the trained model to a file using joblib
# # Uncomment the following lines to save the model
# joblib.dump(best_model, 'knn_model.pkl')
# print("Model saved as 'knn_model.pkl'")