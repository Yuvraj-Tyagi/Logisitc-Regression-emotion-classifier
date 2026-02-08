"""
Machine learning CT-1
1. Problem Definition: Applying linear regression to classify negative emotions from EEG data.
Dataset: DEAP (Database for Emotion Analysis using Physiological Signals) from Kaggle
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

DATA_DIR = r"C:\Users\Yuvraj\Academics\ML lab\deap-dataset\data_preprocessed_python"
N_EEG_CHANNELS = 32

#mapping emotions
def map_emotion(valence, arousal, dominance):
    if valence >= 5:
        return None  # Ignoring positive emotions

    if arousal >= 5 and dominance >= 5:
        return "anger"
    elif arousal >= 5 and dominance < 5:
        return "fear"
    elif arousal < 3:
        return "sadness"
    elif dominance < 3:
        return "disgust"
    else:
        return "frustration"


#Feature extraction from EEG data 
def extract_features(eeg_trial):
    features = []
    for ch in range(eeg_trial.shape[0]):
        signal = eeg_trial[ch]
        features.append(np.mean(signal))
        features.append(np.std(signal))
    return np.array(features)

#Loading Dataset (2. Data collection)
print("Loading DEAP dataset...")
all_features = []
all_labels = []
skipped = 0

for i in range(1, 33):
    file_path = os.path.join(DATA_DIR, f"s{i:02d}.dat")
    with open(file_path, "rb") as f:
        subject = pickle.load(f, encoding="latin1")

    data = subject["data"]
    labels = subject["labels"]

    for trial in range(40):
        valence = labels[trial, 0]
        arousal = labels[trial, 1]
        dominance = labels[trial, 2]

        emotion = map_emotion(valence, arousal, dominance) # 3. Data Pre-processing
        if emotion is None:
            skipped += 1
            continue

        eeg_data = data[trial, :N_EEG_CHANNELS, :]
        features = extract_features(eeg_data)  # 4. Feature extraction

        all_features.append(features)
        all_labels.append(emotion)

    if i % 8 == 0:
        print(f"Processed {i}/32 participants")

X = np.array(all_features)
y = np.array(all_labels)

print(f"\nTotal samples used: {len(y)} (Skipped {skipped} positive samples)")
print(f"Feature vector size: {X.shape[1]} (32 channels × 2 features)")
print("\nClass distribution:")
print(pd.Series(y).value_counts())

# 5. Exploratory Data Analysis (EDA) - class distribution
plt.figure(figsize=(7, 5))
pd.Series(y).value_counts().plot(kind="bar", edgecolor="black")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.title("Class Distribution of 5 Negative Emotions")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150)

#Data splitting 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6 . model training 
print("\nTraining Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# 7. testing
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

#confusion matrix
labels_unique = np.unique(y)
cm = confusion_matrix(y_test, y_pred, labels=labels_unique)

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels_unique,
            yticklabels=labels_unique)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - 5 Negative Emotions")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)

print("\nConfusion matrix saved.")
