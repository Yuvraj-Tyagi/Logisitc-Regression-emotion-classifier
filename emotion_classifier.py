"""
Machine learning CT-1
Applying linear regression to classify negative emotions from EEG data.
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
from scipy import stats
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_DIR = r"C:\Users\Yuvraj\Academics\ML lab\deap-dataset\data_preprocessed_python"
N_EEG_CHANNELS = 32
FS = 128

#1. Emotion Mapping 


def map_emotion(valence, arousal, dominance):
    if valence >= 5:
        return None
    if arousal >= 5 and dominance >= 5:
        return "anger"
    if arousal >= 5 and dominance < 5:
        return "fear"
    if arousal < 3.5:
        return "sadness"
    # Medium arousal (3.5 <= arousal < 5)
    if dominance >= 5:
        return "disgust"
    return "frustration"


#2. Feature Extraction from EEG

def extract_band_power(signal, fs, band):
    low, high = band
    freqs, psd = welch(signal, fs=fs, nperseg=min(256, len(signal)))
    idx = np.where((freqs >= low) & (freqs <= high))[0]
    if len(idx) == 0:
        return 0.0
    return np.mean(psd[idx])


def extract_features(eeg_trial):
    bands = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta":  (13, 30),
        "gamma": (30, 45),
    }
    features = []
    for ch in range(eeg_trial.shape[0]):
        signal = eeg_trial[ch]
        # Statistical features
        features.append(np.mean(signal))
        features.append(np.std(signal))
        features.append(stats.skew(signal))
        features.append(stats.kurtosis(signal))
        # Band power features
        for band_range in bands.values():
            features.append(extract_band_power(signal, FS, band_range))
    return np.array(features)


#3. Load Data and Extract Features 
print("Loading DEAP dataset and extracting EEG features...")
all_features = []
all_labels = []
skipped = 0

for i in range(1, 33):
    filename = os.path.join(DATA_DIR, f"s{i:02d}.dat")
    with open(filename, "rb") as f:
        subject = pickle.load(f, encoding="latin1")

    data = subject["data"]      
    labels = subject["labels"]  

    for trial in range(40):
        valence = labels[trial, 0]
        arousal = labels[trial, 1]
        dominance = labels[trial, 2]

        emotion = map_emotion(valence, arousal, dominance)
        if emotion is None:
            skipped += 1
            continue

        eeg_data = data[trial, :N_EEG_CHANNELS, :] 
        feat = extract_features(eeg_data)
        all_features.append(feat)
        all_labels.append(emotion)

    if i % 8 == 0:
        print(f"  Processed {i}/32 participants...")

X = np.array(all_features)
y = np.array(all_labels)

print(f"\nTotal negative emotion samples: {len(y)} (skipped {skipped} positive samples)")
print(f"Feature vector size: {X.shape[1]} (32 channels x 9 features each)")
emotion_counts = pd.Series(y).value_counts()
print(f"Class distribution:\n{emotion_counts}\n")

# Class Distribution Bar Chart
plt.figure(figsize=(8, 5))
emotion_counts.plot(kind="bar", color=["blue", "red", "purple", "orange", "green"], edgecolor="black")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.title("Class Distribution - 5 Negative Emotions")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("class_distribution.png", dpi=150)
print("Class distribution chart saved to class_distribution.png")

#4. Train/Test Split + Scaling
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}\n")

#5. Train Logistic Regression
print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=2000, solver="lbfgs", C=1.0)
model.fit(X_train_scaled, y_train)

#6. Evaluate
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print(f"\nAccuracy: {accuracy:.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred))

#7. Confusion Matrix 
target_emotions = sorted(list(set(y)))
cm = confusion_matrix(y_test, y_pred, labels=target_emotions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
xticklabels=target_emotions, yticklabels=target_emotions)
plt.xlabel("Predicted")

plt.ylabel("Actual")
plt.title("Confusion Matrix - EEG Negative Emotion Classification (Logistic Regression)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
print("Confusion matrix saved to confusion_matrix.png")

#8. Feature Importance (Top 20)
feature_names = []
bands = ["delta", "theta", "alpha", "beta", "gamma"]
stat_names = ["mean", "std", "skew", "kurtosis"]
for ch in range(N_EEG_CHANNELS):
    for s in stat_names:
        feature_names.append(f"Ch{ch+1}_{s}")
    for b in bands:
        feature_names.append(f"Ch{ch+1}_{b}")

# Average absolute coefficient across all classes
avg_importance = np.mean(np.abs(model.coef_), axis=0)
top_idx = np.argsort(avg_importance)[-20:][::-1]

print("\nTop 20 Most Important Features:")

for idx in top_idx:
    print(f"  {feature_names[idx]:20s} importance: {avg_importance[idx]:.4f}")
