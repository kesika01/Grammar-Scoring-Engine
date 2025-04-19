# === 0. Essential Libraries ===
import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

# === 1. Load and Inspect Dataset ===
def load_dataset():
    train_data = pd.read_csv('/kaggle/input/shl-intern-hiring-assessment/Dataset/train.csv')
    test_data = pd.read_csv('/kaggle/input/shl-intern-hiring-assessment/Dataset/test.csv')

    print("Label distribution in training set:")
    print(train_data['label'].value_counts().sort_index())
    
    return train_data, test_data

train_data, test_data = load_dataset()

# === 2. Audio Feature Engineering ===
def extract_audio_features(audio_file_path):
    try:
        waveform, sample_rate = librosa.load(audio_file_path, sr=None)

        extracted_features = {
            'audio_duration': librosa.get_duration(y=waveform, sr=sample_rate),
            'zero_crossing_mean': librosa.feature.zero_crossing_rate(waveform).mean(),
            'rms_energy_mean': librosa.feature.rms(y=waveform).mean(),
            'spectral_centroid_avg': librosa.feature.spectral_centroid(y=waveform, sr=sample_rate).mean(),
            'spectral_bandwidth_avg': librosa.feature.spectral_bandwidth(y=waveform, sr=sample_rate).mean(),
            'spectral_rolloff_avg': librosa.feature.spectral_rolloff(y=waveform, sr=sample_rate).mean(),
        }

        mfcc_features = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=20)
        for idx in range(mfcc_features.shape[0]):
            extracted_features[f'mfcc_{idx}_avg'] = mfcc_features[idx].mean()
            extracted_features[f'mfcc_{idx}_std'] = mfcc_features[idx].std()

        chroma_features = librosa.feature.chroma_stft(y=waveform, sr=sample_rate)
        for idx in range(chroma_features.shape[0]):
            extracted_features[f'chroma_{idx}_avg'] = chroma_features[idx].mean()

        spectral_contrast = librosa.feature.spectral_contrast(y=waveform, sr=sample_rate)
        for idx in range(spectral_contrast.shape[0]):
            extracted_features[f'spectral_contrast_{idx}_avg'] = spectral_contrast[idx].mean()

        tonal_features = librosa.feature.tonnetz(y=waveform, sr=sample_rate)
        for idx in range(tonal_features.shape[0]):
            extracted_features[f'tonnetz_{idx}_avg'] = tonal_features[idx].mean()

        onset_envelope = librosa.onset.onset_strength(y=waveform, sr=sample_rate)
        extracted_features['tempo_estimation'] = librosa.feature.rhythm.tempo(onset_envelope=onset_envelope, sr=sample_rate)[0]

        harmonic_part, percussive_part = librosa.effects.hpss(waveform)
        extracted_features['harmonic_mean'] = np.mean(harmonic_part)
        extracted_features['percussive_mean'] = np.mean(percussive_part)

        amplitude_db = librosa.amplitude_to_db(np.abs(librosa.stft(waveform)))
        extracted_features['silence_proportion'] = np.mean(amplitude_db < -50)

        return pd.Series(extracted_features)
    
    except Exception as error:
        print(f"Error processing {audio_file_path}: {error}")
        return pd.Series([0] * 150)

# === 3. Parallelized Feature Extraction ===
def extract_features_in_parallel(df, base_dir):
    audio_paths = [os.path.join(base_dir, fname) for fname in df['filename']]
    all_features = Parallel(n_jobs=-1, verbose=1)(delayed(extract_audio_features)(path) for path in audio_paths)
    return pd.concat(all_features, axis=1).T

train_features = extract_features_in_parallel(train_data, '/kaggle/input/shl-intern-hiring-assessment/Dataset/audios/train/')
test_features = extract_features_in_parallel(test_data, '/kaggle/input/shl-intern-hiring-assessment/Dataset/audios/test/')
target_values = train_data['label'].values

# === 4. Feature Refinement ===
def remove_low_info_features(feature_df, correlation_threshold=0.95):
    constant_cols = feature_df.columns[feature_df.nunique() == 1]
    reduced_df = feature_df.drop(columns=constant_cols)

    correlation_matrix = reduced_df.corr().abs()
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    redundant_cols = [col for col in upper_triangle.columns if any(upper_triangle[col] > correlation_threshold)]

    cleaned_df = reduced_df.drop(columns=redundant_cols)
    return cleaned_df

train_features = remove_low_info_features(train_features)
test_features = test_features[train_features.columns]

# === 5. Feature Normalization ===
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(train_features)
X_test_scaled = scaler.transform(test_features)

# === 6. Data Splitting ===
X_train_split, X_validation, y_train_split, y_validation = train_test_split(
    X_train_scaled, target_values, test_size=0.15, random_state=42, stratify=target_values)

# === 7. Train Models and Select the Best ===
def train_and_select_model(X, y):
    candidate_models = {
        'XGBoost': XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=5,
                                subsample=0.8, colsample_bytree=0.8, random_state=42, eval_metric='rmse'),
        'LightGBM': LGBMRegressor(n_estimators=1000, learning_rate=0.05, max_depth=7,
                                  num_leaves=31, subsample=0.8, colsample_bytree=0.8, random_state=42),
        'RandomForest': RandomForestRegressor(n_estimators=500, max_depth=15,
                                              min_samples_split=5, min_samples_leaf=2,
                                              max_features='sqrt', random_state=42, n_jobs=-1)
    }

    best_model_name = None
    lowest_rmse = float('inf')
    for name, estimator in candidate_models.items():
        cv_scores = cross_val_score(estimator, X, y, scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1)
        avg_rmse = -np.mean(cv_scores)
        print(f"{name} CV RMSE: {avg_rmse:.4f}")
        if avg_rmse < lowest_rmse:
            lowest_rmse = avg_rmse
            best_model_name = name

    print(f"\nSelected Model: {best_model_name} with RMSE: {lowest_rmse:.4f}")
    final_model = candidate_models[best_model_name]
    final_model.fit(X, y)
    return final_model

final_estimator = train_and_select_model(X_train_split, y_train_split)

# === 8. Model Evaluation on Validation Set ===
def validate_model(model, X_val, y_val):
    predictions = model.predict(X_val)
    rmse_score = np.sqrt(mean_squared_error(y_val, predictions))
    mae_score = mean_absolute_error(y_val, predictions)
    classification_accuracy = np.mean(np.round(predictions) == y_val)

    print("\nValidation Performance:")
    print(f"RMSE: {rmse_score:.4f}")
    print(f"MAE: {mae_score:.4f}")
    print(f"Rounded Accuracy: {classification_accuracy:.2%}")

    # Visualization
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.regplot(x=y_val, y=predictions, scatter_kws={'alpha': 0.3})
    plt.plot([1, 5], [1, 5], 'r--')
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Prediction vs Actual")

    plt.subplot(1, 2, 2)
    sns.histplot(y_val - predictions, kde=True)
    plt.xlabel("Residuals")
    plt.title("Error Distribution")

    plt.tight_layout()
    plt.show()
    return rmse_score

rmse_validation = validate_model(final_estimator, X_validation, y_validation)

# === 9. Feature Importance Visuals ===
def visualize_feature_importance(model, feature_labels, top_features=20):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_features:]
        plt.figure(figsize=(10, 8))
        plt.title(f"Top {top_features} Feature Importances")
        plt.barh(range(top_features), importances[indices], align='center')
        plt.yticks(range(top_features), [feature_labels[i] for i in indices])
        plt.xlabel("Importance")
        plt.tight_layout()
        plt.show()
    else:
        print("Model doesn't provide feature importances.")

visualize_feature_importance(final_estimator, train_features.columns)

# === 10. Generate Test Set Predictions ===
test_predictions = final_estimator.predict(X_test_scaled)
test_predictions = np.clip(test_predictions, 1, 5)

# === 11. Save Submission File ===
submission_output = pd.DataFrame({
    'filename': test_data['filename'],
    'label': test_predictions
})
submission_output.to_csv('/kaggle/working/submission_enhanced.csv', index=False)
print("✅ Submission file created at /kaggle/working/submission_enhanced.csv")
