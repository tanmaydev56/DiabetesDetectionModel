
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import keras_tuner as kt

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, auc)
from sklearn.base import BaseEstimator, TransformerMixin

# MODIFICATION: Import SMOTE
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks, backend as K

import warnings

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
# Add this class and function definition
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2., alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
    def call(self, y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        return self.alpha * tf.pow(1 - pt, self.gamma) * bce

# Define the hybrid loss function
loss_fn = lambda y_true, y_pred: 0.5 * losses.binary_crossentropy(y_true, y_pred) + 0.5 * FocalLoss()(y_true, y_pred)
# Add this class definition near your other custom classes
class Attention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        n_features = input_shape[-1]
        # These two dense layers learn the importance scores
        self.tanh_dense = layers.Dense(n_features, activation='tanh')
        self.softmax_dense = layers.Dense(n_features, activation='softmax')
    def call(self, inputs):
        attention = self.tanh_dense(inputs)
        attention = self.softmax_dense(attention)
        # This multiplies the original features by their learned importance
        return inputs * attention

# MODIFICATION: Replaced original FeatureEngineer with the advanced version
class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Enhanced feature engineering with systematic interactions, age binning,
    and a symptom counter.
    """

    def __init__(self, interaction_pairs=True, age_binning=True, symptom_count=True):
        self.interaction_pairs = interaction_pairs
        self.age_binning = age_binning
        self.symptom_count = symptom_count
        self.binary_cols = None
        self.final_features = None

    def fit(self, X, y=None):
        # Identify binary symptom columns (excluding Gender)
        self.binary_cols = [col for col in X.columns if X[col].nunique() == 2 and col.lower() != 'gender']
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()

        # 1. Symptom Count
        if self.symptom_count and self.binary_cols:
            X_copy['Symptom_Count'] = X_copy[self.binary_cols].sum(axis=1)

        # 2. Age Binning
        if self.age_binning and 'Age' in X_copy.columns:
            bins = [16, 35, 50, 100]
            labels = ['Young', 'MiddleAged', 'Senior']
            X_copy['Age_Group'] = pd.cut(X_copy['Age'], bins=bins, labels=labels, right=False)
            # One-hot encode the new categorical feature
            X_copy = pd.get_dummies(X_copy, columns=['Age_Group'], prefix='Age', drop_first=True)

        # 3. Key Interaction Pairs
        if self.interaction_pairs and self.binary_cols:
            # Focus on interactions between the most commonly cited symptoms
            key_symptoms = ['Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness']
            for col1, col2 in combinations(key_symptoms, 2):
                if col1 in X_copy.columns and col2 in X_copy.columns:
                    X_copy[f'{col1}_{col2}'] = X_copy[col1] * X_copy[col2]

        if self.final_features is None:
            self.final_features = X_copy.columns.tolist()

        # Ensure consistency in columns for test set
        X_copy = X_copy.reindex(columns=self.final_features, fill_value=0)

        return X_copy


# --- Custom Keras Components (Unchanged) ---
class MultiHeadAttention(layers.Layer):
    def __init__(self, n_heads=4, **kwargs):
        super().__init__(**kwargs)
        self.n_heads = n_heads

    def build(self, input_shape):
        n_features = input_shape[-1]
        self.head_dim = n_features // self.n_heads
        if self.head_dim == 0:
            self.head_dim = 1

        self.W_queries = self.add_weight(shape=(self.n_heads, n_features, self.head_dim), initializer='glorot_uniform',
                                         trainable=True)
        self.W_keys = self.add_weight(shape=(self.n_heads, n_features, self.head_dim), initializer='glorot_uniform',
                                      trainable=True)
        self.W_values = self.add_weight(shape=(self.n_heads, n_features, self.head_dim), initializer='glorot_uniform',
                                        trainable=True)
        self.W_out = self.add_weight(shape=(self.n_heads * self.head_dim, n_features), initializer='glorot_uniform',
                                     trainable=True)

    def call(self, inputs):
        inputs_reshaped = tf.expand_dims(inputs, axis=1)  # (batch, 1, features)

        queries = tf.einsum('bni,hij->bhnj', inputs_reshaped, self.W_queries)
        keys = tf.einsum('bni,hij->bhnj', inputs_reshaped, self.W_keys)
        values = tf.einsum('bni,hij->bhnj', inputs_reshaped, self.W_values)

        d_k = tf.cast(tf.shape(keys)[-1], tf.float32)
        scores = tf.einsum('bhnj,bhmj->bhnm', queries, keys) / tf.math.sqrt(d_k)
        attention_weights = tf.nn.softmax(scores, axis=-1)

        head_output = tf.einsum('bhnm,bhmj->bhnj', attention_weights, values)

        concatenated = tf.reshape(head_output, [-1, self.n_heads * self.head_dim])
        output = tf.matmul(concatenated, self.W_out)
        return output


class AdaptiveFocalLoss(losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.reshape(y_true, tf.shape(y_pred))
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        modulating_factor = tf.pow(1.0 - p_t, self.gamma)
        return tf.reduce_mean(alpha_factor * modulating_factor * bce)

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma, "alpha": self.alpha})
        return config


class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision(thresholds=threshold)
        self.recall = tf.keras.metrics.Recall(thresholds=threshold)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + K.epsilon()))

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()


# --- Model Building and Training ---

# MODIFICATION: Model creation function now accepts hyperparameters as arguments
def create_advanced_model(input_shape, hp):
    """Creates the model using a hyperparameters object from KerasTuner."""
    inputs = layers.Input(shape=input_shape)

    x = Attention()(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(hp.get('dropout_rate'))(x)

    residual = layers.Dense(x.shape[-1])(inputs)
    x = layers.Add()([x, residual])

    for i in range(hp.get('num_dense_layers')):
        x = layers.Dense(units=hp.get(f'dense_{i}_units'),
                         activation='relu',
                         kernel_regularizer=tf.keras.regularizers.l2(hp.get('l2_reg')))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(hp.get('dropout_rate'))(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, outputs)

    optimizer = optimizers.AdamW(
        learning_rate=hp.get('learning_rate'),
        weight_decay=hp.get('l2_reg')
    )
    model.compile(
        optimizer=optimizer,
        loss = loss_fn,
        metrics=[
            'accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            F1Score(name='f1_score')
        ]
    )
    return model


# MODIFICATION: KerasTuner model builder function
def build_tuner_model(hp):
    """Defines the search space for KerasTuner."""
    input_shape = (X_train.shape[1],)

    # Define search space
    # REMOVED n_heads
    hp_dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    hp_l2_reg = hp.Choice('l2_reg', values=[1e-2, 1e-3, 1e-4])
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4])

    # ADJUSTED the search for layer sizes to be closer to the successful model
    hp_num_layers = hp.Int('num_dense_layers', 2, 3)  # Search between 2 and 3 layers
    hp.Choice('dense_0_units', values=[128, 256, 384])
    hp.Choice('dense_1_units', values=[128, 256, 384])
    hp.Choice('dense_2_units', values=[256, 512])  # Only if num_layers is 3

    # REMOVED gamma and alpha as they are in the new loss_fn

    class HParams:
        def get(self, key):
            return hp.get(key)

    return create_advanced_model(input_shape, HParams())

# MODIFICATION: Training function updated to use SMOTE and best HPs
def train_with_cv(X, y, best_hps, n_splits=4, epochs=150, batch_size=24):
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X,y.ravel())):
        print(f"\nTraining fold {fold + 1}/{n_splits}")

        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Apply SMOTE to the training data of the current fold
        print("Applying SMOTE to training data...")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_fold, y_train_fold)
        print(f"Original training shape: {X_train_fold.shape}, Resampled shape: {X_train_resampled.shape}")

        # Create model using the best hyperparameters found by the tuner
        model = create_advanced_model((X_train_resampled.shape[1],), best_hps)

        callbacks_list = [
            callbacks.EarlyStopping(monitor='val_auc', patience=25, mode='max', restore_best_weights=True),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-7),
            callbacks.ModelCheckpoint(f'best_model_fold_{fold + 1}.keras', monitor='val_auc', save_best_only=True,
                                      mode='max')
        ]

        model.fit(
            X_train_resampled, y_train_resampled,
            validation_data=(X_val_fold, y_val_fold),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks_list,
            verbose=1
        )

        val_loss, val_acc, val_auc, val_prec, val_rec, val_f1 = model.evaluate(X_val_fold, y_val_fold, verbose=0)
        cv_scores.append(val_auc)
        print(f"Fold {fold + 1} - Val AUC: {val_auc:.4f}, Val F1: {val_f1:.4f}")

    print(f"\nCross-validation completed. Mean AUC: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")
    return cv_scores


# --- Evaluation and Prediction ---

def comprehensive_evaluation(y_test, y_pred_prob):
    """Evaluates model using probability scores."""
    y_pred = (y_pred_prob > 0.5).astype(int)
    print("🔍 Comprehensive Model Evaluation")
    print("=" * 50)
    print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
    print(f"AUC      : {roc_auc_score(y_test, y_pred_prob):.4f}")
    print("\n" + "=" * 50)
    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Negative', 'Positive']))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    plt.title('Confusion Matrix');
    plt.ylabel('True Label');
    plt.xlabel('Predicted Label')
    plt.show()

    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc(fpr, tpr):.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('Receiver Operating Characteristic (ROC) Curve');
    plt.xlabel('False Positive Rate');
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right");
    plt.show()


# MODIFICATION: New function to predict using the ensemble of CV models
def predict_with_ensemble(X, n_splits=6):
    """Loads all fold models and averages their predictions."""
    all_predictions = []
    # UPDATE this dictionary to include ALL your current custom classes
    custom_objects = {
        'Attention': Attention,       # <-- Add this
        'FocalLoss': FocalLoss,       # <-- Add this
        'F1Score': F1Score
    }
    for i in range(1, n_splits + 1):
        # The load_model call will now work correctly

        model = models.load_model(f'best_model_fold_{i}.keras', custom_objects=custom_objects, safe_mode=False)
        all_predictions.append(model.predict(X))

    # Return the mean of the predictions
    return np.mean(all_predictions, axis=0)

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load and preprocess data
    print("📊 Loading and preprocessing data...")
    df = pd.read_csv('diabetes_data_upload.csv')
    df.replace({'Yes': 1, 'No': 0, 'Positive': 1, 'Negative': 0, 'Male': 1, 'Female': 0}, inplace=True)

    # 2. MODIFICATION: Advanced Feature Engineering
    print("🔧 Performing advanced feature engineering...")
    feature_engineer = AdvancedFeatureEngineer()
    X_engineered = feature_engineer.fit_transform(df.drop('class', axis=1))
    y = df['class'].values.reshape(-1, 1)

    feature_names = X_engineered.columns.tolist()
    X = X_engineered.astype('float32').values
    print(f"Data shape after feature engineering: {X.shape}")

    # 3. Split data (initial train/test split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 4. Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 5. MODIFICATION: Hyperparameter Tuning with KerasTuner
    print("\n🔍 Starting hyperparameter tuning with KerasTuner...")
    tuner = kt.Hyperband(
        build_tuner_model,
        objective=kt.Objective("val_auc", direction="max"),
        max_epochs=50,
        factor=3,
        directory='keras_tuner_dir',
        project_name='diabetes_tuning'
    )
    # Use a subset of training data for faster tuning
    X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    tuner.search(X_train_sub, y_train_sub, epochs=50, validation_data=(X_val_sub, y_val_sub),
                 callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=10)])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("\n✨ Best hyperparameters found:")
    for param, value in best_hps.values.items():
        print(f"  - {param}: {value}")

    # 6. Train final model with Cross-Validation, SMOTE, and Best HPs
    print("\n🤖 Training final model with cross-validation using best hyperparameters...")
    N_SPLITS = 6
    train_with_cv(X_train, y_train, best_hps, n_splits=N_SPLITS, epochs=150)

    # 7. MODIFICATION: Evaluate using the ensemble
    print("\n📈 Evaluating ENSEMBLE model performance on the test set...")
    ensemble_predictions = predict_with_ensemble(X_test, n_splits=N_SPLITS)
    comprehensive_evaluation(y_test.ravel(), ensemble_predictions.ravel())

    # # 8. Save artifacts for deployment
    # print("\n💾 Saving final artifacts...")
    # joblib.dump(scaler, 'final_scaler.pkl')
    # joblib.dump(feature_engineer, 'final_feature_engineer.pkl')

    print("\n✅ Process completed successfully!")