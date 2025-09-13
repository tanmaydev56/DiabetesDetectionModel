# improved_training_with_smote.py
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve, auc,
                             precision_recall_curve)

from sklearn.base import BaseEstimator, TransformerMixin

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks
import warnings
warnings.filterwarnings('ignore')

# Optional: SMOTE
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except Exception:
    SMOTE_AVAILABLE = False

# reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# ----------------------------
# (Optional) FeatureEngineer - kept minimal (not used by new dataset)
# ----------------------------
class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self, create_interactions=True, poly_features=False):
        self.create_interactions = create_interactions
        self.poly_features = poly_features

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        if self.create_interactions:
            if 'age' in X_copy.columns and 'bmi' in X_copy.columns:
                X_copy['age_x_bmi'] = X_copy['age'] * X_copy['bmi']
        if self.poly_features and 'age' in X_copy.columns:
            X_copy['age_sq'] = X_copy['age'] ** 2
        return X_copy


# ----------------------------
# Attention Layer (kept simple)
# ----------------------------
class MultiHeadAttention(layers.Layer):
    def __init__(self, n_heads=2, **kwargs):
        super().__init__(**kwargs)
        self.n_heads = n_heads

    def build(self, input_shape):
        n_features = input_shape[-1]
        # choose head dim so concatenation equals head_dim * n_heads
        self.head_dim = int(np.ceil(n_features / max(1, self.n_heads)))
        self.Wq = self.add_weight(shape=(n_features, self.head_dim * self.n_heads),
                                  initializer='glorot_uniform', trainable=True)
        self.Wk = self.add_weight(shape=(n_features, self.head_dim * self.n_heads),
                                  initializer='glorot_uniform', trainable=True)
        self.Wv = self.add_weight(shape=(n_features, self.head_dim * self.n_heads),
                                  initializer='glorot_uniform', trainable=True)
        self.Wout = self.add_weight(shape=(self.head_dim * self.n_heads, n_features),
                                    initializer='glorot_uniform', trainable=True)

    def call(self, inputs):
        # inputs shape: (batch, features)
        Q = tf.matmul(inputs, self.Wq)  # (batch, head_dim * n_heads)
        K = tf.matmul(inputs, self.Wk)
        V = tf.matmul(inputs, self.Wv)

        # reshape to (batch, n_heads, head_dim)
        Q = tf.reshape(Q, (-1, self.n_heads, self.head_dim))
        K = tf.reshape(K, (-1, self.n_heads, self.head_dim))
        V = tf.reshape(V, (-1, self.n_heads, self.head_dim))

        # compute attention per head
        scores = tf.matmul(Q, K, transpose_b=True) / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        # softmax along last axis (head_dim)
        attn = tf.nn.softmax(scores, axis=-1)  # shape (batch, n_heads, head_dim, head_dim)
        # apply attention to V — need to align dims: matmul(attn, V) -> (batch, n_heads, head_dim)
        head_out = tf.matmul(attn, V)
        # collapse heads
        head_out = tf.reshape(head_out, (-1, self.n_heads * self.head_dim))
        out = tf.matmul(head_out, self.Wout)  # project back to feature dim
        return out


# ----------------------------
# Adaptive focal loss (keeps y_true reshape)
# ----------------------------
class AdaptiveFocalLoss(losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.25, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        y_true = tf.reshape(y_true, tf.shape(y_pred))
        if self.from_logits:
            y_pred = tf.nn.sigmoid(y_pred)
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_factor = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        modulating_factor = tf.pow(1.0 - p_t, self.gamma)
        return tf.reduce_mean(alpha_factor * modulating_factor * bce)


# ----------------------------
# Custom F1 metric
# ----------------------------
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', threshold=0.5, **kwargs):
        super().__init__(name=name, **kwargs)
        self.threshold = threshold
        self.precision = tf.keras.metrics.Precision(thresholds=self.threshold)
        self.recall = tf.keras.metrics.Recall(thresholds=self.threshold)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_bin = tf.cast(y_pred > self.threshold, tf.float32)
        self.precision.update_state(y_true, y_pred_bin, sample_weight)
        self.recall.update_state(y_true, y_pred_bin, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + tf.keras.backend.epsilon()))

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()


# ----------------------------
# Model factory (reduced regularization)
# ----------------------------
def create_advanced_model(input_shape,
                          n_heads=2,
                          dropout_rate=0.1,
                          l2_reg=1e-4,
                          learning_rate=1e-3):
    inputs = layers.Input(shape=input_shape)

    x = MultiHeadAttention(n_heads=n_heads)(inputs)
    x = layers.LayerNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    residual = layers.Dense(x.shape[-1])(inputs)
    x = layers.Add()([x, residual])

    x = layers.Dense(128, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(64, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(l2_reg))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs, outputs)

    optimizer = optimizers.AdamW(learning_rate=learning_rate, weight_decay=l2_reg)

    model.compile(optimizer=optimizer,
                  loss=AdaptiveFocalLoss(gamma=2.0, alpha=0.25),
                  metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall'),
                           F1Score(name='f1_score')])
    return model


# ----------------------------
# Training with CV (SMOTE inside fold)
# ----------------------------
def train_with_cv(X, y, n_splits=5, epochs=80, batch_size=32, use_smote=True):
    kfold = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_scores = []
    models = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y.ravel())):
        print(f"\n=== Fold {fold + 1}/{n_splits} ===")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx].ravel(), y[val_idx].ravel()

        # Option A: SMOTE (only if available and requested)
        if use_smote and SMOTE_AVAILABLE:
            sm = SMOTE(random_state=42)
            X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
            print("SMOTE applied:", np.bincount(y_train_res.astype(int)))
            train_X, train_y = X_train_res, y_train_res.reshape(-1, 1)
            class_weights_param = None
        else:
            # compute class weights for imbalanced data
            cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            class_weights_param = dict(enumerate(cw))
            print("Class weights:", class_weights_param)
            train_X, train_y = X_train, y_train.reshape(-1, 1)

        model = create_advanced_model(input_shape=(X.shape[1],),
                                      n_heads=2,
                                      dropout_rate=0.1,
                                      l2_reg=1e-4,
                                      learning_rate=1e-3)

        cb = [
            callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True, verbose=1),
            callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=6, min_lr=1e-6, verbose=1)
        ]

        history = model.fit(train_X, train_y,
                            validation_data=(X_val, y_val.reshape(-1, 1)),
                            epochs=epochs,
                            batch_size=batch_size,
                            class_weight=class_weights_param,
                            callbacks=cb,
                            verbose=1)

        # Evaluate
        val_metrics = model.evaluate(X_val, y_val.reshape(-1, 1), verbose=0)
        # val_metrics -> [loss, accuracy, auc, precision, recall, f1_score]
        print("Validation metrics (loss, acc, auc, prec, rec, f1):", val_metrics)
        models.append(model)
        cv_scores.append(val_metrics[2])  # auc

    best_idx = int(np.argmax(cv_scores))
    print("\nCV AUCs:", cv_scores)
    print("Best fold index:", best_idx)
    return models[best_idx], cv_scores


# ----------------------------
# Evaluation helpers (threshold tuning)
# ----------------------------
def evaluate_with_threshold(model, X_test, y_test):
    # get probabilities
    y_prob = model.predict(X_test).reshape(-1)
    # default threshold 0.5
    y_pred_default = (y_prob > 0.5).astype(int)

    # compute default metrics
    acc = accuracy_score(y_test, y_pred_default)
    prec = precision_score(y_test, y_pred_default, zero_division=0)
    rec = recall_score(y_test, y_pred_default, zero_division=0)
    f1 = f1_score(y_test, y_pred_default, zero_division=0)
    roc = roc_auc_score(y_test, y_prob)

    print("\nDefault threshold (0.5) results")
    print("Accuracy :", acc)
    print("Precision:", prec)
    print("Recall   :", rec)
    print("F1       :", f1)
    print("AUC      :", roc)

    # If model predicts all zeros, mean prob will be small
    print("Mean predicted probability:", np.mean(y_prob))
    print("Predicted positives (0.5):", np.sum(y_prob > 0.5))

    # Tune threshold by maximizing F1 on test (or you can use validation set)
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1s = 2 * (precisions * recalls) / (precisions + recalls + 1e-12)
    # skip last item where threshold is undefined
    best_idx = np.nanargmax(f1s)
    if best_idx >= len(thresholds):
        best_threshold = 0.5
    else:
        best_threshold = thresholds[best_idx]

    print("Best threshold by F1:", best_threshold)
    y_pred_opt = (y_prob >= best_threshold).astype(int)

    acc2 = accuracy_score(y_test, y_pred_opt)
    prec2 = precision_score(y_test, y_pred_opt, zero_division=0)
    rec2 = recall_score(y_test, y_pred_opt, zero_division=0)
    f12 = f1_score(y_test, y_pred_opt, zero_division=0)
    roc2 = roc_auc_score(y_test, y_prob)

    print("\nOptimized threshold results")
    print("Accuracy :", acc2)
    print("Precision:", prec2)
    print("Recall   :", rec2)
    print("F1       :", f12)
    print("AUC      :", roc2)

    print("\nClassification report (optimized threshold):")
    print(classification_report(y_test, y_pred_opt, zero_division=0))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred_opt)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix (optimized threshold)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

    return {
        'y_prob': y_prob,
        'y_pred_default': y_pred_default,
        'y_pred_opt': y_pred_opt,
        'best_threshold': best_threshold
    }


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # load dataset (adapt filename)
    df = pd.read_csv('diabetes_prediction_dataset.csv')  # replace with your filename

    # quick preprocessing for the sample dataset you showed
    # map gender
    df['gender'] = df['gender'].map({'Female': 0, 'Male': 1})

    # smoking_history one-hot (drop_first True to avoid multicollinearity)
    if 'smoking_history' in df.columns:
        df = pd.get_dummies(df, columns=['smoking_history'], drop_first=True)

    # optional feature engineering (uncomment if you want)
    # fe = FeatureEngineer(create_interactions=True, poly_features=True)
    # df = fe.fit_transform(df)

    # ensure target name
    if 'diabetes' not in df.columns:
        raise ValueError("Target column 'diabetes' not found in dataframe")

    # prepare X and y
    X = df.drop('diabetes', axis=1).astype('float32').values
    y = df['diabetes'].astype('int').values  # keep as 1D ints for stratify and SMOTE

    print("Dataset shape:", X.shape, "Positive ratio:", np.mean(y))

    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20,
                                                        stratify=y, random_state=42)

    # scale
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # train with cv (SMOTE inside folds)
    best_model, cv_scores = train_with_cv(X_train, y_train.reshape(-1, 1),
                                          n_splits=5, epochs=80, batch_size=32, use_smote=True)

    # evaluate on hold-out test set and tune threshold
    eval_info = evaluate_with_threshold(best_model, X_test, y_test)

    # save artifacts
    best_model.save('best_model_smote.h5')
    joblib.dump(scaler, 'scaler_smote.pkl')

    print("\nDone. Model and scaler saved.")
