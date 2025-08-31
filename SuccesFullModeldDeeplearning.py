import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses, callbacks

import shap
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('diabetes_data_upload.csv')
df.replace({'Yes':1, 'No':0, 'Male':1, 'Female':0}, inplace=True)

# Normalize age
scaler = MinMaxScaler()
df['Age'] = scaler.fit_transform(df[['Age']])

# Prepare X and y
X = df.drop('class', axis=1).astype('float32').values
y = df['class'].map({'Positive': 1, 'Negative': 0}).astype('float32').values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(class_weights))

# --- Attention Layer ---
class Attention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        n_features = input_shape[-1]
        self.tanh_dense = layers.Dense(n_features, activation='tanh')
        self.softmax_dense = layers.Dense(n_features, activation='softmax')
    def call(self, inputs):
        attention = self.tanh_dense(inputs)
        attention = self.softmax_dense(attention)
        return inputs * attention

# Model
inputs = layers.Input(shape=(X_train.shape[1],))
x = Attention()(inputs)
x = layers.Dense(250, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(250, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(500, activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs, outputs)

# Focal loss
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2., alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
    def call(self, y_true, y_pred):
        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        return self.alpha * tf.pow(1 - pt, self.gamma) * bce

loss_fn = lambda y_true, y_pred: 0.5 * losses.binary_crossentropy(y_true, y_pred) + 0.5 * FocalLoss()(y_true, y_pred)

model.compile(optimizer=optimizers.Adam(1e-3),
              loss=loss_fn,
              metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])

early = callbacks.EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True)
rlrop = callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=15, min_lr=1e-5)

history = model.fit(X_train, y_train, validation_split=0.15, epochs=600,
                    batch_size=32, class_weight=class_weights,
                    callbacks=[early, rlrop], verbose=1)

# Evaluate
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)
# predict learning rate also
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1-score :", f1_score(y_test, y_pred))
print("AUC      :", roc_auc_score(y_test, y_pred_prob))

# --- SHAP explainability (for 1 sample)
print("\n📈 Running SHAP explainability...")

qs = list(df.drop('class', axis=1).columns)
X_train_df = pd.DataFrame(X_train, columns=qs)
background = X_train_df.sample(100, random_state=42)

model_for_shap = lambda x: model.predict(x.astype(np.float32)).flatten()
explainer = shap.KernelExplainer(model_for_shap, background)
patient = pd.DataFrame(X_test[:1], columns=qs)
shap_values = explainer.shap_values(patient)

shap.summary_plot(shap_values, patient, feature_names=qs)
shap.summary_plot(shap_values, patient, feature_names=qs, plot_type="bar")
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[0], patient.iloc[0])

model.save('diabetes_model.h5')
joblib.dump(scaler, 'age_scaler.pkl')
# # Optional: Run this at the end of the script for custom prediction
def custom_test():
    print("\n🧪 Enter patient details:")
    age = float(input("Age (20–100): "))
    age_scaled = (age - 20) / 80.0
    inputs = [age_scaled]
    questions = qs[1:]
    for q in questions:
        ans = input(f"{q} (Yes/No): ").strip().lower()
        inputs.append(1 if ans == 'yes' else 0)
    inputs = np.array([inputs], dtype=np.float32)
    prob = model.predict(inputs)[0][0]
    print(f"\n🔴 High Risk" if prob >= 0.5 else f"🟢 Low Risk",
          f"– probability: {prob*100:.1f}%")

# Run the live test
custom_test()
