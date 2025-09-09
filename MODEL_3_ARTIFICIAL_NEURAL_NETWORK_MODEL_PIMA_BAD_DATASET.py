#  TRAINING ON PIMA DATASET
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, losses
import warnings
warnings.filterwarnings('ignore')

# 1. Load raw PIMA -----------------------------------------------------------
df_raw = pd.read_csv('pima.csv')

# 2. Sex-stratified zero-value imputation ------------------------------------
def impute_zeros(df):
    """Replace 0 in metabolic cols with sex-stratified median."""
    for col in ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']:
        df[col] = df.groupby('Outcome')[col].transform(
            lambda x: x.mask(x == 0).fillna(x.median()))
    return df

df = impute_zeros(df_raw.copy())

# 3. Features & label ---------------------------------------------------------
X = df[['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']].astype('float32')
y = df['Outcome'].astype('float32')

# 4. Hold-out test (30 %) -----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42)

# 5. Standardise --------------------------------------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 6. Class-weighted focal loss ------------------------------------------------
def focal_loss(y_true, y_pred):
    gamma, alpha = 2.0, 0.75
    bce = losses.binary_crossentropy(y_true, y_pred)
    pt  = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
    return alpha * tf.pow(1 - pt, gamma) * bce

# 7. Build model --------------------------------------------------------------
model = models.Sequential([
    layers.Input(shape=(8,)),
    layers.Dense(32, activation='relu'),
    layers.Dense(16, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss=focal_loss, metrics=['accuracy'])

# 8. Training -----------------------------------------------------------------
cw = class_weight.compute_class_weight(
    'balanced', classes=np.unique(y_train), y=y_train)
cw = dict(enumerate(cw))

early = callbacks.EarlyStopping(
    monitor='val_loss', patience=50, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    validation_split=0.15,
    epochs=1000,
    batch_size=32,
    class_weight=cw,
    callbacks=[early],
    verbose=0
)

# 9. Final evaluation ---------------------------------------------------------
y_prob = model.predict(X_test).ravel()
y_pred = (y_prob > 0.5).astype(int)

print("\n📊 Final Model Performance (Hold-out Set):")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
print(f"F1-score : {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC  : {roc_auc_score(y_test, y_prob):.4f}")

# 10. Learning rate info ------------------------------------------------------
lr = model.optimizer.learning_rate.numpy()
print(f"\n🔹 Final Learning Rate: {lr:.6f}")
