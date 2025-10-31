import joblib
import pandas as pd
import numpy as np
import tensorflow as tf
import shap
import warnings
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from sklearn.metrics import (accuracy_score, precision_score,recall_score, f1_score, roc_auc_score)
from tensorflow.keras import layers, models, optimizers, losses, callbacks
from tensorflow.keras.regularizers import l2
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import warnings

warnings.filterwarnings('ignore')


def load_and_prepare_data(file_path='pima.csv', test_size=0.2, random_state=42):
    """
    Load and prepare Pima Indians Diabetes dataset following best practices.

    Parameters:
    - file_path: Path to the CSV file
    - test_size: Proportion of dataset to include in test split
    - random_state: Random seed for reproducibility

    Returns:
    - X_train, X_test, y_train, y_test: Train-test splits
    - feature_names: List of feature names
    - scaler: Fitted scaler for potential future use
    """

    print("Step 1: Loading and Cleaning Pima Data...")

    # Load data
    df = pd.read_csv(file_path)
    print(f"Original dataset shape: {df.shape}")

    # Data Quality Assessment
    print("\nData Quality Assessment:")
    print(f"Missing values:\n{df.isnull().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()}")

    # Create a copy to avoid modifying original data
    df_clean = df.copy()

    # 1. Handle Invalid Zero Values (Medical Domain Knowledge)
    invalid_zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    # Replace physiologically impossible zeros with NaN
    for col in invalid_zero_cols:
        zero_count = (df_clean[col] == 0).sum()
        print(f"Invalid zeros in {col}: {zero_count}")
        df_clean[col] = df_clean[col].replace(0, np.nan)

    # 2. Advanced Missing Value Imputation
    print("\nMissing values after zero replacement:")
    print(df_clean[invalid_zero_cols].isnull().sum())

    # Use KNN imputer for better missing value handling
    imputer = KNNImputer(n_neighbors=5)
    df_clean[invalid_zero_cols] = imputer.fit_transform(df_clean[invalid_zero_cols])

    # 3. Outlier Detection and Treatment
    def handle_outliers_iqr(df, columns):
        """Handle outliers using IQR method with capping"""
        df_out = df.copy()
        for col in columns:
            Q1 = df_out[col].quantile(0.25)
            Q3 = df_out[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # Count outliers
            outliers_low = (df_out[col] < lower_bound).sum()
            outliers_high = (df_out[col] > upper_bound).sum()
            print(f"Outliers in {col}: Lower={outliers_low}, Upper={outliers_high}")

            # Cap outliers
            df_out[col] = np.clip(df_out[col], lower_bound, upper_bound)

        return df_out

    numerical_cols = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                      'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

    df_clean = handle_outliers_iqr(df_clean, numerical_cols)

    # 4. Feature Engineering (Domain-Informed)
    print("\nFeature Engineering...")

    # BMI categories (medical relevance)
    df_clean['BMI_Category'] = pd.cut(df_clean['BMI'],
                                      bins=[0, 18.5, 25, 30, 100],
                                      labels=['Underweight', 'Normal', 'Overweight', 'Obese'])

    # Age groups
    df_clean['Age_Group'] = pd.cut(df_clean['Age'],
                                   bins=[0, 30, 45, 60, 100],
                                   labels=['Young', 'Middle', 'Senior', 'Elderly'])

    # Medical interaction terms
    df_clean['Glucose_BMI_Interaction'] = df_clean['Glucose'] * df_clean['BMI']
    df_clean['BP_Age_Interaction'] = df_clean['BloodPressure'] * df_clean['Age']
    df_clean['Insulin_Glucose_Ratio'] = df_clean['Insulin'] / (df_clean['Glucose'] + 1e-8)

    # 5. Encode Categorical Variables
    categorical_cols = ['BMI_Category', 'Age_Group']

    # One-hot encoding for categorical variables
    df_encoded = pd.get_dummies(df_clean, columns=categorical_cols, drop_first=True)

    # 6. Prepare Features and Target
    target = 'Outcome'
    X = df_encoded.drop(columns=[target])
    y = df_encoded[target].astype('float32')

    # 7. Train-Test Split (BEFORE scaling to avoid data leakage)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=y  # Important for imbalanced datasets
    )

    print(f"\nTrain set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Class distribution - Train: {np.bincount(y_train.astype(int))}")
    print(f"Class distribution - Test: {np.bincount(y_test.astype(int))}")

    # 8. Feature Scaling (Fit on train, transform both)
    scaler = StandardScaler()  # Better for neural networks than MinMax

    # Scale numerical features (exclude already encoded categoricals)
    numerical_features = [col for col in X_train.columns if col not in
                          [c for c in X_train.columns if any(x in c for x in ['Category', 'Group'])]]

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[numerical_features] = scaler.fit_transform(X_train[numerical_features])
    X_test_scaled[numerical_features] = scaler.transform(X_test[numerical_features])

    # Convert to float32 for better performance with deep learning
    X_train_scaled = X_train_scaled.astype('float32')
    X_test_scaled = X_test_scaled.astype('float32')
    y_train = y_train.astype('float32')
    y_test = y_test.astype('float32')

    feature_names = list(X_train_scaled.columns)

    print(f"\nFinal feature names ({len(feature_names)}):")
    for feature in feature_names:
        print(f"  - {feature}")

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler


# Additional utility function for deep learning preparation
def prepare_for_deep_learning(X_train, X_test, y_train, y_test):
    """
    Convert data to formats suitable for deep learning frameworks.
    """
    # For TensorFlow/Keras, the data is already in the right format
    # For PyTorch, you might want to convert to tensors

    print(f"\nDeep Learning Preparation:")
    print(f"X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
    print(f"X_test shape: {X_test.shape}, dtype: {X_test.dtype}")
    print(f"y_train shape: {y_train.shape}, dtype: {y_train.dtype}")
    print(f"y_test shape: {y_test.shape}, dtype: {y_test.dtype}")

    return X_train, X_test, y_train, y_test


# Example usage:


# 2.  Custom Layers and Loss Functions
@tf.keras.utils.register_keras_serializable()
class Attention(layers.Layer):
    """ self-attention layer with residual connection."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        n = input_shape[-1]
        self.tanh_dense = layers.Dense(n, activation='tanh', name='attn_tanh')
        self.softmax_dense = layers.Dense(n, activation='softmax', name='attn_softmax')
        super().build(input_shape)

    def call(self, inputs):
        attn_weights = self.tanh_dense(inputs)
        attn_weights = self.softmax_dense(attn_weights)
        attended = inputs * attn_weights
        # Residual connection
        return inputs + attended


@tf.keras.utils.register_keras_serializable()
class FocalLoss(losses.Loss):
    """ Focal Loss with dynamic alpha."""

    def __init__(self, gamma=2., alpha=0.25, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        y_true = tf.cast(y_true, dtype=y_pred.dtype)

        bce = tf.keras.losses.binary_crossentropy(y_true, y_pred, from_logits=False)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)

        alpha_factor = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)
        focal_loss = alpha_factor * tf.pow(1. - pt, self.gamma) * bce

        return focal_loss

    def get_config(self):
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha
        })
        return config


@tf.keras.utils.register_keras_serializable()
class CombinedLoss(losses.Loss):
    """Combined loss function with BCE and Focal Loss."""

    def __init__(self, bce_weight=0.4, focal_weight=0.6, **kwargs):
        super().__init__(**kwargs)
        self.bce_weight = bce_weight
        self.focal_weight = focal_weight
        self.bce_loss = losses.BinaryCrossentropy()
        self.focal_loss = FocalLoss(gamma=2.0, alpha=0.25)

    def call(self, y_true, y_pred):
        bce = self.bce_loss(y_true, y_pred)
        focal = self.focal_loss(y_true, y_pred)
        return self.bce_weight * bce + self.focal_weight * focal

    def get_config(self):
        config = super().get_config()
        config.update({
            'bce_weight': self.bce_weight,
            'focal_weight': self.focal_weight
        })
        return config



# 3. Enhanced Model Building Function

def build_model(input_dim):
    """Builds an enhanced neural network model for higher AUC."""
    #  hyperparameters taken from keras tunner (algorithm)
    learning_rate = 3e-4
    units_layer1 = 128
    units_layer2 = 64
    units_layer3 = 32
    dropout_rate1 = 0.4
    dropout_rate2 = 0.3
    dropout_rate3 = 0.2
    l2_reg = 1e-4

    inp = layers.Input(shape=(input_dim,))
    x = inp

    # Layer 1 with residual connection
    x1 = layers.Dense(units_layer1, activation='relu',
                      kernel_regularizer=l2(l2_reg))(x)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Dropout(dropout_rate1)(x1)

    # Layer 2
    x2 = layers.Dense(units_layer2, activation='relu',
                      kernel_regularizer=l2(l2_reg))(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Dropout(dropout_rate2)(x2)

    # Layer 3
    x3 = layers.Dense(units_layer3, activation='relu',
                      kernel_regularizer=l2(l2_reg))(x2)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Dropout(dropout_rate3)(x3)

    # Attention mechanism
    attention = Attention()(x3)

    # Concatenate features from different layers
    concatenated = layers.Concatenate()([x1, x2, attention])

    # Final layers
    x = layers.Dense(32, activation='relu')(concatenated)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)

    out = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inp, out)

    # Use the registered CombinedLoss class
    combined_loss = CombinedLoss(bce_weight=0.4, focal_weight=0.6)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=combined_loss,
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )

    print("\n model built with optimized architecture.")
    model.summary()
    return model



# 4.  Training with  Regularization

def train_enhanced_model(X, y):
    """Trains the  model with advanced regularization techniques."""
    print("\nStep 2: Training Enhanced Model...")

    # Computing class weights
    cw_full = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
    cw_full = dict(enumerate(cw_full))

    # Used stratified split for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Build model
    model = build_model(X_train.shape[1])

    # Enhanced callbacks with improved settings
    early_stop = callbacks.EarlyStopping(
        monitor='val_auc',
        patience=60,
        restore_best_weights=True,
        mode='max',
        verbose=1
    )

    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_auc',
        factor=0.5,
        patience=25,
        min_lr=1e-7,
        mode='max',
        verbose=1
    )

    # Training with enhanced parameters
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=400,
        batch_size=16,  # Smaller batch size for better generalization
        class_weight=cw_full,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )

    return model, history

# 5. Enhanced SHAP Explanation
def explain_model_with_shap(model, X, feature_names):
    """Generates and displays SHAP summary plots."""
    print('\nStep 3: Generating SHAP explanations...')
    X_df = pd.DataFrame(X, columns=feature_names)

    # Use smaller sample for efficiency
    sample_size = min(150, len(X_df))
    background = X_df.sample(sample_size, random_state=42)
    explanation_data = X_df.sample(min(200, len(X_df)), random_state=42)

    model_for_shap = lambda x: model.predict(x.astype(np.float32), verbose=0)
    explainer = shap.KernelExplainer(model_for_shap, background)
    shap_values = explainer.shap_values(explanation_data)

    # Plot summary
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, explanation_data, show=False)
    plt.title('Enhanced SHAP Feature Importance Summary', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # Plot bar summary
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, explanation_data, plot_type="bar", show=False)
    plt.title('Enhanced SHAP Mean Absolute Feature Importance', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()


def main():

    X, y, feature_names, scaler = load_and_prepare_data()

    # Train single enhanced model
    print("\nTraining single enhanced model...")
    final_model, history = train_enhanced_model(X, y)

    # Evaluate model performance
    y_pred_prob = final_model.predict(X, verbose=0).ravel()
    y_pred = (y_pred_prob > 0.5).astype(int)

    # Enhanced performance metrics
    print("\nMODEL PERFORMANCE:")
    print("=" * 50)
    print(f"{'Accuracy':<12}: {accuracy_score(y, y_pred):.4f}")
    print(f"{'Precision':<12}: {precision_score(y, y_pred):.4f}")
    print(f"{'Recall':<12}: {recall_score(y, y_pred):.4f}")
    print(f"{'F1 Score':<12}: {f1_score(y, y_pred):.4f}")
    print(f"{'AUC':<12}: {roc_auc_score(y, y_pred_prob):.4f}")
    print("=" * 50)

    # Explain model with SHAP
    explain_model_with_shap(final_model, X, feature_names)


    print("\n  pipeline completed successfully!")
    return final_model


# 8. Enhanced Prediction Function
def predict_custom_input_pima(model, artifacts):
    """Interactive CLI to predict for Pima features (numeric inputs)."""
    scaler = artifacts['scaler']
    feature_names = artifacts['feature_names']  # will be the Pima columns

    print("\nEnter numeric values for each feature (press Ctrl+C to cancel):")
    user_vals = []
    for f in feature_names:
        while True:
            try:
                val = float(input(f"{f}: ").strip())
                user_vals.append(val)
                break
            except ValueError:
                print("Invalid number. Try again.")

    # scale and reshape
    input_array = scaler.transform([user_vals]).astype(np.float32)
    prob = model.predict(input_array, verbose=0)[0][0]

    print("\n" + "=" * 20 + " ENHANCED RESULTS " + "=" * 20)
    print(f"Probability of Diabetes: {prob:.2%}")
    if prob > 0.7:
        print("🟥 HIGH RISK - DIABETES (Positive). Consult a clinician.")
    elif prob > 0.5:
        print("🟨 MODERATE RISK - Borderline. Recommend follow-up testing.")
    elif prob > 0.3:
        print("🟦 LOW RISK - Likely Negative.")
    else:
        print("🟩 VERY LOW RISK - Negative.")
    print("=" * 58)


if __name__ == "__main__":

    trained_model, artifacts_file_path = main()

    # Load only preprocessing artifacts
    loaded_artifacts = joblib.load(artifacts_file_path)


    while True:
        try:
            response = input("\nMake a custom prediction? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                predict_custom_input_pima(trained_model, loaded_artifacts)
            elif response in ['n', 'no']:
                print("Exiting enhanced prediction tool. Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 'y' or 'n'.")
        except KeyboardInterrupt:
            print("\nExiting. Goodbye!")
            break