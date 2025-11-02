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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from tensorflow.keras import layers, models, optimizers, losses, callbacks
from tensorflow.keras.regularizers import l2

warnings.filterwarnings('ignore')

def load_and_prepare_data(file_path='diabetes_data_upload.csv'):
    print("Step 1: Loading and Preparing Data...")
    df = pd.read_csv(file_path)
    df.replace({'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0, 'Positive': 1, 'Negative': 0}, inplace=True)
    scaler = MinMaxScaler()
    df['Age'] = scaler.fit_transform(df[['Age']])
    df['Age_Polyuria'] = df['Age'] * df['Polyuria']
    df['Age_Polydipsia'] = df['Age'] * df['Polydipsia']
    df['Gender_Age'] = df['Gender'] * df['Age']
    df['Polyuria_Polydipsia'] = df['Polyuria'] * df['Polydipsia']
    df['Gender_Polyuria'] = df['Gender'] * df['Polyuria']
    df['BMI_Proxy'] = df['Age'] * df['Gender']
    symptom_columns = ['Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia',
                      'Genital thrush', 'visual blurring', 'Itching', 'Irritability',
                      'delayed healing', 'partial paresis', 'muscle stiffness', 'Alopecia', 'Obesity']
    df['Symptom_Count'] = df[symptom_columns].sum(axis=1)
    X = df.drop('class', axis=1).astype('float32').values
    y = df['class'].astype('float32').values
    feature_names = list(df.drop('class', axis=1).columns)
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y.astype(int))}")
    return X, y, feature_names, scaler

@tf.keras.utils.register_keras_serializable()
class AdvancedAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def build(self, input_shape):
        n = input_shape[-1]
        self.tanh_dense = layers.Dense(n, activation='tanh', name='attn_tanh')
        self.sigmoid_dense = layers.Dense(n, activation='sigmoid', name='attn_gate')
        super().build(input_shape)
    def call(self, inputs):
        attn_weights = self.tanh_dense(inputs)
        gate_weights = self.sigmoid_dense(inputs)
        attended = inputs * attn_weights * gate_weights
        return inputs + 0.3 * attended

@tf.keras.utils.register_keras_serializable()
class FocalLoss(losses.Loss):
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
        config.update({'gamma': self.gamma, 'alpha': self.alpha})
        return config

@tf.keras.utils.register_keras_serializable()
class CombinedLoss(losses.Loss):
    def __init__(self, bce_weight=0.3, focal_weight=0.7, **kwargs):
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
        config.update({'bce_weight': self.bce_weight, 'focal_weight': self.focal_weight})
        return config

def build_enhanced_model(input_dim):
    learning_rate = 2e-4
    units_layer1 = 256
    units_layer2 = 128
    units_layer3 = 64
    units_layer4 = 32
    dropout_rate1 = 0.3
    dropout_rate2 = 0.25
    dropout_rate3 = 0.2
    dropout_rate4 = 0.15
    l2_reg = 8e-5
    inp = layers.Input(shape=(input_dim,))
    x = inp
    x1 = layers.Dense(units_layer1, activation='swish',
                      kernel_regularizer=l2(l2_reg),
                      bias_regularizer=l2(l2_reg))(x)
    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Dropout(dropout_rate1)(x1)
    x2 = layers.Dense(units_layer2, activation='swish',
                      kernel_regularizer=l2(l2_reg),
                      bias_regularizer=l2(l2_reg))(x1)
    x2 = layers.BatchNormalization()(x2)
    x2 = layers.Dropout(dropout_rate2)(x2)
    x3 = layers.Dense(units_layer3, activation='swish',
                      kernel_regularizer=l2(l2_reg),
                      bias_regularizer=l2(l2_reg))(x2)
    x3 = layers.BatchNormalization()(x3)
    x3 = layers.Dropout(dropout_rate3)(x3)
    x4 = layers.Dense(units_layer4, activation='swish',
                      kernel_regularizer=l2(l2_reg))(x3)
    x4 = layers.BatchNormalization()(x4)
    x4 = layers.Dropout(dropout_rate4)(x4)
    attention = AdvancedAttention()(x4)
    concatenated = layers.Concatenate()([x1, x2, x3, attention])
    x = layers.Dense(64, activation='swish')(concatenated)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.15)(x)
    x = layers.Dense(32, activation='swish')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.1)(x)
    out = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inp, out)
    combined_loss = CombinedLoss(bce_weight=0.3, focal_weight=0.7)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=combined_loss,
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    print("\nEnhanced model built for 96-97% accuracy target.")
    model.summary()
    return model

def train_enhanced_model_advanced(X, y):
    print("\nStep 2: Training Enhanced Model for 96-97% Accuracy...")
    cw_full = class_weight.compute_class_weight('balanced', classes=np.unique(y), y=y)
    cw_full = dict(enumerate(cw_full))
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    model = build_enhanced_model(X_train.shape[1])
    early_stop = callbacks.EarlyStopping(
        monitor='val_auc',
        patience=100,
        restore_best_weights=True,
        mode='max',
        verbose=1
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_auc',
        factor=0.5,
        patience=35,
        min_lr=1e-7,
        mode='max',
        verbose=1
    )
    checkpoint = callbacks.ModelCheckpoint(
        'best_model.h5',
        monitor='val_auc',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    print("Training enhanced model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=250,
        batch_size=16,
        class_weight=cw_full,
        callbacks=[early_stop, reduce_lr, checkpoint],
        verbose=1
    )
    model.load_weights('best_model.h5')
    return model, history, X_train, X_val, y_train, y_val

def explain_model_with_shap_fast(model, X_train, feature_names, background_size=100, explanation_size=200):
    print('\nStep 3: Generating FAST SHAP explanations...')
    X_df = pd.DataFrame(X_train, columns=feature_names)
    print(f"Features in explanation: {feature_names}")
    print(f"Number of features: {len(feature_names)}")
    if len(X_df) > background_size:
        background = X_df.sample(background_size, random_state=42).values
    else:
        background = X_df.values
    if len(X_df) > explanation_size:
        explanation_data = X_df.sample(explanation_size, random_state=42).values
    else:
        explanation_data = X_df.values
    print(f"Background data shape: {background.shape}")
    print(f"Explanation data shape: {explanation_data.shape}")
    try:
        print("Creating SHAP explainer...")
        explainer = shap.DeepExplainer(model, background)
        print("Calculating SHAP values...")
        shap_values = explainer.shap_values(explanation_data)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        print(f"SHAP values shape: {shap_values.shape}")
        print(f"SHAP values range: [{shap_values.min():.3f}, {shap_values.max():.3f}]")
        explanation_df = pd.DataFrame(explanation_data, columns=feature_names)
        print(f"Features in explanation_df: {list(explanation_df.columns)}")
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, explanation_df, show=False)
        plt.title(f'SHAP Feature Importance Summary ({len(feature_names)} features)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, explanation_df, plot_type="bar", show=False)
        plt.title(f'SHAP Mean Absolute Feature Importance ({len(feature_names)} features)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.show()
        return shap_values, explanation_data
    except Exception as e:
        print(f"Error in SHAP explanation: {e}")
        print("Trying alternative SHAP method...")
        try:
            background_small = X_df.sample(min(50, len(X_df)), random_state=42)
            explain_data_small = X_df.sample(min(20, len(X_df)), random_state=42)
            explainer = shap.KernelExplainer(model.predict, background_small)
            shap_values_small = explainer.shap_values(explain_data_small)
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values_small, explain_data_small, show=False)
            plt.title('SHAP Feature Importance (Kernel Explainer)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            return shap_values_small, explain_data_small.values
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            return None, None

def predict_custom_input_enhanced(model, scaler, feature_names):
    print("\n" + "=" * 50)
    print(" DIABETES PREDICTION")
    print("=" * 50)
    print("Please enter the following values (Y/N for Yes/No, M/F for Gender):")
    user_input = {}
    base_features = [f for f in feature_names if not f.startswith(('Age_', 'Gender_', 'Polyuria_', 'BMI_Proxy', 'Symptom_Count'))]
    for feature in base_features:
        while True:
            try:
                if feature == 'Age':
                    age = float(input(f"Enter {feature} (years): "))
                    user_input[feature] = scaler.transform([[age]])[0][0]
                elif feature == 'Gender':
                    val = input(f"Enter {feature} (M/F): ").strip().upper()
                    if val not in ['M', 'F']:
                        raise ValueError("Invalid input. Please enter M or F.")
                    user_input[feature] = 1 if val == 'M' else 0
                else:
                    val = input(f"Enter '{feature}' (Y/N): ").strip().upper()
                    if val not in ['Y', 'N']:
                        raise ValueError("Invalid input. Please enter Y or N.")
                    user_input[feature] = 1 if val == 'Y' else 0
                break
            except (ValueError, KeyboardInterrupt) as e:
                if isinstance(e, KeyboardInterrupt):
                    print("\nPrediction cancelled.")
                    return
                print("Invalid input. Please try again.")
    user_input['Age_Polyuria'] = user_input['Age'] * user_input.get('Polyuria', 0)
    user_input['Age_Polydipsia'] = user_input['Age'] * user_input.get('Polydipsia', 0)
    user_input['Gender_Age'] = user_input['Gender'] * user_input['Age']
    user_input['Polyuria_Polydipsia'] = user_input.get('Polyuria', 0) * user_input.get('Polydipsia', 0)
    user_input['Gender_Polyuria'] = user_input['Gender'] * user_input.get('Polyuria', 0)
    user_input['BMI_Proxy'] = user_input['Age'] * user_input['Gender']
    symptom_features = ['Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia',
                       'Genital thrush', 'visual blurring', 'Itching', 'Irritability',
                       'delayed healing', 'partial paresis', 'muscle stiffness', 'Alopecia', 'Obesity']
    user_input['Symptom_Count'] = sum(user_input.get(f, 0) for f in symptom_features)
    input_array = np.array([[user_input.get(f, 0) for f in feature_names]], dtype=np.float32)
    probability = model.predict(input_array, verbose=0)[0][0]
    print("\n" + "=" * 20 + " ENHANCED RESULTS " + "=" * 20)
    print(f"Probability of Diabetes: {probability:.2%}")
    if probability > 0.7:
        print("Prediction: 🟥 HIGH RISK - DIABETES (Positive)")
        print("\n🚨 CLINICAL NOTE: High probability detected.")
        print("   Please consult a healthcare professional immediately.")
    elif probability > 0.5:
        print("Prediction: 🟨 MODERATE RISK - Borderline")
        print("\n⚠️  CLINICAL NOTE: Moderate risk detected.")
        print("   Recommend follow-up testing and consultation.")
    elif probability > 0.3:
        print("Prediction: 🟦 LOW RISK - Likely Negative")
        print("\n💡 CLINICAL NOTE: Low risk detected.")
        print("   Maintain healthy lifestyle with regular check-ups.")
    else:
        print("Prediction: 🟩 VERY LOW RISK - Negative")
        print("\n✅ CLINICAL NOTE: Very low risk detected.")
        print("   Continue healthy habits and preventive care.")
    print("=" * 58)

def main():
    X, y, feature_names, scaler = load_and_prepare_data()
    print(f"\nDEBUG: Feature names: {feature_names}")
    print(f"DEBUG: Number of features: {len(feature_names)}")
    print(f"DEBUG: X shape: {X.shape}")
    print("\nTraining enhanced model for 96-97% accuracy target...")
    final_model, history, X_train, X_val, y_train, y_val = train_enhanced_model_advanced(X, y)
    print("\nMODEL PERFORMANCE (On Unseen Validation Data):")
    y_pred_prob = final_model.predict(X_val, verbose=0).ravel()
    y_pred = (y_pred_prob > 0.5).astype(int)
    print("=" * 50)
    print(f"{'Accuracy':<12}: {accuracy_score(y_val, y_pred):.4f}")
    print(f"{'Precision':<12}: {precision_score(y_val, y_pred):.4f}")
    print(f"{'Recall':<12}: {recall_score(y_val, y_pred):.4f}")
    print(f"{'F1 Score':<12}: {f1_score(y_val, y_pred):.4f}")
    print(f"{'AUC':<12}: {roc_auc_score(y_val, y_pred_prob):.4f}")
    print("=" * 50)
    explain_model_with_shap_fast(final_model, X_train, feature_names)
    artifacts = {'scaler': scaler, 'feature_names': feature_names}
    final_model.save('diabetes_model_enhanced.h5')
    joblib.dump(artifacts, 'model_artifacts_enhanced.pkl')
    print("\nEnhanced pipeline completed successfully!")
    return final_model, artifacts

if __name__ == "__main__":
    trained_model, artifacts = main()
    while True:
        try:
            response = input("\nMake a custom prediction? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                predict_custom_input_enhanced(trained_model, artifacts['scaler'], artifacts['feature_names'])
            elif response in ['n', 'no']:
                print("Exiting enhanced prediction tool. Goodbye!")
                break
            else:
                print("Invalid choice. Please enter 'y' or 'n'.")
        except KeyboardInterrupt:
            print("\nExiting. Goodbye!")
            break