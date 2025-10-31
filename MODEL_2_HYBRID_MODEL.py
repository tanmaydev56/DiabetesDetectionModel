import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, \
    average_precision_score, precision_score, recall_score, f1_score
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import joblib
import os
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import lightgbm as lgb
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_and_preprocess_data(file_path='diabetes_data_upload.csv'):
    """
    Loads and preprocesses the diabetes dataset with categorical features
    """
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Original class distribution:\n{df['class'].value_counts()}")

    # Convert categorical features to numerical
    for col in df.columns:
        if df[col].dtype == 'object':
            # Map Gender
            if 'Gender' in col:
                df[col] = df[col].map({'Male': 1, 'Female': 0})
            # Map the target variable 'class'
            elif 'class' in col:
                df[col] = df[col].map({'Positive': 1, 'Negative': 0})
            # Map all other 'Yes'/'No' columns
            else:
                df[col] = df[col].map({'Yes': 1, 'No': 0})

    # Drop any remaining missing values
    df.dropna(inplace=True)

    print(f"Processed dataset shape: {df.shape}")
    print(f"Final class distribution:\n{df['class'].value_counts()}")

    X = df.drop('class', axis=1)
    y = df['class']
    feature_names = X.columns.tolist()
    class_names = ['No Diabetes', 'Diabetes']

    return X, y, feature_names, class_names


def feature_selection(X, y, k=8):  # Reduced from 10 to 8 features
    """
    Selects the top 'k' most informative features
    """
    print(f"\nPerforming feature selection to find best {k} features...")
    selector = SelectKBest(f_classif, k=min(k, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    print(f"Selected features: {selected_features.tolist()}")
    return X_selected, selected_features, selector


def add_noise_to_features(X, noise_level=0.02):
    """
    Adds small random noise to features to prevent overfitting
    """
    noise = np.random.normal(0, noise_level, X.shape)
    return X + noise


def train_hybrid_model(X_train, y_train, use_smote=True):
    """
    Trains a hybrid ensemble model with regularization to reduce overfitting
    """
    if use_smote:
        print("\nBalancing training data with SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE - Training set: {X_train_res.shape}")
        print(f"Class distribution after SMOTE: {np.bincount(y_train_res)}")
    else:
        print("\nUsing original imbalanced data (no SMOTE)...")
        X_train_res, y_train_res = X_train, y_train

    # Add small noise to prevent overfitting
    X_train_res = add_noise_to_features(X_train_res, noise_level=0.01)

    # Base Models with STRONGER REGULARIZATION
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        eval_metric='logloss',
        max_depth=3,  # Reduced from 5
        learning_rate=0.05,  # Reduced from 0.1
        n_estimators=100,  # Reduced from 200
        subsample=0.7,  # Reduced from 0.8
        colsample_bytree=0.7,
        reg_alpha=1.0,  # Added L1 regularization
        reg_lambda=1.0  # Added L2 regularization
    )

    rf_model = RandomForestClassifier(
        n_estimators=100,  # Reduced from 200
        max_depth=8,  # Reduced from 10
        min_samples_split=10,  # Added regularization
        min_samples_leaf=5,  # Added regularization
        max_features='sqrt',  # Limit features
        random_state=42,
        bootstrap=True
    )

    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=0.01,  # Increased regularization (lower C)
        penalty='l2',
        solver='liblinear'
    )

    lgb_model = lgb.LGBMClassifier(
        random_state=42,
        n_estimators=100,  # Reduced from 150
        max_depth=3,  # Reduced from 5
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=1.0,
        reg_lambda=1.0
    )

    svm_model = SVC(
        probability=True,
        random_state=42,
        C=0.1,  # Increased regularization (lower C)
        kernel='rbf',
        gamma='scale'
    )

    # Hybrid Voting Classifier with fewer models
    hybrid_model = VotingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('rf', rf_model),
            ('lr', lr_model),
            # Removed lgb and svm to reduce overfitting
        ],
        voting='soft',
        n_jobs=-1
    )

    print("Training hybrid ensemble model with regularization...")
    hybrid_model.fit(X_train_res, y_train_res)
    return hybrid_model


def evaluate_model(model, X_test, y_test, feature_names, fold_num=None):
    """
    Comprehensive model evaluation with metrics
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    if fold_num:
        print(f"\n=== Fold {fold_num} Evaluation ===")
    else:
        print("\n=== Final Model Evaluation ===")

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    pr_auc = average_precision_score(y_test, y_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Return metrics for aggregation
    return {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }


def plot_feature_importance(model, feature_names):
    """
    Plots feature importance from the best tree-based model in the ensemble
    """
    plt.figure(figsize=(12, 6))

    # Check if it's a VotingClassifier
    if hasattr(model, 'estimators_') and isinstance(model.estimators_, list):
        tree_model = None
        model_name = None

        # Iterate through estimators safely
        for estimator_tuple in model.estimators_:
            if isinstance(estimator_tuple, tuple) and len(estimator_tuple) == 2:
                name, estimator = estimator_tuple
                if hasattr(estimator, 'feature_importances_'):
                    tree_model = estimator
                    model_name = name
                    break

        if tree_model is not None:
            importances = tree_model.feature_importances_
            indices = np.argsort(importances)[::-1]

            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
            plt.title(f'Feature Importances ({model_name.upper()})')
            plt.tight_layout()
            plt.show()
        else:
            print("No tree-based model found in ensemble for feature importance plot.")

    # For single tree-based models (XGBoost, RandomForest, etc.)
    elif hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.bar(range(len(importances)), importances[indices])
        plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
        plt.title('Feature Importances')
        plt.tight_layout()
        plt.show()

    else:
        print("Model doesn't support feature importance visualization.")


def explain_model_shap(model, X_train, X_test, feature_names):
    """
    Generates SHAP explanations for model interpretability
    """
    print("\nGenerating SHAP explanations...")

    try:
        # For VotingClassifier, use the first tree-based model for SHAP
        if hasattr(model, 'estimators_') and isinstance(model.estimators_, list):
            tree_model = None
            for estimator_tuple in model.estimators_:
                if isinstance(estimator_tuple, tuple) and len(estimator_tuple) == 2:
                    name, estimator = estimator_tuple
                    if hasattr(estimator, 'feature_importances_'):
                        tree_model = estimator
                        break

            if tree_model is not None:
                explainer = shap.TreeExplainer(tree_model)
                shap_values = explainer.shap_values(X_test)

                # Handle binary classification output
                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values_positive = shap_values[1]
                else:
                    shap_values_positive = shap_values

                # Summary plot
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values_positive, X_test, feature_names=feature_names, plot_type="bar",
                                  show=False)
                plt.title('Global Feature Importance (SHAP)')
                plt.tight_layout()
                plt.show()

        else:
            # For single tree-based models
            if hasattr(model, 'feature_importances_'):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test)

                if isinstance(shap_values, list) and len(shap_values) == 2:
                    shap_values_positive = shap_values[1]
                else:
                    shap_values_positive = shap_values

                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values_positive, X_test, feature_names=feature_names, plot_type="bar",
                                  show=False)
                plt.title('Global Feature Importance (SHAP)')
                plt.tight_layout()
                plt.show()

    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        print("Continuing without SHAP plots...")


def save_artifacts(model, scaler, feature_names, selector=None):
    """
    Saves model artifacts for deployment
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts = {
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'selector': selector,
        'timestamp': timestamp
    }
    filename = f'diabetes_hybrid_model_{timestamp}.pkl'
    joblib.dump(artifacts, filename)
    print(f"\n✅ Model artifacts saved as '{filename}'")
    return filename


def predict_custom_input(model, scaler, feature_names):
    """
    Interactive prediction for custom input
    """
    print("\n" + "=" * 50)
    print("CUSTOM PREDICTION")
    print("=" * 50)
    print("Please enter the following values:")
    print("For Gender: M for Male, F for Female")
    print("For other features: Y for Yes, N for No")
    print("-" * 50)

    user_input_dict = {}
    for feature in feature_names:
        while True:
            try:
                if feature == 'Age':
                    val = float(input(f"Enter value for {feature}: "))
                elif feature == 'Gender':
                    val = input(f"Enter value for {feature} (M for Male, F for Female): ").strip().upper()
                    if val == 'M':
                        val = 1
                    elif val == 'F':
                        val = 0
                    else:
                        print("Invalid input. Please enter M or F.")
                        continue
                else:
                    val = input(f"Enter value for {feature} (Y for Yes, N for No): ").strip().upper()
                    if val == 'Y':
                        val = 1
                    elif val == 'N':
                        val = 0
                    else:
                        print("Invalid input. Please enter Y or N.")
                        continue

                user_input_dict[feature] = val
                break
            except ValueError:
                print("Please enter a valid value.")
            except KeyboardInterrupt:
                print("\nPrediction cancelled.")
                return

    # Create DataFrame and scale
    input_df = pd.DataFrame([user_input_dict])[feature_names]
    user_scaled = scaler.transform(input_df)

    prediction = model.predict(user_scaled)[0]
    prob = model.predict_proba(user_scaled)[0][1]

    print("\n" + "=" * 50)
    print("PREDICTION RESULT")
    print("=" * 50)
    print(f"Predicted Class: {'🟥 DIABETES' if prediction == 1 else '🟩 NO DIABETES'}")
    print(f"Probability of Diabetes: {prob * 100:.2f}%")

    if prob > 0.7:
        print("\n🔴 Recommendation: High risk of diabetes. Consult a doctor immediately.")
    elif prob > 0.3:
        print("\n🟡 Recommendation: Moderate risk. Consider lifestyle changes and monitoring.")
    else:
        print("\n🟢 Recommendation: Low risk. Maintain healthy habits.")
    print("=" * 50)


def run_stratified_cv(X, y, feature_names, n_splits=5, use_smote=True):
    """
    Performs stratified k-fold cross-validation
    """
    print("\n" + "=" * 60)
    print("STARTING 5-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 60)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_metrics = []
    fold_models = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f'\n\n{"=" * 20} FOLD {fold}/{n_splits} {"=" * 20}')

        # Split data
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Feature selection for this fold
        X_train_sel, selected_features, selector = feature_selection(X_train, y_train, k=8)
        X_test_sel = selector.transform(X_test)

        # Scaling
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sel)
        X_test_scaled = scaler.transform(X_test_sel)

        # Convert to DataFrame for better feature names in SHAP
        X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=selected_features)
        X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=selected_features)

        # Train model
        model = train_hybrid_model(X_train_scaled_df, y_train, use_smote=use_smote)

        # Evaluate
        metrics = evaluate_model(model, X_test_scaled_df, y_test, selected_features, fold_num=fold)
        fold_metrics.append(metrics)
        fold_models.append((model, scaler, selector, selected_features))

        # Plot feature importance for this fold
        plot_feature_importance(model, selected_features)

    return fold_metrics, fold_models


def train_final_model(X, y, feature_names, use_smote=True):
    """
    Trains final model on entire dataset
    """
    print("\n" + "=" * 50)
    print("TRAINING FINAL MODEL ON FULL DATASET")
    print("=" * 50)

    # Feature selection on full data
    X_selected, selected_features, selector = feature_selection(X, y, k=8)

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    X_scaled_df = pd.DataFrame(X_scaled, columns=selected_features)

    # Train final model
    final_model = train_hybrid_model(X_scaled_df, y, use_smote=use_smote)

    return final_model, scaler, selector, selected_features


def main():
    """
    Main function with cross-validation and final model training
    """
    # Load and preprocess data
    X, y, original_feature_names, class_names = load_and_preprocess_data()

    # Ask user if they want to use SMOTE
    use_smote_input = input("\nUse SMOTE for balancing? (y/n, default=y): ").strip().lower()
    use_smote = use_smote_input not in ['n', 'no']

    # Perform 5-fold cross-validation
    cv_metrics, cv_models = run_stratified_cv(X, y, original_feature_names, n_splits=5, use_smote=use_smote)

    # Display cross-validation results
    print("\n\n" + "=" * 60)
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print("=" * 60)

    metrics_df = pd.DataFrame(cv_metrics)
    print(metrics_df)

    print("\nAverage Performance across 5 folds:")
    for metric in metrics_df.columns:
        mean_val = metrics_df[metric].mean()
        std_val = metrics_df[metric].std()
        print(f"{metric:12}: {mean_val:.4f} ± {std_val:.4f}")

    # Train final model on full dataset
    final_model, final_scaler, final_selector, final_features = train_final_model(X, y, original_feature_names,
                                                                                  use_smote=use_smote)

    # Create a proper test set for final evaluation
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Transform test data
    X_test_selected = final_selector.transform(X_test)
    X_test_scaled = final_scaler.transform(X_test_selected)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=final_features)

    # Evaluate final model on proper test set
    print("\n" + "=" * 50)
    print("FINAL MODEL EVALUATION ON HELD-OUT TEST SET")
    print("=" * 50)
    final_metrics = evaluate_model(final_model, X_test_scaled_df, y_test, final_features)

    # Generate explanations on test set
    X_final_selected = final_selector.transform(X_temp)
    X_final_scaled = final_scaler.transform(X_final_selected)
    X_final_scaled_df = pd.DataFrame(X_final_scaled, columns=final_features)

    explain_model_shap(final_model, X_final_scaled_df, X_test_scaled_df, final_features)

    # Save artifacts
    saved_filename = save_artifacts(final_model, final_scaler, final_features, final_selector)

    # Interactive prediction
    while True:
        try:
            response = input("\nWould you like to make a custom prediction? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                predict_custom_input(final_model, final_scaler, final_features)
            elif response in ['n', 'no']:
                print("Thank you for using the Diabetes Prediction System!")
                break
            else:
                print("Please enter 'y' or 'n'.")
        except KeyboardInterrupt:
            print("\n\nExiting. Thank you!")
            break


if __name__ == "__main__":
    main()