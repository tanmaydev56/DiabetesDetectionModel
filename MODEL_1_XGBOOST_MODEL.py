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

warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_and_preprocess_data(file_path='diabetes_data_upload.csv'):
    """
    Loads the new dataset, cleans it, and converts categorical
    features to a numerical format suitable for machine learning.
    """
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['class'].value_counts()}")

    # Convert binary categorical features to numerical format
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

    # Drop rows with any missing values
    df.dropna(inplace=True)

    print(f"Processed dataset shape: {df.shape}")
    print(f"Final class distribution:\n{df['class'].value_counts()}")

    X = df.drop('class', axis=1)
    y = df['class']
    feature_names = X.columns.tolist()
    class_names = ['No Diabetes', 'Diabetes']
    return X, y, feature_names, class_names


def feature_selection(X, y, k=10):
    """
    Selects the top 'k' most informative features for the model.
    """
    print(f"Performing feature selection to find the best {k} features...")
    selector = SelectKBest(f_classif, k=min(k, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    print("Selected Features:", selected_features.tolist())
    return X_selected, selected_features, selector


def train_model_with_smote(X_train, y_train):
    """
    Handles class imbalance with SMOTE and trains the XGBoost model
    using GridSearchCV to find the best hyperparameters.
    """
    print("\nBalancing training data with SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print(f"After SMOTE - Training set: {X_train_res.shape}")
    print(f"Class distribution after SMOTE: {np.bincount(y_train_res)}")

    print("Training XGBoost model with GridSearchCV...")
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.2],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0],
    }
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=3,  # Reduced for speed
        n_jobs=-1,
        scoring='roc_auc',
        verbose=1
    )
    grid_search.fit(X_train_res, y_train_res)
    print(f"\nBest parameters found: {grid_search.best_params_}")
    print(f"Best cross-validation AUC: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the final model's performance on the unseen test set.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n" + "=" * 50)
    print("MODEL EVALUATION ON TEST DATA")
    print("=" * 50)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"PR AUC: {average_precision_score(y_test, y_proba):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Plot built-in feature importance
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model, max_num_features=10, height=0.8)
    plt.title('Feature Importance (from XGBoost)')
    plt.tight_layout()
    plt.show()

    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred)
    }


def explain_model(model, X_train, X_test, feature_names):
    """
    Provides model explanations using the SHAP library.
    """
    print("\nGenerating SHAP explanations...")

    # Use smaller sample for faster computation
    if len(X_test) > 100:
        X_test_sample = X_test[:100]
    else:
        X_test_sample = X_test

    if len(X_train) > 100:
        X_train_sample = X_train[:100]
    else:
        X_train_sample = X_train

    try:
        # Convert to DataFrame for better SHAP output
        X_test_df = pd.DataFrame(X_test_sample, columns=feature_names)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_df)

        # Bar plot for global feature importance
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False)
        plt.title('Global Feature Importance (SHAP)')
        plt.tight_layout()
        plt.show()

        # Beeswarm plot for detailed feature importance
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test_df, show=False)
        plt.title('SHAP Feature Importance (Beeswarm)')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        print("Continuing without SHAP plots...")


def predict_custom_input(model, scaler, feature_names):
    """
    Allows the user to enter custom data for a real-time prediction.
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
                print("Please enter a valid number.")
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
        print("\n🔴 Recommendation: High risk of diabetes. It is strongly recommended to consult a doctor.")
    elif prob > 0.3:
        print("\n🟡 Recommendation: Moderate risk. Consider lifestyle changes and regular monitoring.")
    else:
        print("\n🟢 Recommendation: Low risk. Continue to maintain healthy habits.")
    print("=" * 50)


def run_one_fold(X, y, train_idx, test_idx, original_feature_names):
    """
    Executes everything for a single fold:
    feature selection → SMOTE → scaling → grid-search → evaluation
    """
    X_train_full, X_test_full = X.iloc[train_idx], X.iloc[test_idx]
    y_train_full, y_test_full = y.iloc[train_idx], y.iloc[test_idx]

    # Feature selection on THIS fold's training data
    selector = SelectKBest(f_classif, k=min(10, X_train_full.shape[1]))
    X_train_sel = selector.fit_transform(X_train_full, y_train_full)

    # Only transform test set if it has samples
    if len(test_idx) > 0:
        X_test_sel = selector.transform(X_test_full)
    else:
        X_test_sel = None

    selected_features = X_train_full.columns[selector.get_support()]

    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_sel, y_train_full)

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sm)

    if X_test_sel is not None:
        X_test_scaled = scaler.transform(X_test_sel)
    else:
        X_test_scaled = None

    # Grid-search XGBoost
    param_grid = {
        'max_depth': [3, 5],
        'learning_rate': [0.1, 0.2],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0]
    }
    xgb_clf = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42
    )
    grid = GridSearchCV(
        xgb_clf, param_grid,
        cv=3, scoring='roc_auc',
        n_jobs=-1, verbose=0
    )
    grid.fit(X_train_scaled, y_train_sm)
    best_model = grid.best_estimator_

    # Evaluation only if we have test data
    if X_test_scaled is not None and len(y_test_full) > 0:
        y_pred = best_model.predict(X_test_scaled)
        y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
        metrics = {
            'acc': accuracy_score(y_test_full, y_pred),
            'prec': precision_score(y_test_full, y_pred, zero_division=0),
            'rec': recall_score(y_test_full, y_pred, zero_division=0),
            'f1': f1_score(y_test_full, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test_full, y_prob),
            'pr_auc': average_precision_score(y_test_full, y_prob)
        }
    else:
        metrics = None

    return metrics, best_model, scaler, selector, selected_features


def train_final_model(X, y, feature_names):
    """
    Train final model on entire dataset without test split
    """
    print("\n" + "=" * 50)
    print("TRAINING FINAL MODEL ON FULL DATASET")
    print("=" * 50)

    # Feature selection on full data
    selector = SelectKBest(f_classif, k=min(10, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()

    print(f"Selected features for final model: {selected_features}")

    # SMOTE on full data
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_selected, y)

    print(f"After SMOTE - Dataset shape: {X_resampled.shape}")
    print(f"Class distribution after SMOTE: {np.bincount(y_resampled)}")

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)

    # Train final model with best parameters (simplified for speed)
    final_model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
        max_depth=5,
        learning_rate=0.1,
        n_estimators=200,
        subsample=0.8
    )

    final_model.fit(X_scaled, y_resampled)
    print("Final model trained successfully!")

    return final_model, scaler, selector, selected_features


def cv_main():
    """
    Main function with cross-validation and final model training
    """
    X, y, original_feature_names, class_names = load_and_preprocess_data()

    print("\n" + "=" * 60)
    print("STARTING 5-FOLD STRATIFIED CROSS-VALIDATION")
    print("=" * 60)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f'\n\n{"=" * 20} FOLD {fold}/5 {"=" * 20}')
        metrics, model, scaler, selector, features = run_one_fold(
            X, y, train_idx, test_idx, original_feature_names
        )

        if metrics is not None:
            fold_scores.append(metrics)
            print(f"Fold {fold} Results:")
            for k, v in metrics.items():
                print(f"  {k}: {v:.4f}")

    # Display cross-validation results
    if fold_scores:
        print('\n\n' + "=" * 60)
        print("5-FOLD CROSS-VALIDATION SUMMARY")
        print("=" * 60)

        df_cv = pd.DataFrame(fold_scores)
        print(df_cv)
        print('\nMean ± Standard Deviation:')
        for col in df_cv.columns:
            mean_val = df_cv[col].mean()
            std_val = df_cv[col].std()
            print(f"{col:8}: {mean_val:.4f} ± {std_val:.4f}")

    # Train final model on full dataset
    final_model, final_scaler, final_selector, final_features = train_final_model(X, y, original_feature_names)


    # Explain final model using a sample of the data
    print("\nGenerating explanations for final model...")
    X_selected = final_selector.transform(X)
    X_scaled = final_scaler.transform(X_selected)

    # Use a sample for SHAP to avoid memory issues
    if len(X_scaled) > 100:
        sample_idx = np.random.choice(len(X_scaled), 100, replace=False)
        X_sample = X_scaled[sample_idx]
    else:
        X_sample = X_scaled

    explain_model(final_model, X_scaled, X_sample, final_features)

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
    cv_main()