import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, \
    average_precision_score
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

warnings.filterwarnings("ignore", category=UserWarning)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_and_preprocess_data():
    df = pd.read_csv('pima.csv')
    cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in cols_with_zero:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df.groupby('Outcome')[col].transform(lambda x: x.fillna(x.median()))
    df['Glucose_BMI_Ratio'] = df['Glucose'] / df['BMI']
    df['Age_Glucose_Interaction'] = df['Age'] * df['Glucose']
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    feature_names = X.columns.tolist()
    class_names = ['No Diabetes', 'Diabetes']
    return X, y, feature_names, class_names


def feature_selection(X, y, k=8):
    selector = SelectKBest(f_classif, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    return X_selected, selected_features


def train_hybrid_model(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # Base Models
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        eval_metric='logloss'
    )
    rf_model = RandomForestClassifier(random_state=42)
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lgb_model = lgb.LGBMClassifier(random_state=42)
    svm_model = SVC(probability=True, random_state=42)

    # Hybrid Voting Classifier
    hybrid_model = VotingClassifier(
        estimators=[
            ('xgb', xgb_model),
            ('rf', rf_model),
            ('lr', lr_model),
            ('lgb', lgb_model),
            ('svm', svm_model)
        ],
        voting='soft'  # uses probability averaging
    )

    hybrid_model.fit(X_train_res, y_train_res)
    return hybrid_model


def evaluate_model(model, X_test, y_test, feature_names):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    print("\n=== Model Evaluation ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
    print(f"PR AUC: {average_precision_score(y_test, y_proba):.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Create a figure for feature importance
    plt.figure(figsize=(10, 6))

    # For VotingClassifier, we'll use one of the tree-based models for feature importance
    if hasattr(model, 'estimators_'):
        # Find the first tree-based model with feature_importances_
        tree_model = None
        for estimator in model.estimators_:
            if hasattr(estimator, 'feature_importances_'):
                tree_model = estimator
                break

        if tree_model is not None:
            importances = tree_model.feature_importances_
            indices = np.argsort(importances)[::-1]

            # Plot feature importances
            plt.title('Feature Importances')
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.show()
    else:
        # For single models that have feature_importances_
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]

            plt.title('Feature Importances')
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.show()

    return y_pred, y_proba


def explain_model(model, X_train, X_test, feature_names):
    print("\nGenerating SHAP explanations...")

    # For VotingClassifier, we need to use one of the tree-based models for SHAP
    if hasattr(model, 'estimators_'):
        # Find the first tree-based model
        tree_model = None
        for estimator in model.estimators_:
            if hasattr(estimator, 'feature_importances_'):
                tree_model = estimator
                break

        if tree_model is not None:
            # Create a SHAP explainer
            explainer = shap.TreeExplainer(tree_model)
            shap_values = explainer.shap_values(X_test)

            # Plot summary
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar")
            plt.title('Feature Importance (SHAP)')
            plt.tight_layout()
            plt.show()

            # Plot waterfall for a sample
            sample_idx = 0
            plt.figure(figsize=(10, 6))
            if isinstance(shap_values, list) and len(shap_values) == 2:
                # For binary classification, shap_values is a list of two arrays
                shap_values_sample = shap_values[1][sample_idx]  # Use the positive class
            else:
                shap_values_sample = shap_values[sample_idx]

            shap.plots.waterfall(shap.Explanation(values=shap_values_sample,
                                                  base_values=explainer.expected_value,
                                                  data=X_test.iloc[sample_idx],
                                                  feature_names=feature_names))
            plt.title(f'Prediction Explanation for Sample {sample_idx}')
            plt.tight_layout()
            plt.show()
    else:
        # For single tree-based models
        if hasattr(model, 'feature_importances_'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)

            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, plot_type="bar")
            plt.title('Feature Importance (SHAP)')
            plt.tight_layout()
            plt.show()

            sample_idx = 0
            plt.figure(figsize=(10, 6))
            if isinstance(shap_values, list) and len(shap_values) == 2:
                # For binary classification, shap_values is a list of two arrays
                shap_values_sample = shap_values[1][sample_idx]  # Use the positive class
            else:
                shap_values_sample = shap_values[sample_idx]

            shap.plots.waterfall(shap.Explanation(values=shap_values_sample,
                                                  base_values=explainer.expected_value,
                                                  data=X_test.iloc[sample_idx],
                                                  feature_names=feature_names))
            plt.title(f'Prediction Explanation for Sample {sample_idx}')
            plt.tight_layout()
            plt.show()


# def save_artifacts(model, scaler, feature_names):
#     import datetime
#     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     artifacts = {
#         'model': model,
#         'scaler': scaler,
#         'feature_names': feature_names,
#         'timestamp': timestamp
#     }
#     joblib.dump(artifacts, f'diabetes_model_{timestamp}.pkl')
#     print(f"\nArtifacts saved as diabetes_model_{timestamp}.pkl")


def predict_custom_input(model, scaler, feature_names):
    print("\n=== Custom Prediction ===")
    user_input = []
    for feature in feature_names:
        while True:
            try:
                val = float(input(f"Enter value for {feature}: "))
                user_input.append(val)
                break
            except ValueError:
                print("Please enter a valid number")
    user_scaled = scaler.transform([user_input])
    prediction = model.predict(user_scaled)[0]
    prob = model.predict_proba(user_scaled)[0][1]
    print("\n=== Prediction Result ===")
    print(f"Predicted Class: {'Diabetes' if prediction == 1 else 'No Diabetes'}")
    print(f"Probability of Diabetes: {prob * 100:.2f}%")
    if prob > 0.7:
        print("High risk of diabetes - consult a doctor")
    elif prob > 0.3:
        print("Moderate risk - consider lifestyle changes")
    else:
        print("Low risk - maintain healthy habits")


def main():
    X, y, feature_names, class_names = load_and_preprocess_data()
    X_selected, selected_features = feature_selection(X, y)
    feature_names = selected_features.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    model = train_hybrid_model(X_train_scaled_df, y_train)
    evaluate_model(model, X_test_scaled_df, y_test, feature_names)
    explain_model(model, X_train_scaled_df, X_test_scaled_df, feature_names)
    # save_artifacts(model, scaler, feature_names)
    predict_custom_input(model, scaler, feature_names)


if __name__ == "__main__":
    main()