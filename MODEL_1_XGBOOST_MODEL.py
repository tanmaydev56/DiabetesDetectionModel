import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report,roc_auc_score, average_precision_score
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


def train_model(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'n_estimators': [100, 200, 300],
        'gamma': [0, 0.1, 0.2]
    }
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        eval_metric='logloss',
        scale_pos_weight=np.sqrt(len(y_train_res[y_train_res == 0]) /
                                 len(y_train_res[y_train_res == 1]))
    )
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='roc_auc',
        verbose=1
    )
    grid_search.fit(X_train_res, y_train_res)
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV AUC: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test):
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
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model, max_num_features=10)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.show()
    return y_pred, y_proba


def explain_model(model, X_train, X_test, feature_names):
    print("\nGenerating SHAP explanations...")
    X_train_df = pd.DataFrame(X_train, columns=feature_names)
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    explainer = shap.Explainer(model, X_train_df)
    shap_values = explainer(X_test_df)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_df, plot_type="bar")
    plt.title('Feature Importance (SHAP)')
    plt.tight_layout()
    plt.show()
    sample_idx = 0
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[sample_idx], max_display=10)
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
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_names)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_names)
    model = train_model(X_train_scaled, y_train)
    evaluate_model(model, X_test_scaled, y_test)
    explain_model(model, X_train_scaled, X_test_scaled, feature_names)
    # save_artifacts(model, scaler, feature_names)
    predict_custom_input(model, scaler, feature_names)


if __name__ == "__main__":
    main()