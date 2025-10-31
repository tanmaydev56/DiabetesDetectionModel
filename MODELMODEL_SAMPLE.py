import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
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
    df = pd.read_csv(file_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Class distribution:\n{df['class'].value_counts()}")
    for col in df.columns:
        if df[col].dtype == 'object' or col == 'class':
            if 'Gender' in col:
                df[col] = df[col].map({'Male': 1, 'Female': 0})
            elif 'class' in col:
                df[col] = df[col].map({'Positive': 1, 'Negative': 0})
            else:
                df[col] = df[col].map({'Yes': 1, 'No': 0})
    df.dropna(inplace=True)
    print(f"Processed dataset shape: {df.shape}")
    X = df.drop('class', axis=1)
    y = df['class']
    feature_names = X.columns.tolist()
    class_names = ['No Diabetes', 'Diabetes']
    return X, y, feature_names, class_names

def feature_selection(X, y, k=10):
    print(f"Performing feature selection to find the best {k} features...")
    selector = SelectKBest(f_classif, k=min(k, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()]
    print("Selected Features:", selected_features.tolist())
    return X_selected, selected_features, selector

def run_one_fold(X, y, train_idx, test_idx, original_feature_names):
    X_train_full, X_test_full = X.iloc[train_idx], X.iloc[test_idx]
    y_train_full, y_test_full = y.iloc[train_idx], y.iloc[test_idx]
    selector = SelectKBest(f_classif, k=min(10, X_train_full.shape[1]))
    X_train_sel = selector.fit_transform(X_train_full, y_train_full)
    X_test_sel  = selector.fit_transform(X_test_full, y_test_full)
    selected_features = X_train_full.columns[selector.get_support()]
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_sel, y_train_full)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sm)
    X_test_scaled  = scaler.transform(X_test_sel)
    param_grid = {'max_depth': [3, 5], 'learning_rate': [0.1, 0.2], 'n_estimators': [100, 200], 'subsample': [0.8, 1.0]}
    xgb_clf = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', use_label_encoder=False, random_state=42)
    grid = GridSearchCV(xgb_clf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=0)
    grid.fit(X_train_scaled, y_train_sm)
    best_model = grid.best_estimator_
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
    return metrics, best_model, scaler, selector, selected_features

def train_final_model(X, y, feature_names):
    print("\nTRAINING FINAL MODEL ON FULL DATASET")
    selector = SelectKBest(f_classif, k=min(10, X.shape[1]))
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_selected, y)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_resampled)
    final_model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False,
        max_depth=5,
        learning_rate=0.1,
        num_boost_round=200,
        subsample=0.8
    )
    final_model.fit(X_scaled, y_resampled)
    print("Final model trained successfully!")
    return final_model, scaler, selector, selected_features

def explain_model(model, X_train, X_test, feature_names):
    print("\nGenerating SHAP explanations...")
    if len(X_test) > 100:
        X_test = X_test[:100]
    X_test_df = pd.DataFrame(X_test, columns=feature_names)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_df)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_df, plot_type="bar", show=False)
    plt.title('Global Feature Importance (SHAP)')
    plt.tight_layout()
    plt.show()

def predict_custom_input(model, scaler, feature_names):
    print("\nCUSTOM PREDICTION – enter values:")
    user_input_dict = {}
    for feature in feature_names:
        if feature == 'Age':
            val = float(input(f"Enter value for {feature}: "))
        elif feature == 'Gender':
            val = input(f"Enter value for {feature} (M/F): ").strip().upper()
            val = 1 if val == 'M' else 0
        else:
            val = input(f"Enter value for {feature} (Y/N): ").strip().upper()
            val = 1 if val == 'Y' else 0
        user_input_dict[feature] = val
    input_df = pd.DataFrame([user_input_dict])[feature_names]
    user_scaled = scaler.transform(input_df)
    prediction = model.predict(user_scaled)[0]
    prob = model.predict_proba(user_scaled)[0][1]
    print(f"\nPredicted Class: {'DIABETES' if prediction == 1 else 'NO DIABETES'}")
    print(f"Probability of Diabetes: {prob * 100:.2f}%")

def cv_main():
    X, y, original_feature_names, class_names = load_and_preprocess_data()
    if X.shape[0] == 0:
        print("No data left after preprocessing – aborting.")
        return
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        print(f'\n{"="*20} FOLD {fold}/5 {"="*20}')
        metrics, model, scaler, selector, features = run_one_fold(
            X, y, train_idx, test_idx, original_feature_names
        )
        fold_scores.append(metrics)
    df_cv = pd.DataFrame(fold_scores)
    print('\nCross-Validation Summary:')
    print(df_cv)
    for col in df_cv.columns:
        mean_val = df_cv[col].mean()
        std_val = df_cv[col].std()
        print(f"{col:8}: {mean_val:.4f} ± {std_val:.4f}")
    final_model, final_scaler, final_selector, final_features = train_final_model(X, y, original_feature_names)
    explain_model(final_model, X, X, final_features)
    while True:
        try:
            response = input("\nMake a custom prediction? (y/n): ").strip().lower()
            if response in ['y', 'yes']:
                predict_custom_input(final_model, final_scaler, final_features)
            elif response in ['n', 'no']:
                print("Thank you for using the Diabetes Prediction System!")
                break
        except KeyboardInterrupt:
            print("\n\nExiting. Thank you!")
            break

if __name__ == "__main__":
    cv_main()
