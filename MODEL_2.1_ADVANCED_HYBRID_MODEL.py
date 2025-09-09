import pandas as pd
import numpy as np
import joblib, os, datetime, warnings, gc
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*feature names.*")
warnings.filterwarnings("ignore", message=".*inplace.*")
warnings.filterwarnings("ignore", message=".*NumPy global RNG.*")

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import (
    StackingClassifier, RandomForestClassifier,
    GradientBoostingClassifier, ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import uniform, randint
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import shap
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1. Data Loading + Preprocessing + Feature Engineering
def load_and_preprocess_data():
    df = pd.read_csv('pima.csv')

    # Replace impossible 0 values with NaN and impute class-wise median
    cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols] = df[cols].replace(0, np.nan)
    for col in cols:
        df[col] = df.groupby('Outcome')[col].transform(lambda x: x.fillna(x.median()))

    # Feature Engineering
    df['Glucose_BMI_Ratio'] = df['Glucose'] / (df['BMI'] + 1e-3)
    df['Age_Glucose_Int'] = df['Age'] * df['Glucose']
    df['Insulin_BMI_Ratio'] = df['Insulin'] / (df['BMI'] + 1e-3)
    df['Age_BMI_Int'] = df['Age'] * df['BMI']
    df['Is_Obese'] = (df['BMI'] > 30).astype(int)
    df['Is_Young'] = (df['Age'] < 30).astype(int)

    # Non-linear transformations
    df['Glucose2'] = df['Glucose'] ** 2
    df['BMI2'] = df['BMI'] ** 2
    df['Pregnancies_log1p'] = np.log1p(df['Pregnancies'])
    df['Insulin_log1p'] = np.log1p(df['Insulin'])

    X = df.drop(columns='Outcome')
    y = df['Outcome']
    return X, y, X.columns.tolist()

# 2. Feature Selection
def feature_selection(X, y, k=20):
    selector = SelectKBest(f_classif, k=k)
    X_sel = selector.fit_transform(X, y)
    selected_names = X.columns[selector.get_support()].tolist()
    return X_sel, selected_names

# 3. Hyperparameter Tuning for Multiple Models
def tune_best_models(X, y, random_state=42):
    cv_inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

    def make_pipeline(model):
        return ImbPipeline([
            ('smote', SMOTE(random_state=random_state)),
            ('clf', model)
        ])

    best_models = {}

    # XGBoost
    xgb_clf = xgb.XGBClassifier(objective='binary:logistic', eval_metric='auc', n_jobs=-1, random_state=random_state)
    rs_xgb = RandomizedSearchCV(
        make_pipeline(xgb_clf),
        {
            'clf__n_estimators': randint(200, 800),
            'clf__max_depth': randint(3, 8),
            'clf__learning_rate': uniform(0.05, 0.2),
            'clf__subsample': uniform(0.7, 0.3),
            'clf__colsample_bytree': uniform(0.7, 0.3),
            'clf__reg_lambda': uniform(0, 3)
        },
        n_iter=25, scoring='roc_auc', cv=cv_inner, n_jobs=-1, verbose=0, random_state=random_state
    )
    rs_xgb.fit(X, y)
    best_models['xgb'] = rs_xgb.best_estimator_.named_steps['clf']
    print("XGB AUC:", rs_xgb.best_score_)

    # LightGBM
    lgb_clf = lgb.LGBMClassifier(objective='binary', metric='auc', n_jobs=-1, random_state=random_state, verbosity=-1)
    rs_lgb = RandomizedSearchCV(
        make_pipeline(lgb_clf),
        {
            'clf__n_estimators': randint(50, 300),
            'clf__max_depth': randint(3, 6),
            'clf__learning_rate': uniform(0.05, 0.2),
            'clf__num_leaves': randint(8, 32),
            'clf__min_data_in_leaf': randint(5, 20),
            'clf__feature_fraction': uniform(0.6, 0.4),
            'clf__bagging_fraction': uniform(0.6, 0.4),
            'clf__reg_lambda': uniform(0, 3)
        },
        n_iter=25, scoring='roc_auc', cv=cv_inner, n_jobs=-1, verbose=0, random_state=random_state
    )
    rs_lgb.fit(X, y)
    best_models['lgb'] = rs_lgb.best_estimator_.named_steps['clf']
    print("LGB AUC:", rs_lgb.best_score_)

    # CatBoost
    cb_clf = cb.CatBoostClassifier(loss_function='Logloss', eval_metric='AUC', thread_count=-1,
                                   random_state=random_state, verbose=False)
    rs_cb = RandomizedSearchCV(
        make_pipeline(cb_clf),
        {
            'clf__iterations': randint(200, 800),
            'clf__depth': randint(3, 8),
            'clf__learning_rate': uniform(0.05, 0.2),
            'clf__l2_leaf_reg': uniform(1, 8)
        },
        n_iter=25, scoring='roc_auc', cv=cv_inner, n_jobs=-1, verbose=0, random_state=random_state
    )
    rs_cb.fit(X, y)
    best_models['cb'] = rs_cb.best_estimator_.named_steps['clf']
    print("CatBoost AUC:", rs_cb.best_score_)

    # RandomForest
    rf_clf = RandomForestClassifier(n_jobs=-1, random_state=random_state)
    rs_rf = RandomizedSearchCV(
        make_pipeline(rf_clf),
        {
            'clf__n_estimators': randint(200, 800),
            'clf__max_depth': [None] + list(range(3, 15)),
            'clf__min_samples_split': randint(2, 10),
            'clf__max_features': ['sqrt', 'log2', 0.5, 0.7]
        },
        n_iter=25, scoring='roc_auc', cv=cv_inner, n_jobs=-1, verbose=0, random_state=random_state
    )
    rs_rf.fit(X, y)
    best_models['rf'] = rs_rf.best_estimator_.named_steps['clf']
    print("RF AUC:", rs_rf.best_score_)

    # GradientBoosting
    gb_clf = GradientBoostingClassifier(random_state=random_state)
    rs_gb = RandomizedSearchCV(
        make_pipeline(gb_clf),
        {
            'clf__n_estimators': randint(100, 500),
            'clf__learning_rate': uniform(0.05, 0.2),
            'clf__max_depth': randint(3, 8),
            'clf__subsample': uniform(0.7, 0.3),
            'clf__min_samples_split': randint(2, 10)
        },
        n_iter=25, scoring='roc_auc', cv=cv_inner, n_jobs=-1, verbose=0, random_state=random_state
    )
    rs_gb.fit(X, y)
    best_models['gb'] = rs_gb.best_estimator_.named_steps['clf']
    print("GB AUC:", rs_gb.best_score_)

    # ExtraTrees
    et_clf = ExtraTreesClassifier(n_jobs=-1, random_state=random_state)
    rs_et = RandomizedSearchCV(
        make_pipeline(et_clf),
        {
            'clf__n_estimators': randint(200, 800),
            'clf__max_depth': [None] + list(range(3, 15)),
            'clf__min_samples_split': randint(2, 10),
            'clf__max_features': ['sqrt', 'log2', 0.5, 0.7]
        },
        n_iter=25, scoring='roc_auc', cv=cv_inner, n_jobs=-1, verbose=0, random_state=random_state
    )
    rs_et.fit(X, y)
    best_models['et'] = rs_et.best_estimator_.named_steps['clf']
    print("ET AUC:", rs_et.best_score_)

    return best_models

# 4. Build Stacking Ensemble
def build_stacking(best_models, X, y):
    base_learners = [(name, model) for name, model in best_models.items()]
    meta = LogisticRegression(max_iter=1000, C=0.1, penalty='l2')
    stack = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        n_jobs=-1,
        passthrough=True
    )
    X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
    stack.fit(X_res, y_res)
    return stack

# 5. Evaluation
def evaluate(model, X, y):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    print("\n=== Evaluation ===")
    print("Accuracy :", accuracy_score(y, y_pred))
    print("ROC AUC  :", roc_auc_score(y, y_proba))
    print("PR AUC   :", average_precision_score(y, y_proba))
    print("\nConfusion Matrix:\n", confusion_matrix(y, y_pred))
    print("\nClassification Report:\n", classification_report(y, y_pred))
    return y_pred, y_proba

# 6. SHAP Explanations
def shap_explain(model, X, feature_names):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X, columns=feature_names)
    explainer = shap.Explainer(model.predict_proba, X, seed=42)
    shap_values = explainer(X)
    shap.summary_plot(shap_values[:, :, 1], X, feature_names=feature_names)
    plt.figure(figsize=(10, 6))
    shap.plots.beeswarm(shap_values[:, :, 1], show=False)
    plt.title("SHAP Value Distribution")
    plt.tight_layout()
    plt.show()
    return explainer, shap_values

# 7. Save Model
def save_all(model, scaler, feature_names):
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    joblib.dump({
        'model': model,
        'scaler': scaler,
        'feature_names': feature_names,
        'timestamp': stamp
    }, f'diabetes_model_{stamp}.pkl')
    print("Saved as", f'diabetes_model_{stamp}.pkl')

# 8. Interactive Prediction
def predict_single(model, scaler, feature_names):
    print("\n=== Single Prediction ===")
    vals = [float(input(f"{feat}: ")) for feat in feature_names]
    X_scaled = pd.DataFrame([vals], columns=feature_names)
    X_scaled = pd.DataFrame(scaler.transform(X_scaled), columns=feature_names)
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0, 1]
    print(f"Prediction : {'Diabetes' if pred else 'No Diabetes'}")
    print(f"Probability: {prob:.2%}")

# 9. Main Function
def main():
    X, y, _ = load_and_preprocess_data()
    X_sel, selected_names = feature_selection(X, y, k=20)

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        pd.DataFrame(X_sel, columns=selected_names), y,
        test_size=0.2, stratify=y, random_state=42
    )

    # Scaling
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=selected_names, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test), columns=selected_names, index=X_test.index)

    # Tune Models + Build Stacking Ensemble
    best_models = tune_best_models(X_train, y_train)
    stack = build_stacking(best_models, X_train, y_train)

    # Evaluate
    evaluate(stack, X_test, y_test)
    shap_explain(stack, X_test, selected_names)

    # Save Model
    save_all(stack, scaler, selected_names)

    # Custom Prediction
    predict_single(stack, scaler, selected_names)

if __name__ == "__main__":
    main()
