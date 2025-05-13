import pandas as pd #
import numpy as np #
from sklearn.model_selection import train_test_split #
from sklearn.preprocessing import StandardScaler #
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import shap
from lime.lime_tabular import LimeTabularExplainer
from alibi.explainers import Counterfactual
import tensorflow as tf
# loading and exploring of the data sets

df = pd.read_csv('pima.csv')
print(df.head())  # see first few rows
print(df.info())  # check data types
print(df.describe())  # get basic stats


# In this dataset, some 0s are actually missing values, especially for:
# Glucose
# BloodPressure
# SkinThickness
# Insulin
# BMI

cols_with_zero = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

# Replace 0s with NaN
df[cols_with_zero] = df[cols_with_zero].replace(0, np.nan)

# Fill missing values with the column's median
df.fillna(df.median(), inplace=True)

# Split Features and Target

X = df.drop('Outcome', axis=1)   # features
y = df['Outcome']                # target (0 = non-diabetic, 1 = diabetic)

#   Scale the Features - Scaling helps improve model performance (especially for distance-based or tree models).

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,
                                                    test_size=0.2,
                                                    random_state=42,
                                                    stratify=y)


# NEXT STEP IS to train the  Model - Train your XGBoost model (or Random Forest/Logistic Regression if you prefer to start simple).
# We’ll use XGBoost — a powerful tree-based model that's fast, accurate, and works well with structured data like Pima.
# install XGBoost



# Initialize the model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

# Train the model
model.fit(X_train, y_train)

# evaluate the model
# Make predictions
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Confusion Matrix
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Full Report
print("Classification Report:\n", classification_report(y_test, y_pred))

#  What Next? Step 3: Explain Your Model with SHAP
# SHAP (SHapley Additive exPlanations) helps you understand why your model made certain predictions. It shows the impact of each feature on the prediction.


# Initialize the explainer
explainer = shap.Explainer(model, X_train)

# Calculate SHAP values
shap_values = explainer(X_test)

# Summary plot (shows feature importance)
shap.summary_plot(shap_values, X_test, feature_names=df.columns[:-1])

# # Explain the first prediction
# shap.plots.waterfall(shap_values[0], max_display=10)

# 🧠 What You'll Learn:
# Which features (like Glucose, BMI) are most important
#
# Why the model predicts a person as diabetic (1) or non-diabetic (0)
#
# Which features increase or decrease the ris




# SHAP Dependence Plots
# See how individual features affect output:

shap.dependence_plot("Glucose", shap_values, X_test)

# Force Plot for a Single Prediction
# Visualize how features push a single prediction:

shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])

# Global Feature Importance (bar plot)
# Show average impact per feature:

shap.plots.bar(shap_values)


# 🔹 Force Plot (Single Prediction)

shap.initjs()
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_test.iloc[0]
)

# 🔹 Dependence Plot (Feature Interaction)

shap.dependence_plot("Glucose", shap_values.values, X_test)

# 2. LIME (Local Interpretable Model-Agnostic Explanations)
# 🔹 Basic LIME Integration (For Classifier)

explainer = LimeTabularExplainer(
    training_data=np.array(X_train),
    feature_names=X_train.columns,
    class_names=['No Diabetes', 'Diabetes'],
    mode='classification'
)

# Explain one instance
exp = explainer.explain_instance(
    data_row=X_test.iloc[0],
    predict_fn=model.predict_proba
)

# Visualize
exp.show_in_notebook(show_table=True)

# Counterfactual Explanations
# This shows how to change input features to flip the model prediction.

# Wrap your model into a keras-style function
predict_fn = lambda x: model.predict_proba(x).astype(np.float32)

shape = (1, X_train.shape[1])  # assuming 1D input

cf = Counterfactual(
    predict_fn,
    shape=shape,
    target_proba=1.0,
    tol=0.01,
    lam_init=1e-1,
    max_iter=1000,
    learning_rate_init=0.1
)

explanation = cf.explain(X_test.iloc[0:1].to_numpy())
print("Counterfactual instance:", explanation.cf['X'])

# achieveMents

# | Tool                | Purpose                                     | Scope          |
# | ------------------- | ------------------------------------------- | -------------- |
# | **SHAP**            | How each feature contributed to prediction  | Global + Local |
# | **LIME**            | Local approximation of prediction logic     | Local          |
# | **Counterfactuals** | What to change in input to alter prediction | Local          |



# 🧪 1. Compare Explanations for Same Instance
# Explain the same X_test.iloc[0] using SHAP, LIME, and Counterfactual. Then:
#
# Summarize: Are they consistent?
#
# Where do they disagree, and why?
#
# You can show a side-by-side table like this:
#
# Feature	SHAP Impact	LIME Impact	Counterfactual Change
# Glucose	+0.42	+0.35	-10 (to flip outcome)
# BMI	+0.15	+0.18	No change
# Age	-0.10	-0.12	No change
#
# This shows your critical thinking.


#  2. Automate & Export Explanations for All Test Samples (optional)
# You can create a loop that:
#
# Runs LIME & SHAP for the first 10 test samples
#
# Saves them as HTML or plots
#
# Useful for report or web dashboard


for i in range(10):
    exp = explainer.explain_instance(X_test[i], model.predict_proba)
    exp.save_to_file(f'lime_exp_{i}.html')
    shap.plots.waterfall(shap_values[i], max_display=10)

# 📄 3. Create a Report or Dashboard (Final Presentation)
# Summarize your project for final submission:
#
# Model overview + metrics
#
# SHAP summary & dependence plots
#
# LIME explanation (1–2 samples)
#
# Counterfactual suggestion
#
# Feature importance table
#
# Tools: Jupyter Notebook / PDF export / PowerPoint






