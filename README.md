# 🩺 Diabetes Detection Hub

A curated collection of state-of-the-art **binary-classification** models trained on the classic **PIMA Indians Diabetes** dataset (and an extended **“diabetes_data_upload”** set).  
From classic ML to modern deep-learning + attention, every experiment is fully reproducible and shipped with **SHAP explainability**, **hyper-parameter tuning**, **class-imbalance handling** and **interactive inference**.

---

## 📁 Repository Map
| File | Purpose | Data | Key Tech |
|---|---|---|---|
| `MODEL_1_XGBOOST_MODEL.py` | Gradient-boosted trees + SMOTE | PIMA | XGBoost, GridSearch, SHAP |
| `MODEL_2_HYBRID_MODEL.py` | Stacking ensemble of 6 algorithms | PIMA | XGBoost, LGBM, CatBoost, RF, GBM, ET, LogReg meta |
| `MODEL_2.1_ADVANCED_HYBRID_MODEL.py` | Same as above but extended feature-engineering | PIMA | ditto |
| `MODEL_3_ARTIFICIAL_NEURAL_NETWORK.py` | Keras sequential network | PIMA | Focal loss, class-weights, early-stopping |
| `MODEL_4_DEEP_LEARNING_MODEL_GOOD_D.py` | Attention-augmented deep net | diabetes_data_upload | Custom Attention layer, Focal-loss mix |
| `app.py` | Minimal FastAPI/Flask wrapper (inference) | any | load `.h5`/`.pkl` |
| `diabetes_model.h5` | Exported Keras model (Attention-net) | — | — |
| `age_scaler.pkl` | Min-max scaler for age | — | — |

---

## 🚀 Quick Start (local)
1. Clone & enter repo  
   ```bash
   git clone https://github.com/&lt;user&gt;/DiabetesDetectionModel.git
   cd DiabetesDetectionModel
