# Sri Lankan Rainfall Forecasting Using Ensemble Learning and Hybrid Models

## Project Overview

This project explores the application of ensemble learning techniques and hybrid models for forecasting rainfall across Sri Lanka.

Historical rainfall observations from different geographical areas are used to train and evaluate multiple machine learning models. The project compares individual ensemble algorithms, a stacking approach, and a hybrid Random Forest–ARIMA model to identify their forecasting performance.

An interactive Streamlit dashboard is also developed to visualise model performance, explore feature importance, and generate rainfall predictions for selected geographical areas.

## Dataset

- **Source:** [Sri Lanka Rainfall Analysis Dataset](https://github.com/wethumlansakara/Sri-Lanka-Rainfall-Analysis)
- **Original data provider:** Humanitarian Data Exchange (HDX)
- **Geographical coverage:** 9 provinces and 25 districts in Sri Lanka
- **Time period:** 1981–2026
- **Number of observations:** 55,590 (before preprocessing)

The dataset contains historical rainfall measurements and geographical information for different administrative areas.

## Methodology

### 1. Data Exploration and Preprocessing
- Explored rainfall observations and geographical coverage.
- Examined missing values and duplicate records.
- Prepared chronological observations for forecasting.
- Created lag-based features to capture historical rainfall patterns.
- Applied one-hot encoding to categorical geographical features.

### 2. Forecasting Approach

The project uses historical rainfall observations and lag features to predict rainfall in the next available observation period.

A chronological 80:20 training and testing split was used to preserve temporal order and avoid using future observations during training.

### 3. Models Implemented

The following approaches were implemented and compared:

- Random Forest
- XGBoost
- LightGBM
- CatBoost
- Stacking Ensemble
- ARIMA
- Hybrid Random Forest–ARIMA
- Tuned XGBoost

### 4. Model Evaluation

Models were evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

## Model Performance

| Model | MAE | RMSE | R² |
|---|---:|---:|---:|
| Random Forest | 36.4007 | 54.4462 | 0.2949 |
| XGBoost | 34.2437 | 51.4821 | 0.3696 |
| LightGBM | 34.3427 | 51.5076 | 0.3690 |
| CatBoost | 35.0272 | 52.1662 | 0.3527 |
| Stacking | 35.5968 | 51.3663 | 0.3724 |
| ARIMA | 38.3072 | 55.5040 | 0.2673 |
| Hybrid RF–ARIMA | 36.2229 | 53.2939 | 0.3245 |
| Tuned XGBoost | 34.6044 | 51.3243 | 0.3735 |

**Key findings:**
- XGBoost achieved the lowest MAE.
- Tuned XGBoost achieved the lowest RMSE and highest R².
- Stacking achieved competitive performance.
- The tested hybrid model did not outperform the strongest boosting models.

## Interactive Dashboard

The Streamlit dashboard provides:

1. **Overview:** Project summary and model performance metrics.
2. **Model Comparison:** Interactive comparison of forecasting models.
3. **Feature Importance:** Visualisation of important prediction features.
4. **Rainfall Prediction:** Rainfall prediction based on the selected province or district and historical rainfall inputs.

## Project Structure

```text
SriLankan_Weather_Ensemble_Project/
│
├── dashboard/
│   ├── app.py
│   └── assets/
│
├── data/
│   └── raw/
│       └── weather.csv
│
├── notebooks/
│   ├── 01_Dataset_Exploration.ipynb
│   └── models/
│       ├── catboost.pkl
│       ├── final_model_comparison.csv
│       ├── hybrid_rf_arima.pkl
│       ├── lightgbm.pkl
│       ├── stacking_model.pkl
│       ├── tuned_xgboost.pkl
│       └── xgboost.pkl
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation and Execution

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd SriLankan_Weather_Ensemble_Project
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the Streamlit dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard will open in your browser.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Statsmodels
- Matplotlib and Seaborn
- Plotly
- Streamlit
- Jupyter Notebook

## Generative AI Usage

Generative AI tools were used for assistance with selected aspects of development, including code troubleshooting, explanations, and dashboard UI improvements. The dataset analysis, model implementation, experimentation, evaluation, and interpretation were carried out as part of this project.

## Author

Tharushi De Silva