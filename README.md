\# Sri Lankan Rainfall Forecasting Using Ensemble Learning and Hybrid Models



\## Project Overview



This project explores the application of ensemble learning techniques and hybrid models for forecasting rainfall across Sri Lanka.



Historical rainfall observations from different geographical areas are used to train and evaluate multiple machine learning models. The project compares individual ensemble algorithms, a stacking approach, and a hybrid Random Forest–ARIMA model to identify their forecasting performance.



An interactive Streamlit dashboard is also developed to visualise model performance, explore feature importance, and generate rainfall predictions for selected geographical areas.



\## Dataset



\- \*\*Source:\*\* \[Sri Lanka Rainfall Analysis Dataset](https://github.com/wethumlansakara/Sri-Lanka-Rainfall-Analysis)

\- \*\*Original data provider:\*\* Humanitarian Data Exchange (HDX)

\- \*\*Geographical coverage:\*\* 9 provinces and 25 districts in Sri Lanka

\- \*\*Time period:\*\* 1981–2026

\- \*\*Number of observations:\*\* 55,590 (before preprocessing)



The dataset contains historical rainfall measurements and geographical information for different administrative areas.



\## Methodology



\### 1. Data Exploration and Preprocessing

\- Explored rainfall observations and geographical coverage.

\- Examined missing values and duplicate records.

\- Prepared chronological observations for forecasting.

\- Created lag-based features to capture historical rainfall patterns.

\- Applied one-hot encoding to categorical geographical features.



\### 2. Forecasting Approach



The project uses historical rainfall observations and lag features to predict rainfall in the next available observation period.



A chronological 80:20 training and testing split was used to preserve temporal order and avoid using future observations during training.



\### 

