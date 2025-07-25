# Crypto Forecasting

The project focuses on predicting future trends in cryptocurrency markets using historical market data( dataset with ~525K rows ). It employs XGBoost, a powerful gradient boosting framework, along with feature engineering and hyperparameter optimization to build a robust predictive model. The goal is to identify patterns and predict market movements based on various indicators.


**Model**: XGBoost Regressor

**Tuning**: Optuna with TimeSeriesSplit to ensure no future information leaks into past training.

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.






