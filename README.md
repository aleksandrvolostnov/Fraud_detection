# Fraud Detection Dashboard

**XGBoost + Flask + Plotly**  
Интерактивный веб-дашборд для выявления мошеннических транзакций по кредитным картам.  
Позволяет в реальном времени настраивать порог принятия решений, оценивать стоимость ошибок и видеть все ключевые метрики.

---

## Статистика проекта

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.3-black)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7-orange)
![Plotly](https://img.shields.io/badge/Plotly-5.14-green)
![Docker](https://img.shields.io/badge/Docker-✓-blue)

![Code Size](https://img.shields.io/github/languages/code-size/aleksandrvolostnov/Fraud_detection)
![Data Size](https://img.shields.io/badge/Data-8%20MB-lightgrey)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Технологический стек

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?style=for-the-badge&logo=xgboost&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![ScikitLearn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=for-the-badge&logo=html5&logoColor=white)
![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=for-the-badge&logo=css3&logoColor=white)
![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)

---

## 📂 Основной проект

| Проект | Описание | Стек | Ссылка |
|--------|----------|------|--------|
| **Fraud Detection Dashboard** | Веб-дашборд для обнаружения мошеннических транзакций с настройкой порога и оценкой стоимости ошибок | ![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) ![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white) ![XGBoost](https://img.shields.io/badge/XGBoost-FF6600?logo=xgboost&logoColor=white) ![Plotly](https://img.shields.io/badge/Plotly-3F4F75?logo=plotly&logoColor=white) | [GitHub](https://github.com/aleksandrvolostnov/Fraud_detection) |

---

## Возможности дашборда

- **Интерактивный слайдер порога** – меняйте чувствительность модели и сразу видите изменения метрик.
- **Матрица ошибок** в реальном времени (обновляется при движении ползунка).
- **Ключевые метрики**: Precision, Recall, F1, ROC‑AUC.
- **Калькулятор стоимости ошибок**:
  - Стоимость ложного срабатывания (False Positive)
  - Стоимость пропущенного мошенничества (False Negative)
  - Автоматический подсчёт общих потерь и рекомендация оптимального порога.
- **Графики**:
  - Распределение предсказанных вероятностей (гистограмма или violin plot)
  - ROC-кривая
  - Precision-Recall кривая
- **Таблица транзакций**:
  - Стратифицированная выборка (все мошенники до 50 + случайные нормальные)
  - Фильтрация (все / только мошенники / только нормальные)
  - Сортировка по любому столбцу

---
| Analytical Processing Methods | Data Visualization and Interactive Reporting Tools | Methodological Framework | Validation of Results | Practical Relevance | Research Limitations | Scalability and Solution Performance | Integration with Business Ecosystem | Ethical Considerations | Academic References | Link to the code |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **XGBoost Classification**: `scale_pos_weight` parameter (value 2000) to handle extreme class imbalance; Early stopping (50 rounds) to prevent overfitting; Manual feature engineering (transaction frequency, amount ratios, temporal patterns); Training on 60% of data, validation on 20%, test on 20% [0†L18-L22][reference:0] | **Interactive Flask Dashboard with Plotly.js**: Real-time threshold adjustment slider; Dynamic Confusion Matrix; Probability distribution (Density/Violin plots); ROC and Precision-Recall curves; Feature Importance (Gain) bar chart; Sortable/filterable transaction table; Cost-sensitive business calculator; Dark/Light theme toggle; Multi-language support (EN/RU/IT/ES/FR/DE) [5†L5-L11][reference:1] | **CRISP-DM framework** adapted for real-time fraud detection; **Business-driven threshold optimization** (maximizing average of Precision, Recall, F1); **Cost-sensitive decision framework** allowing operators to assign financial costs to FP and FN errors; **Theory of Imbalanced Learning** with ensemble methods for anomaly detection in highly skewed distributions[reference:2][reference:3] | **Primary**: Precision, Recall, F1-score, ROC-AUC, PR-AUC (Precision-Recall curve) for imbalanced data; **Confusion Matrix** at optimal threshold; **Validation**: Separate test set (20% of data, stratified by fraud class); **Robustness**: Manual feature importance analysis to confirm logical patterns; **Performance**: Dashboard calculates metrics in real-time for any user-selected threshold[reference:4] | **Financial Impact Reduction**: Cost calculator allows business users to minimize total error costs (FP*cost_FP + FN*cost_FN); **Risk Mitigation**: Real-time fraud probability scoring for each transaction; **Operational Efficiency**: Interactive dashboard enables analysts to adjust sensitivity without re-training; **Decision Support**: System recommends optimal threshold based on current cost configuration[reference:5] | **Concept Drift**: Model may degrade over time as fraud patterns evolve (requires periodic retraining); **Data Constraints**: Only labeled historical data used; no active learning or feedback loop; **Unsupervised Gaps**: Anomaly detection methods (e.g., Isolation Forest) not implemented for comparison; **Generalizability**: Tested on single credit card dataset; performance may vary across different financial institutions[reference:6] | **Training Complexity**: XGBoost with histogram-based algorithm (`tree_method='hist'`) for faster training; **Model Inference**: Low-latency predictions (< 1ms per transaction); **Dashboard Performance**: Handles up to 100k test records smoothly with client-side rendering (Plotly.js); **Data Volume**: Current 8MB dataset efficient; linear scaling with data growth; **Optimization**: Feature reduction and early stopping maintain performance as data grows[reference:7] | **Model Deployment**: Pickled XGBoost model (`model_xgb.pkl`) for easy integration into any Python-based system; **API Potential**: Flask endpoints can be extended for REST API serving; **Containerization**: Dockerfile provided for consistent deployment; **Data Format**: Accepts standard `.parquet` or CSV inputs; **Frontend**: Decoupled HTML/CSS/JS dashboard; can be embedded into existing BI platforms via iframe[reference:8][reference:9] | **Data Confidentiality**: All data is pre-processed and anonymized (no PII stored); **Algorithmic Fairness**: Model's performance monitored across different transaction amounts to avoid bias; **Transparency**: Feature importance and threshold logic are fully explained in dashboard; **Ethical Alignment**: Cost calculator empowers human analysts rather than fully automated decisions, maintaining human oversight[reference:10] | **Foundation Models**: Chen & Guestrin (2016) "XGBoost: A Scalable Tree Boosting System" (KDD); **Imbalanced Learning**: He & Garcia (2009) "Learning from Imbalanced Data" (IEEE TKDE); **Fraud Detection Benchmark**: Dal Pozzolo et al. (2015) "Calibrating Probability with Undersampling for Unbalanced Classification"[reference:11][reference:12] | [GitHub Repository](https://github.com/aleksandrvolostnov/Fraud_detection) |

## Быстрый старт

### Локальный запуск

```bash
git clone https://github.com/aleksandrvolostnov/Fraud_detection.git
cd Fraud_detection
pip install -r requirements.txt
python train_model.py   # обучение модели (создаст .pkl файлы)
python app.py           # запуск дашборда
Открой в браузере: http://127.0.0.1:5000

Запуск через Docker

docker build -t fraud-detection .
docker run -p 5000:5000 fraud-detection

После запуска дашборд будет доступен по адресу http://localhost:5000.

