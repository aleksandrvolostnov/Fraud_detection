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

