import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template
import plotly.graph_objs as go
import plotly.utils
import json

app = Flask(__name__)

# Глобальные переменные
model = None
scaler = None
best_threshold = None
X_test = None
y_test = None
y_pred = None
y_proba = None
roc_data = None
pr_data = None
metrics = {}
feature_importance = {}
sample_transactions = None

def load_artifacts():
    global model, scaler, best_threshold, X_test, y_test, y_pred, y_proba
    global roc_data, pr_data, metrics, feature_importance, sample_transactions

    try:
        model = joblib.load('model_xgb.pkl')
        scaler = joblib.load('scaler.pkl')
        best_threshold = joblib.load('best_threshold.pkl')
        test_data = joblib.load('test_data.pkl')
        X_test, y_test, y_pred, y_proba = test_data
        roc_data = joblib.load('roc_data.pkl')
        pr_data = joblib.load('pr_data.pkl')
        metrics = joblib.load('metrics.pkl')
        feature_importance = joblib.load('feature_importance.pkl')
        sample_transactions = joblib.load('sample_transactions.pkl')

        print("Артефакты и примеры транзакций загружены")
    except Exception as e:
        print(f"Ошибка загрузки артефактов: {e}")
        raise

load_artifacts()
print("=== ОТЛАДКА ===")
print(f"y_test тип: {type(y_test)}")
print(f"y_test размер: {len(y_test) if hasattr(y_test, '__len__') else '?'}")
print(f"y_proba тип: {type(y_proba)}")
print(f"y_proba размер: {len(y_proba) if hasattr(y_proba, '__len__') else '?'}")
if len(y_test) > 0:
    print(f"Первые 5 y_test: {y_test[:5]}")
    print(f"Первые 5 y_proba: {y_proba[:5]}")
print("===============")
def pastel_layout(title, x_title, y_title):
    """Тёмный фон + пастельные акценты"""
    return go.Layout(
        title=dict(text=title, font=dict(color='#e0e4f0', size=15), x=0.02),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#b7c4dd'),
        xaxis=dict(title=x_title, gridcolor='rgba(255,255,255,0.06)', zerolinecolor='rgba(255,255,255,0.12)'),
        yaxis=dict(title=y_title, gridcolor='rgba(255,255,255,0.06)', zerolinecolor='rgba(255,255,255,0.12)'),
        margin=dict(l=50, r=30, t=60, b=40),
        legend=dict(font=dict(color='#d0d8f0'), bgcolor='rgba(20,25,45,0.5)')
    )

@app.route('/')
def dashboard():
    # 1. ROC кривая
    fpr, tpr, auc = roc_data
    roc_trace = go.Scatter(
        x=fpr, y=tpr, mode='lines',
        name=f'ROC (AUC={auc:.4f})',
        line=dict(color='#B0D4F4', width=2.5),
        fill='tozeroy', fillcolor='rgba(176,212,244,0.12)'
    )
    roc_layout = pastel_layout('ROC-кривая', 'Доля ложных срабатываний (FPR)', 'Доля верных срабатываний (TPR)')
    roc_layout.shapes = [dict(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(color='#6c7a9e', width=1.2, dash='dash'))]
    roc_json = json.dumps({'data': [roc_trace], 'layout': roc_layout}, cls=plotly.utils.PlotlyJSONEncoder)

    # 2. PR кривая
    prec, rec = pr_data
    pr_trace = go.Scatter(
        x=rec, y=prec, mode='lines',
        name='PR-кривая',
        line=dict(color='#FBC4B0', width=2.5),
        fill='tozeroy', fillcolor='rgba(251,196,176,0.12)'
    )
    pr_layout = pastel_layout('Precision-Recall кривая', 'Полнота (Recall)', 'Точность (Precision)')
    pr_json = json.dumps({'data': [pr_trace], 'layout': pr_layout}, cls=plotly.utils.PlotlyJSONEncoder)

    # 3. Гистограмма вероятностей
    proba_fraud = y_proba[y_test == 1]
    proba_nonfraud = y_proba[y_test == 0]
    hist_trace1 = go.Histogram(x=proba_nonfraud, name='Не мошенничество', opacity=0.7, marker_color='#B0E4D4', nbinsx=30)
    hist_trace2 = go.Histogram(x=proba_fraud, name='Мошенничество', opacity=0.7, marker_color='#FBC4B0', nbinsx=30)
    hist_layout = pastel_layout('Распределение предсказанных вероятностей', 'Вероятность мошенничества', 'Количество транзакций')
    hist_layout.barmode = 'overlay'
    hist_json = json.dumps({'data': [hist_trace1, hist_trace2], 'layout': hist_layout}, cls=plotly.utils.PlotlyJSONEncoder)

    # 4. Важность признаков
    names = feature_importance['names'][::-1]
    values = feature_importance['values'][::-1]
    imp_trace = go.Bar(
        x=values, y=names, orientation='h',
        marker_color='#B0D4F4',
        text=[f'{v:.3f}' for v in values],
        textposition='outside',
        textfont=dict(color='#e0e4f0')
    )
    imp_layout = pastel_layout('Топ-10 важности признаков (Gain)', 'Важность', 'Признак')
    imp_layout.height = 420
    imp_json = json.dumps({'data': [imp_trace], 'layout': imp_layout}, cls=plotly.utils.PlotlyJSONEncoder)

    # 5. Матрица ошибок (статичная для оптимального порога)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    z = cm.tolist()
    cm_trace = go.Heatmap(
        z=z, x=['Не мошенничество', 'Мошенничество'], y=['Не мошенничество', 'Мошенничество'],
        colorscale=['#2a3348', '#B0D4F4'],
        showscale=True,
        text=z, texttemplate='%{text}', textfont=dict(size=13, color='#eef2ff')
    )
    cm_layout = pastel_layout('Матрица ошибок (при оптимальном пороге)', 'Предсказано', 'Реально')
    cm_layout.height = 350
    cm_json = json.dumps({'data': [cm_trace], 'layout': cm_layout}, cls=plotly.utils.PlotlyJSONEncoder)

    if hasattr(y_test, 'tolist'):
        y_test_list = [int(x) for x in y_test]  # ключевое изменение
    else:
        y_test_list = [int(x) for x in list(y_test)]
    if hasattr(y_proba, 'tolist'):
        y_proba_list = y_proba.tolist()
    else:
        y_proba_list = list(y_proba)

    return render_template('index.html',
                           metrics=metrics,
                           roc_json=roc_json,
                           pr_json=pr_json,
                           cm_json=cm_json,
                           hist_json=hist_json,
                           imp_json=imp_json,
                           y_test=y_test_list,
                           y_proba=y_proba_list,
                           sample_transactions=sample_transactions)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)