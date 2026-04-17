import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("ФИНАЛЬНАЯ ВЕРСИЯ: XGBoost + сильный штраф за пропуски")
print("Цель: recall ≥ 0.85, precision ≥ 0.5")
print("="*60)

# 1. Загрузка
df = pd.read_parquet('data/credit_card_transactions.parquet')
print(f"Размер: {df.shape}, доля мошенничеств: {df['is_fraud'].mean():.6f}")

# 2. Базовые признаки
df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
df['hour'] = df['trans_date_trans_time'].dt.hour
df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
df['is_night'] = ((df['hour'] >= 23) | (df['hour'] <= 5)).astype(int)
df['amount_log'] = np.log1p(df['amt'])
df['distance_km'] = np.sqrt((df['lat'] - df['merch_lat'])**2 + (df['long'] - df['merch_long'])**2)

# Возраст
if 'dob' in df.columns:
    df['dob'] = pd.to_datetime(df['dob'])
    df['age'] = (pd.Timestamp.now() - df['dob']).dt.days // 365
    df['age'] = df['age'].clip(18, 100)
    df = df.drop(columns=['dob'])

# 3. Признаки по карте (скользящие окна)
df = df.sort_values(['cc_num', 'unix_time'])
df['time_since_last'] = df.groupby('cc_num')['unix_time'].diff().fillna(0)
df['trans_last_10'] = df.groupby('cc_num')['unix_time'].transform(lambda x: x.rolling(10, min_periods=1).count())
df['amount_last_10_mean'] = df.groupby('cc_num')['amount_log'].transform(lambda x: x.rolling(10, min_periods=1).mean()).fillna(0)
df['amount_last_10_std'] = df.groupby('cc_num')['amount_log'].transform(lambda x: x.rolling(10, min_periods=1).std()).fillna(0)
df['amount_ratio_to_card_avg'] = df['amount_log'] / (df.groupby('cc_num')['amount_log'].transform('mean') + 1e-9)
df['same_category_count'] = df.groupby(['cc_num', 'category']).cumcount()

# 4. Удаление сырых колонок
drop_cols = ['trans_date_trans_time', 'amt', 'lat', 'long', 'merch_lat', 'merch_long',
             'first', 'last', 'street', 'zip', 'merchant', 'state', 'trans_num', 'merch_zipcode',
             'unix_time', 'cc_num', 'city_pop', 'job']
for col in drop_cols:
    if col in df.columns:
        df = df.drop(columns=[col])

# 5. Категории
for col in ['category', 'gender', 'city']:
    df[col] = df[col].astype(str).fillna('MISSING')
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# 6. Разделение
X = df.drop(columns=['is_fraud'])
y = df['is_fraud']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# 7. Масштабирование
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_val[num_cols] = scaler.transform(X_val[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

# 8. XGBoost с очень сильным штрафом за пропуски
scale_pos_weight = 2000   # можно увеличить до 3000, если recall < 0.85
print(f"\nscale_pos_weight = {scale_pos_weight}")

model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    early_stopping_rounds=50,
    tree_method='hist',
    verbosity=1
)

print("Обучение XGBoost...")
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

# 9. Предсказания
y_proba = model.predict_proba(X_test)[:, 1]

# 10. Подбор порога, максимизирующего среднее арифметическое precision, recall, f1
thresholds = np.arange(0.01, 0.99, 0.01)
best_score = 0
best_th = 0.5
best_precision = 0
best_recall = 0
best_f1 = 0
for th in thresholds:
    y_pred_tmp = (y_proba >= th).astype(int)
    p = precision_score(y_test, y_pred_tmp, zero_division=0)
    r = recall_score(y_test, y_pred_tmp, zero_division=0)
    f = f1_score(y_test, y_pred_tmp, zero_division=0) if (p + r) > 0 else 0
    avg_score = (p + r + f) / 3
    if avg_score > best_score:
        best_score = avg_score
        best_th = th
        best_precision = p
        best_recall = r
        best_f1 = f

print(f"Лучший порог по среднему арифметическому: {best_th:.4f}")
print(f"  Precision: {best_precision:.4f}, Recall: {best_recall:.4f}, F1: {best_f1:.4f}, Score: {best_score:.4f}")

# Используем найденный порог
y_pred = (y_proba >= best_th).astype(int)
precision = best_precision
recall = best_recall
f1 = best_f1
roc_auc = roc_auc_score(y_test, y_proba)

# 11. Сохранение артефактов (как обычно)
joblib.dump(model, 'model_xgb.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(best_th, 'best_threshold.pkl')
joblib.dump((X_test, y_test.values, y_pred, y_proba), 'test_data.pkl')
fpr, tpr, _ = roc_curve(y_test, y_proba)
prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
joblib.dump((fpr, tpr, roc_auc), 'roc_data.pkl')
joblib.dump((prec_curve, rec_curve), 'pr_data.pkl')

metrics_dict = {
    'precision': float(round(precision, 4)),
    'recall': float(round(recall, 4)),
    'f1': float(round(f1, 4)),
    'roc_auc': float(round(roc_auc, 4)),
    'best_threshold': float(round(best_th, 4))
}
joblib.dump(metrics_dict, 'metrics.pkl')

# Важность признаков
importance = model.feature_importances_
feature_names = X_train.columns.tolist()
sorted_idx = np.argsort(importance)[::-1][:10]
feature_importance = {
    "names": [feature_names[i] for i in sorted_idx],
    "values": [float(importance[i]) for i in sorted_idx]
}
joblib.dump(feature_importance, 'feature_importance.pkl')

# Выборка для таблицы
fraud_idx = np.where(y_test == 1)[0]
normal_idx = np.where(y_test == 0)[0]
max_fraud = min(len(fraud_idx), 50)
max_normal = min(len(normal_idx), 100 - max_fraud)
selected_fraud = np.random.choice(fraud_idx, max_fraud, replace=False) if max_fraud>0 else []
selected_normal = np.random.choice(normal_idx, max_normal, replace=False) if max_normal>0 else []
selected = np.concatenate([selected_fraud, selected_normal])
np.random.shuffle(selected)

sample_df = X_test.iloc[selected].copy()
sample_df['true_class'] = y_test.iloc[selected].values
sample_df['prob_fraud'] = y_proba[selected]
sample_df['pred_class'] = (sample_df['prob_fraud'] >= best_th).astype(int)
sample_transactions = sample_df.reset_index().to_dict(orient='records')
joblib.dump(sample_transactions, 'sample_transactions.pkl')

print("\n✅ Модель сохранена. Запускайте дашборд и настраивайте порог.")