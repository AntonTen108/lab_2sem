# Лабораторная работа — Предсказание дефолта по кредиту

## Цель

Построить модель машинного обучения для классификации кредитных дефолтов. Изучить метрики качества классификаторов, реализовать и сравнить несколько алгоритмов.

---

## Задание

На основе данных из Kaggle-соревнования *"Give Me Some Credit"* обучить модели классификации, которые по анкетным данным клиента предсказывают вероятность дефолта (просрочки более 90 дней).

---

## Ссылка на ноутбук

📓 [Открыть ноутбук в Google Colab](https://colab.research.google.com/drive/1iYxVr8D-o0NoeKtoxUJPPsM63Bo8r4cx?usp=sharing)

> Ноутбук содержит полный код: загрузку данных, предобработку, обучение моделей, метрики качества и самостоятельную работу.

---

## Используемые модели

| Модель | ROC-AUC |
|--------|---------|
| Logistic Regression | ~0.70 |
| Random Forest | ~0.85 |
| SVM (rbf) | ~0.79 |
| kNN (k=5) | ~0.72 |

**Лучшая модель:** Random Forest с ROC-AUC ≈ 0.85

---

## Описание решения

### 1. Загрузка и предобработка данных

- Данные загружены с Dropbox (50 000 обучающих и 37 500 тестовых записей)
- Пропуски в столбцах `MonthlyIncome` и `NumberOfDependents` заполнены средними значениями из обучающей выборки
- Целевая переменная `SeriousDlqin2yrs` отделена от входных признаков

### 2. Обучение моделей

```python
from sklearn import linear_model, ensemble

logistic_regression_model = linear_model.LogisticRegression()
logistic_regression_model.fit(training_points, training_values)

random_forest_model = ensemble.RandomForestClassifier(n_estimators=100)
random_forest_model.fit(training_points, training_values)
```

### 3. Оценка качества

Использованы метрики:
- **Accuracy** — доля правильных предсказаний
- **Confusion Matrix** — матрица ошибок классификации
- **ROC-AUC** — площадь под ROC-кривой (основная метрика)

### 4. Самостоятельная работа — SVM и kNN

```python
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(training_points)
X_test_sc = scaler.transform(test_points)

svm_model = SVC(probability=True, kernel='rbf', C=1.0, random_state=42)
svm_model.fit(X_train_sc, training_values)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_sc, training_values)
```

---

## Современные алгоритмы классификации

По данным исследований Kaggle и научных публикаций:

| Алгоритм | ROC-AUC | Применение |
|---|---|---|
| XGBoost / LightGBM | 0.87–0.92 | Лидер в соревнованиях |
| Random Forest | 0.82–0.88 | Хорошая базовая модель |
| SVM | 0.78–0.85 | Малые датасеты |
| Logistic Regression | 0.70–0.78 | Интерпретируемость |
| Neural Networks | 0.85–0.93 | Большие данные |

---

## Интеграция модели с FastAPI

Пошаговый алгоритм развёртывания модели как веб-сервиса:

**1. Сохранить модель:**
```python
import pickle
with open("model.pkl", "wb") as f:
    pickle.dump(random_forest_model, f)
```

**2. Создать FastAPI приложение:**
```python
from fastapi import FastAPI
from pydantic import BaseModel
import pickle, numpy as np

app = FastAPI()
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

class Client(BaseModel):
    age: int
    MonthlyIncome: float
    DebtRatio: float

@app.post("/predict")
def predict(c: Client):
    X = np.array([[c.age, c.MonthlyIncome, c.DebtRatio]])
    prob = model.predict_proba(X)[0][1]
    return {"default_probability": round(prob, 4)}
```

**3. Запустить:**
```bash
pip install fastapi uvicorn
uvicorn app:app --reload
```

**4. Отправить запрос:**
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 45, "MonthlyIncome": 5000, "DebtRatio": 0.3}'
```

---

## Выводы

- Задача предсказания дефолта — задача бинарной классификации с несбалансированными классами (6% дефолтов)
- Accuracy — плохая метрика для несбалансированных данных; правильнее использовать ROC-AUC
- Random Forest показал лучшее качество среди протестированных моделей
- SVM и kNN требуют масштабирования данных и уступают Random Forest по ROC-AUC
- Для продакшена рекомендуется попробовать XGBoost/LightGBM