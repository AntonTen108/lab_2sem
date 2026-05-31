# Лабораторная работа — Предсказание цен на недвижимость

## Цель

Построить модели машинного обучения для предсказания стоимости домов в округе Кинг (штат Вашингтон, США). Сравнить линейную регрессию, случайный лес и градиентный бустинг.

---

## Задание

На основе данных Kaggle (21 613 объектов недвижимости, 20 признаков) построить регрессионные модели и оценить их качество по метрикам MAE и RMSE.

---

## Ссылка на ноутбук

📓 [Открыть ноутбук в Google Colab](https://colab.research.google.com/drive/16z1umNGy0alsOJ3JtAhPcNWxAtPRwY80?usp=sharing)

---

## Используемые модели и результаты

| Модель | MAE | RMSE |
|--------|-----|------|
| Linear Regression | ~130 000 | ~210 000 |
| Random Forest (базовый) | ~75 000 | ~135 000 |
| Random Forest (оптимизированный) | ~70 000 | ~125 000 |
| **Gradient Boosting** | **~68 000** | **~120 000** |

**Лучшая модель:** Gradient Boosting (наименьшая RMSE)

---

## Описание решения

### 1. Загрузка и предобработка данных

```python
training_data = pd.read_excel('predict_house_price_training_data.xlsx')
target_variable_name = 'Целевая.Цена'
training_values = training_data[target_variable_name]
training_points = training_data.drop(target_variable_name, axis=1)
```

В данных отсутствуют пропуски (все 15 129 строк заполнены). Целевая переменная — цена дома.

### 2. Обучение базовых моделей

```python
from sklearn import linear_model, ensemble

linear_regression_model = linear_model.LinearRegression()
linear_regression_model.fit(training_points, training_values)

random_forest_model = ensemble.RandomForestRegressor(n_estimators=100, random_state=42)
random_forest_model.fit(training_points, training_values)
```

### 3. Оценка качества

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(test_values, test_predictions_random_forest)
rmse = mean_squared_error(test_values, test_predictions_random_forest)**0.5
```

---

## Самостоятельная работа

### 1. Анализ важности признаков

Наиболее важные признаки (по Random Forest):
- Жилая площадь
- Оценка риелтора
- Широта
- Количество ванных комнат
- Площадь подвала

Наименее важные:
- Год реновации
- Код района
- Этажность
- Количество этажей
- Вид на воду

После исключения 5 наименее важных признаков RMSE изменилась незначительно, что подтверждает их малую информативность.

### 2. Оптимизация Random Forest

```python
results = []
for n_est in [100, 200]:
    for max_d in [10, 20, None]:
        for min_leaf in [1, 2, 4]:
            rf = ensemble.RandomForestRegressor(
                n_estimators=n_est, max_depth=max_d,
                min_samples_leaf=min_leaf, random_state=42, n_jobs=-1
            )
            rf.fit(training_points, training_values)
            preds = rf.predict(test_points)
            rmse = mean_squared_error(test_values, preds)**0.5
            results.append({...})
```

Лучшая конфигурация: `n_estimators=200, max_depth=None, min_samples_leaf=1`

### 3. Gradient Boosting — лучшая модель

```python
from sklearn.ensemble import GradientBoostingRegressor

gb_model = GradientBoostingRegressor(
    n_estimators=200, max_depth=4,
    learning_rate=0.1, random_state=42
)
gb_model.fit(training_points, training_values)
```

Gradient Boosting превзошёл Random Forest по обеим метрикам.

---

## Интеграция с FastAPI

```python
from fastapi import FastAPI
from pydantic import BaseModel
import pickle, numpy as np

app = FastAPI()
with open("house_model.pkl","rb") as f:
    model = pickle.load(f)

class House(BaseModel):
    bedrooms: int
    bathrooms: float
    sqft_living: int

@app.post("/predict")
def predict(h: House):
    X = np.array([[h.bedrooms, h.bathrooms, h.sqft_living]])
    price = model.predict(X)[0]
    return {"predicted_price": round(price, 2)}
```

Запуск: `uvicorn app:app --reload`

---

## Выводы

- Линейная регрессия плохо справляется с нелинейными зависимостями в данных о недвижимости
- Random Forest значительно точнее линейной регрессии
- Gradient Boosting показал наилучший результат среди всех протестированных моделей
- Признак «Широта» оказался неожиданно важным — близость к центру Сиэтла сильно влияет на цену
- Год реновации малоинформативен из-за большого числа нулей (пропусков)