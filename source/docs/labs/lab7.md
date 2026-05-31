# Лабораторная работа 6 — Очистка и трансформация данных (Pandas)

## Цель

Освоить методы очистки и трансформации данных с использованием библиотеки pandas на примере датасета Titanic.

---

## Ссылка на ноутбук

📓 [Открыть ноутбук в Google Colab](https://colab.research.google.com/drive/1VCjkySZZLPp1x39ageOMlTETc9gUV1BB?usp=sharing)

---

## Описание набора данных

**Titanic** — датасет с информацией о 891 пассажире. Содержит 12 столбцов:

| Признак | Описание |
|---------|----------|
| Survived | Выжил ли пассажир (0/1) |
| Pclass | Класс билета (1/2/3) |
| Name | Имя пассажира |
| Sex | Пол |
| Age | Возраст |
| SibSp | Братья/сёстры/супруги на борту |
| Parch | Родители/дети на борту |
| Fare | Стоимость билета |
| Cabin | Номер каюты |
| Embarked | Порт посадки |

---

## Первичный анализ

```python
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
df.head(10)
df.describe().T
```

**Пропуски в данных:**

| Столбец | Пропусков | Процент |
|---------|-----------|---------|
| Age | 177 | 19.9% |
| Cabin | 687 | 77.1% |
| Embarked | 2 | 0.2% |

---

## Обработка пропусков

```python
# Age — заполняем медианой (28 лет)
df["Age"] = df["Age"].fillna(df["Age"].median())

# Возрастные группы
df["Age_group"] = df["Age"].apply(lambda x:
    "Ребёнок" if x < 13 else
    "Подросток" if x < 18 else
    "Взрослый" if x < 60 else "Пожилой")

# Embarked — заполняем модой
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Cabin — первая буква
df["Cabin_letter"] = df["Cabin"].apply(lambda x: x[0] if pd.notna(x) else "Unknown")
```

---

## Трансформация данных

```python
# Pclass: 1->F, 2->S, 3->T
df["Pclass"] = df["Pclass"].map({1:"F", 2:"S", 3:"T"}).astype("category")

# Title из имени
df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")

# Sex -> 0/1
df["Sex_num"] = df["Sex"].map({"male": 0, "female": 1})

# FamilySize и IsAlone
df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
```

---

## Обработка выбросов

Метод IQR выявил выбросы в столбце `Fare`. Применена winsorization (95-й перцентиль):

```python
fare_95 = df["Fare"].quantile(0.95)
df["Fare_winsorized"] = df["Fare"].clip(upper=fare_95)

age_95 = df["Age"].quantile(0.95)
df["Age"] = df["Age"].clip(upper=age_95)
```

---

## Агрегация и анализ

| Класс | Выживаемость (%) |
|-------|-----------------|
| F (1-й) | 63% |
| S (2-й) | 47% |
| T (3-й) | 24% |

**Вывод:** пассажиры 1-го класса выживали значительно чаще.

---

## Метрики качества очистки

- Процент заполненных значений: **99.8%**
- Оставшихся пропусков: **0** (в обработанных столбцах)
- Создано новых признаков: **7** (Age_group, Cabin_letter, Title, Sex_num, FamilySize, IsAlone, Fare_winsorized)

---

## Заключение

- Данные Titanic содержат значительные пропуски, особенно в Cabin (77%)
- Пол и класс билета — наиболее значимые факторы выживаемости
- Winsorization эффективно устранила влияние экстремальных значений Fare
- Созданные признаки (Title, FamilySize, IsAlone) повышают информативность датасета для последующего ML-моделирования