import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import optuna


# Load the data
df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')


# Feature engineering
def engineer_features(df):
    # Add a column for the mean of each row
    df['mean'] = df.mean(axis=1)
    # Add a column for the standard deviation of each row
    df['std'] = df.std(axis=1)
    # Add a column for the median of each row
    df['median_balance'] = df.median(axis=1)
    # Add a column for the max of each row
    df['max'] = df.max(axis=1)
    # Add a column for the min of each row
    df['min'] = df.min(axis=1)
    # Отношение максимального баланса к минимальному балансу за период 90 дней:
    df['max_min_balance_ratio'] = df.max(axis=1) / df.min(axis=1)
    # Рассчитать, насколько баланс за период 90 дней увеличился или уменьшился в процентах относительно начального значения
    df['balance_percent_change'] = (df.iloc[:, -1] - df.iloc[:, 0]) / df.iloc[:, 0] * 100
    # Add skewness and kurtosis as features
    df['skewness'] = df.skew(axis=1)
    df['kurtosis'] = df.kurtosis(axis=1)
    # Добавить признак, показывающий изменение баланса на последнем дне относительно первого дня в периоде 90 дней
    df['balance_change'] = df.iloc[:, -1] - df.iloc[:, 0]
    # Создать признак, отображающий количество дней, когда баланс был отрицательным/положительным
    df['negative_balance_days'] = df[df < 0].count(axis=1)
    df['positive_balance_days'] = df[df > 0].count(axis=1)

    # Минимальный баланс за последние 30 дней
    df['min_last_30_days'] = df.iloc[:, -30:].min(axis=1)
    df['max_last_30_days'] = df.iloc[:, -30:].max(axis=1)

    df['min_middle_days'] = df.iloc[:, 30:60].min(axis=1)
    df['max_middle_days'] = df.iloc[:, 30:60].max(axis=1)

    # Максимальный баланс за первые 30 дней
    df['max_first_30_days'] = df.iloc[:, :30].max(axis=1)
    df['min_first_30_days'] = df.iloc[:, :30].min(axis=1)
    # Количество дней, когда баланс менялся на определенный процент (например, на 5%)
    threshold_percent = 5
    df[f'balance_change_{threshold_percent}%'] = ((df.iloc[:, :-1] / df.iloc[:, 1:]) - 1).abs().apply(
        lambda x: (x >= threshold_percent / 100).sum(), axis=1)

    # Среднее значение баланса в выходные дни
    df['mean_weekend_balance'] = df.iloc[:,
                                 [5, 6, 12, 13, 19, 20, 26, 27, 33, 34, 40, 41, 47, 48, 54, 55, 61, 62, 68, 69, 75, 76,
                                  82, 83]].mean(axis=1)

    # Медианное значение баланса в дни зарплаты:
    df['median_payday'] = df.iloc[:, [0, 14, 28, 42, 56, 70, 84]].median(axis=1)

    # Mean for the first 30 days
    df['mean_first_30'] = df.iloc[:, :30].mean(axis=1)
    # Mean for the next 30 days
    df['mean_next_30'] = df.iloc[:, 30:60].mean(axis=1)
    # Mean for the last 30 days
    df['mean_last_30'] = df.iloc[:, -30:].mean(axis=1)

    df['mean_first_last_ratio'] = df.iloc[:, :30].mean(axis=1) / df.iloc[:, -30:].mean(axis=1)

    # median for the first 30 days
    df['median_first_30'] = df.iloc[:, :30].median(axis=1)
    # median for the next 30 days
    df['median_next_30'] = df.iloc[:, 30:60].median(axis=1)
    # median for the last 30 days
    df['median_last_30'] = df.iloc[:, -30:].median(axis=1)

    df['median_first_last_ratio'] = df.iloc[:, :30].median(axis=1) / df.iloc[:, -30:].median(axis=1)

    # Std for the first 30 days
    df['std_first_30'] = df.iloc[:, :30].std(axis=1)
    # Std for the next 30 days
    df['std_next_30'] = df.iloc[:, 30:60].std(axis=1)
    # Std for the last 30 days
    df['std_last_30'] = df.iloc[:, -30:].std(axis=1)

    df['std_first_last_ratio'] = df.iloc[:, :30].std(axis=1) / df.iloc[:, -30:].std(axis=1)

    # Max for the first 30 days
    df['max_first_30'] = df.iloc[:, :30].max(axis=1)
    # Max for the next 30 days
    df['max_next_30'] = df.iloc[:, 30:60].max(axis=1)
    # Max for the last 30 days
    df['max_last_30'] = df.iloc[:, -30:].max(axis=1)

    df['max_first_last_ratio'] = df.iloc[:, :30].max(axis=1) / df.iloc[:, -30:].max(axis=1)

    # Min for the first 30 days
    df['min_first_30'] = df.iloc[:, :30].min(axis=1)
    # Min for the next 30 days
    df['min_next_30'] = df.iloc[:, 30:60].min(axis=1)
    # Min for the last 30 days
    df['min_last_30'] = df.iloc[:, -30:].min(axis=1)

    df['min_first_last_ratio'] = df.iloc[:, :30].min(axis=1) / df.iloc[:, -30:].min(axis=1)

    # Создание столбца дат на основе идентификатора временного ряда и номера дня
    df['date'] = pd.to_datetime((df['id'] - 1) * 90 + df.groupby('id').cumcount(), origin='2022-01-01')
    # Извлечение дня недели из столбца дат и создание нового столбца 'weekday'
    df['weekday'] = df['date'].dt.weekday

    df.drop(columns=['date'])

    return df


def metric(true, pred, coef=1000):
    assert len(true) == len(pred)
    error = 0
    for i in range(len(true)):
        if true[i] > pred[i]:
            error += true[i] - pred[i]
        else:
            error += (pred[i] - true[i]) * coef
    return -round(error / len(true) / 1_000_000_000)


X = df_train.drop(columns=['id', 'day 90'])
y = df_train['day 90']

# Применение feature engineering
X = engineer_features(X)


def train_model_for_study(X, y, model):
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42
    )

    model.fit(
        X_train, y_train,
        early_stopping_rounds=300,
        eval_set=[(X_valid, y_valid)],
        verbose=False
    )

    pred = model.predict(X_valid)
    return metric(y_valid.tolist(), [0.95 * x for x in pred]), model


# Define the objective function for Optuna
def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int("n_estimators", 1000, 10000),
        'reg_alpha': trial.suggest_float("reg_alpha", 1e-8, 100.0, log=True),
        'reg_lambda': trial.suggest_float("reg_lambda", 1e-8, 100.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0, step=0.1),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 1.0, log=True),
        'max_depth': trial.suggest_int("max_depth", 2, 9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.1, 1.0),
        'tree_method': 'gpu_hist',
        'predictor': 'gpu_predictor'
    }

    model = XGBRegressor(
        booster="gbtree",
        objective="reg:squarederror",
        random_state=42,
        **params
    )

    return train_model_for_study(X, y, model)[0]


# Tune hyperparameters using Optuna
study = optuna.create_study(direction="maximize")
study.optimize(objective_xgb, n_trials=10)

# Train the model on the entire training set using the best hyperparameters
params = study.best_params
model = XGBRegressor(**params)
model = train_model_for_study(X, y, model)[1]

# Make predictions on the test set
X_test = df_test.drop(columns=['id', 'day 90'])
X_test = engineer_features(X_test)
pred = model.predict(X_test)
df_test['day 91'] = pred * 0.95
df_test["day 91"] = df_test["day 91"].round()
df_test[['id', 'day 91']].to_csv('samplesubmission2.csv', index=False)
