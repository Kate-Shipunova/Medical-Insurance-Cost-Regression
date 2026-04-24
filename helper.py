import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (OrdinalEncoder, OneHotEncoder, StandardScaler,
                                   MinMaxScaler, TargetEncoder)
from imblearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import cross_validate
from sklearn.preprocessing import PowerTransformer


def divide_data(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]
    return X, y


def get_quality_metrics(model):
  # Рассчитаем метрики качества с применением кросс-валидации на 5 фолдах
  scoring = {
      'mae':  'neg_mean_absolute_error',
      'mse':  'neg_mean_squared_error',
      'r2':   'r2',
      'rmse': 'neg_root_mean_squared_error'  # доступно в sklearn >= 1.1
  }

  cv_results = cross_validate(
      model,
      X_train_preprocessed,
      y_train,
      cv=5,
      scoring=scoring,
      n_jobs=-1
  )

  # sklearn возвращает ОТРИЦАТЕЛЬНЫЕ значения для ошибок,
  # т.к. его оптимизатор всегда МАКСИМИЗИРУЕТ скор.
  print("R² (mean):   ", cv_results['test_r2'].mean())
  print("MAE (mean):  ", -cv_results['test_mae'].mean())
  print("MSE (mean):  ", -cv_results['test_mse'].mean())
  print("RMSE (mean): ", -cv_results['test_rmse'].mean())

  return {key: round(cv_results[f'test_{key}'].mean(), 2) for key in scoring.keys()}


def fit_models(models, preprocessor, X, y, cv_strategy, seed=42, is_log_y=False,
               is_box_cox=False):
    all_metrics = {}
    for model_name, model in models:
        # Клонируем, чтобы не менять исходные объекты
        current_model = clone(model)
        if is_log_y:
            # используем трансформер таргета перед вычислением метрик
            current_model = TransformedTargetRegressor(
                regressor=current_model,
                func=np.log,
                inverse_func=np.exp
            )

        if is_box_cox:
          # Создаём трансформер Box-Cox
          boxcox_transformer = PowerTransformer(method='box-cox',
                                                standardize=False)
          # Используем в модели BoxCoxTransformer
          current_model = TransformedTargetRegressor(
              regressor=current_model,
              transformer=boxcox_transformer
          )


        current_preprocessor = clone(preprocessor)

        all_metrics[model_name] = train_evaluate_model_cv(
            current_model, model_name, X, y, current_preprocessor, cv_strategy, seed
        )

    metrics_df = pd.DataFrame.from_dict(all_metrics, orient='index')

    plt.figure(figsize=(8, 4))
    sns.heatmap(metrics_df, cmap='RdBu_r', annot=True, fmt=".2f")
    plt.title('Model Evaluation Metrics Comparison')
    plt.tight_layout()
    plt.show()

    return metrics_df


def train_evaluate_model_cv(model, model_name, X, y, preprocessor, cv_strategy, seed=None):
    # Установка seed
    if seed is not None:
        if hasattr(model, 'random_state'):
            model.set_params(random_state=seed)
        if hasattr(model, 'seed'):
            model.set_params(seed=seed)

    # Оборачиваем в Pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    scoring = {
        'mae': 'neg_mean_absolute_error',
        'mse': 'neg_mean_squared_error',
        'r2': 'r2',
        'rmse': 'neg_root_mean_squared_error'  # доступно в sklearn >= 1.4
    }

    # Передаем СЫРЫЕ X и y. Pipeline сам разберется с y внутри каждого фолда.
    cv_results = cross_validate(
        pipeline,
        X, y,
        cv=cv_strategy,
        scoring=scoring,
        return_train_score=False
    )

    metrics = {key: round(cv_results[f'test_{key}'].mean(), 4) for key in scoring.keys()}
    return metrics


def create_preprocessor(delete_feature=False, without_feature=None):
      '''
      На этапе анализа корреляций среди признаками обнаружено много дублирующих
      Все признаки - кандидаты на удаление - числовые, поэтому в данной ф-ции
      реализуем сборку пайплайна без одного или нескольких таких признаков,
      указанных в without_feature
      '''
      categorical_cols = ['sex', 'region', 'smoker', 'alcohol_freq']
      ordinal_cols = ['urban_rural', 'education', 'marital_status',
                      'employment_status', 'plan_type', 'network_tier']
      numeric_cols = ['age', 'income', 'household_size', 'dependents',
                      'bmi', 'visits_last_year', 'hospitalizations_last_3yrs',
                      'days_hospitalized_last_3yrs', 'medication_count', 'systolic_bp',
                      'diastolic_bp', 'ldl', 'hba1c', 'deductible', 'copay',
                      'policy_term_years', 'policy_changes_last_2yrs',
                      'provider_quality', 'risk_score',
                      'annual_premium', 'monthly_premium', 'claims_count',
                      'avg_claim_amount', 'total_claims_paid', 'chronic_count',
                      'hypertension', 'diabetes', 'asthma', 'copd',
                      'cardiovascular_disease', 'cancer_history', 'kidney_disease',
                      'liver_disease', 'arthritis', 'mental_health',
                      'proc_imaging_count', 'proc_surgery_count', 'proc_physio_count',
                      'proc_consult_count', 'proc_lab_count', 'is_high_risk',
                      'had_major_procedure']
      print(f'Стартовое кол-во числовых фич: {len(numeric_cols)}')
      if delete_feature:
        for feature in without_feature:
          if feature in numeric_cols:
            numeric_cols.remove(feature)
      print(f'Итоговое кол-во числовых фич: {len(numeric_cols)}')

      # пропуски в этих колонках будут заполняться наиболее частыми значениями
      most_frequent_cat_cols = ['alcohol_freq']

      # Выпишем все категории для каждого столбца в порядке увеличения номера
      category_orders = [
          ['Urban', 'Suburban', 'Rural'],                          # urban_rural
          ['No HS', 'HS', 'Some College', 'Bachelors',             # education
          'Masters', 'Doctorate'],
          ['Single', 'Married', 'Divorced', 'Widowed'],            # marital_status
          ['Self-employed', 'Employed',  'Unemployed', 'Retired'], # employment_status
          ['HMO', 'EPO', 'POS', 'PPO'],                            # plan_type
          ['Bronze', 'Silver', 'Gold', 'Platinum']                 # network_tier
      ]

      # Создаем предобработчик данных в виде последовательного Pipeline
      preprocessor = Pipeline([
          # Шаг 1: Заполнение пропусков alcohol_freq наиболее частой категорией
          ('nan_remover', ColumnTransformer(
              [
                  ('most_frequent_cat', SimpleImputer(strategy='most_frequent'),
                   most_frequent_cat_cols)
              ],
              remainder='passthrough',  # колонки, не указанные в трансформерах, передаются без изменений
              verbose_feature_names_out=False  # не добавлять префиксы к именам колонок
          )),

          # Шаг 2: Кодирование категориальных признаков (зависит от завершения этапа импутации)
          ('transformations', ColumnTransformer(
              [
                  ('cat', OneHotEncoder(drop='first', handle_unknown='ignore',
                                        sparse_output=False), categorical_cols),
                  ('ord', OrdinalEncoder(categories=category_orders), ordinal_cols)
              ],
              remainder='passthrough',  # числовые колонки передаются без изменений
              verbose_feature_names_out=False
          ))
      ])

      # Устанавливаем вывод в формате pandas DataFrame
      preprocessor.set_output(transform="pandas")

      return preprocessor


# Получение очищенного от бизнесово-ошибочных фич датасета
def get_clear_df(df):
    del_features = ['monthly_premium', 'annual_premium', 'avg_claim_amount',
                    'total_claims_paid']
    df.drop(del_features, axis=1, inplace=True)

    return df


def evaluate_model(model, model_name, X_train, y_train, X_test, y_true, seed=None):
    from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                                 r2_score, mean_absolute_percentage_error,
                                 median_absolute_error, explained_variance_score)

    # Работаем с копией модели, чтобы не изменять исходные модели
    model = clone(model)

    # Train the model
    model.fit(X_train, y_train)

    # Get predictions
    y_pred = model.predict(X_test)

    # Evaluate
    mse = mean_squared_error(y_true, y_pred)
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'mse': mse,
        'rmse': np.sqrt(mse),
        'r^2': r2_score(y_true, y_pred)
    }

    # Convert metrics to DataFrame
    metrics_df = pd.DataFrame.from_dict(metrics, orient='index')

    # Plot heatmap
    plt.figure(figsize=(8, 4))
    sns.heatmap(metrics_df, cmap='RdBu_r', annot=True, fmt=".4f")
    plt.title('Model Evaluation Metrics Comparison')
    plt.tight_layout()
    plt.show()

    return metrics_df
