import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.tree import plot_tree
from matplotlib.colors import LinearSegmentedColormap
import phik


def numeric_feature_plot(df, features):
    """
    Отрисовка распределения и выбросов для n числовых признаков.
    :param df: DataFrame
    :param features: список из имён числовых колонок
    """
    if not len(features):
        raise ValueError("Функция ожидает признаки")

    num_fig = len(features)
    fig, axes = plt.subplots(num_fig, 2, figsize=(14, 12))
    fig.suptitle('Анализ числовых признаков', fontsize=16, fontweight='bold')

    for idx, feature in enumerate(features):
        # Гистограмма с KDE
        sns.histplot(data=df, x=feature, bins='auto', kde=True, ax=axes[idx, 0],
                     color='steelblue')
        axes[idx, 0].set_title(f'Распределение: {feature}')
        axes[idx, 0].set_xlabel('Значение')
        axes[idx, 0].set_ylabel('Частота')

        # Boxplot для выбросов
        sns.boxplot(data=df, x=feature, ax=axes[idx, 1], color='lightcoral')
        axes[idx, 1].set_title(f'Выбросы: {feature}')
        axes[idx, 1].set_xlabel('Значение')

    plt.tight_layout()
    plt.show()


# Функция отрисовки распределений категориальных переменных
def categorical_feature_plot(df, feature):
  h = len(list(df[feature].unique()))//2
  plt.figure(figsize=(7, h))
  sns.countplot(df[feature])
  plt.title(f'Кол-во значений в каждой категории {feature}')
  plt.show()


# Матрица корреляций
def plot_phik(data, figsize=(12, 8)):
    phik_matrix = data.phik_matrix()
    plt.figure(figsize=figsize)
    sns.heatmap(phik_matrix, annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.show()


def plot_numeric_relationship(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    target_col: str = None
):
    """
    Строит join plot зависимости между двумя числовыми переменными.
    При наличии бинарной таргетной переменной — точки окрашиваются по её значению.

    :param df: pandas DataFrame
    :param x_col: Название числовой переменной по оси X
    :param y_col: Название числовой переменной по оси Y
    :param target_col: (опционально) Название бинарной переменной для окраски точек
    """
    # Проверка колонок
    for col in [x_col, y_col, target_col] if target_col else [x_col, y_col]:
        if col not in df.columns:
            raise ValueError(f"Колонка '{col}' отсутствует в DataFrame.")

    # Проверка типов
    if not pd.api.types.is_numeric_dtype(df[x_col]):
        raise TypeError(f"{x_col} не является числовой переменной.")
    if not pd.api.types.is_numeric_dtype(df[y_col]):
        raise TypeError(f"{y_col} не является числовой переменной.")

    # Проверка бинарного таргета
    if target_col is not None:
        unique_vals = sorted(df[target_col].dropna().unique())
        if len(unique_vals) != 2:
            raise ValueError(
                f"Таргет '{target_col}' должен быть бинарным (2 уникальных значения).")

    chart = sns.jointplot(data=df,
                          x=x_col,
                          y=y_col,
                          hue=target_col)
    chart.fig.suptitle(f'Совместные распределения {y_col} от {x_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)

    plt.show()


def categorical_vs_num_violinplot(df, col_cat, col_num, figsize=(8, 4)):
    df_plot = df.copy()
    # Преобразуем в строку, сейчас у нас числа 0 и 1
    df_plot[col_cat] = df_plot[col_cat].astype(str)

    plt.figure(figsize=figsize)
    sns.violinplot(data=df_plot,
                   x=col_num,
                   y=col_cat,
                   hue=col_cat)
    plt.title(f'violinplot по {col_cat} и {col_num}')

    plt.show()


def plot_feature_importance(df, importance='coef', moel_name='LinearRegression',
                            x_param='',
                            figsize=(12, 10)):
  plt.figure(figsize=figsize)
  plt.barh(df['feature'], df[importance], color='skyblue')
  plt.title(f'Важность признаков в {moel_name}')
  plt.xlabel(f'{x_param}')
  plt.gca().invert_yaxis()  # самый важный сверху
  plt.tight_layout()

  plt.show()


def visualize_decision_tree_reg(model, feature_names=None,
                                figsize=(20, 10), max_depth=None):
    from sklearn.tree import plot_tree

    """
    Visualize the structure of a trained DecisionTreeRegressor.

    Parameters:
    - model: Trained DecisionTreeRegressor
    - feature_names: List of feature names
    - figsize: Figure size
    - max_depth: Maximum depth to display (None for full tree)
    """
    plt.figure(figsize=figsize)
    plot_tree(model,
              feature_names=feature_names,
              fontsize=9,         # шрифт
              filled=True,
              rounded=True,
              proportion=False,   # Игнорируется для регрессии, но явно отключаем
              impurity=True,      # Показывает MSE/дисперсию в узлах
              node_ids=True,      # Номера узлов для отладки путей
              precision=2,        # Округление значений для читаемости
              max_depth=max_depth)
    plt.title('Decision Tree Regressor Visualization', fontsize=16)
    plt.tight_layout()

    plt.show()


def plot_regression_results(metrics, model_name="Model"):
    """
    Plot regression evaluation results

    Parameters:
    -----------
    metrics : dict
        Dictionary containing all metrics (output from calculate_classification_metrics)
    model_name : str, optional
        Name of the model for display purposes
    """
    plt.figure(figsize=(15, 6))

    # Plot 1: Confusion Matrix
    if 'Confusion Matrix' in metrics:
        plt.subplot(1, 2, 1)
        sns.heatmap(metrics['Confusion Matrix'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Predicted Negative', 'Predicted Positive'],
                    yticklabels=['Actual Negative', 'Actual Positive'])
        plt.title(f'{model_name} - Confusion Matrix', fontsize=14)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)

    plt.tight_layout()
    plt.show()


def true_vs_pred_plot(model, X_test_preprocessed, y_test, figsize=(10, 10),
                      is_y_log=False):
  y_pred = model.predict(X_test_preprocessed)

  # обратное преобразование к исходной шкале
  if is_y_log:
    y_pred = np.exp(y_pred)

  plt.figure(figsize=figsize)
  plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k')

  # Линия идеального предсказания
  min_val = min(y_test.min(), y_pred.min())
  max_val = max(y_test.max(), y_pred.max())
  plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Идеальное предсказание')

  plt.xlabel('Фактические значения')
  plt.ylabel('Предсказанные значения')
  plt.title('Actual vs Predicted')
  plt.legend()
  plt.grid(alpha=0.3)
  plt.tight_layout()
  plt.show()

def plot_hist_categorical(data, feature, figsize=(4, 4)):
    category_counts = data[feature].value_counts()
    category_counts = category_counts.sort_values(ascending=False)
    plt.figure(figsize=figsize)
    plt.grid()
    sns.barplot(x=category_counts.values,
                y=category_counts.index,
                hue=category_counts.index,  # Add this
                palette="viridis",
                orient='h',
                legend=False)  # Add this
    plt.title(f'Distribution of {feature}')
    plt.ylabel(feature)
    plt.xlabel('Frequency')
    plt.show()


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


def plot_hyperparam_search_results(
        results,
        score_key='mean_test_score',
        title='Hyperparameter Tuning Results',
        xtick_step=5
):
    # Normalize input
    if isinstance(results, dict):
        params = results.get('params')
        scores = results.get(score_key)
        if params is None or scores is None:
            raise ValueError(
                f"'params' and '{score_key}' must exist in results dict.")
        df = pd.DataFrame(params)
        df[score_key] = scores
    elif isinstance(results, pd.DataFrame):
        if 'params' in results.columns:
            df = pd.DataFrame(results['params'].tolist())
            df[score_key] = results[score_key].values
        else:
            raise ValueError("DataFrame input must have a 'params' column.")
    else:
        raise TypeError(
            "results must be a dict (like cv_results_) or a DataFrame.")

    df = df.reset_index().rename(columns={'index': 'Set #'})

    # Best score
    best_idx = df[score_key].idxmax()
    best_score = df.loc[best_idx, score_key]

    # Plot
    plt.figure(figsize=(12, 6))
    x = df['Set #']
    y = df[score_key]
    plt.plot(x, y, marker='o', linestyle='-')
    plt.title(title)
    plt.xlabel("Hyperparameter Set #")
    plt.ylabel(score_key)
    plt.grid(True)

    # Clean x-ticks
    plt.xticks(ticks=x[::xtick_step])

    # Highlight best
    plt.plot(df.loc[best_idx, 'Set #'], best_score,
             'ro', label=f'Best: {best_score:.4f}')
    plt.annotate(f'Best\n{best_score:.4f}',
                 xy=(df.loc[best_idx, 'Set #'], best_score),
                 xytext=(df.loc[best_idx, 'Set #'], best_score + 0.02),
                 arrowprops=dict(facecolor='red', shrink=0.05),
                 ha='center')

    plt.legend()
    plt.tight_layout()
    plt.show()

    return df


def compare_metrics_heatmap(df1, df2,
                            figsize=(8, 4), annot_fontsize=10,
                            title='Comparison of ML Metrics'):
    '''
    Значения отрицательные - значит метрика ухудшилась
    положительное - улучшилась
    Изначально в датафреймах метрики с отрицательными значениями
    '''
    # Calculate delta (difference) between DataFrames
    delta = df2 - df1

    # Create a custom red-white-green colormap
    colors = ["#ff2700", "#ffffff", "#00b975"]  # Red -> White -> Green
    cmap = LinearSegmentedColormap.from_list("rwg", colors)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    sns.heatmap(
        delta,
        annot=True,
        fmt=".3f",
        cmap=cmap,
        center=0,
        linewidths=.5,
        ax=ax,
        annot_kws={"size": annot_fontsize},
        cbar_kws={'label': 'Difference df2 - df1)'}
    )

    # Customize plot
    ax.set_title(title, pad=20, fontsize=14)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.tight_layout()

    return fig, delta

