# **Задание по ML Ops**
## Цель:
**В этом задании познакомимся с основами управления данными с помощью DVC, управления экспериментами с использованием MLflow и автоматизации с ClearML. 
Наша задача — интегрировать все три инструмента для построения полного цикла ML-проекта**

## Описание проекта

* Часть 1: Управление данными с DVC — включает добавление данных в управление с помощью DVC, настройку удаленного хранилища и интеграцию с Git Hub Action для автоматического запуска пайплайна.


* Часть 2: Управление экспериментами с MLflow — управление ML-экспериментами, регистрация метрик и моделей.

* Часть 3: Автоматизация экспериментов с ClearML — настройка ClearML для отслеживания задач, логирования метрик и автоматизации экспериментов

# Отчет по экспериментам MLflow

## Описание задачи

В данном отчете представлены результаты экспериментов по обучению модели логистической регрессии на наборе данных Iris. Целью экспериментов было сравнение производительности моделей с различными параметрами регуляризации (`C`) и алгоритмами оптимизации (`solver`).

## Данные

В качестве данных был использован набор данных Iris, загруженный и предобработанный с помощью файлов `download.py` и `process_data.py`.

## Эксперименты

Было проведено два эксперимента с разными параметрами модели логистической регрессии:

![Скриншот MLflow UI](https://github.com/kcherenkovv/ml_ops3/blob/main/screens/mlflow.png)

### Эксперимент 1

*   **Параметры:**
    *   `C`: 0.1
    *   `solver`: `liblinear`
*   **Метрики:**
    *   `accuracy`: 1.0
    *   `Duration`: 2.8s

### Эксперимент 2

*   **Параметры:**
    *   `C`: 1.0
    *   `solver`: `lbfgs`
*   **Метрики:**
    *   `accuracy`: 0.8444
    *   `Duration`: 9.0s

## Сравнение моделей

По результатам проведенных экспериментов можно сделать следующие выводы:

*   **Точность:** Модель, обученная с параметрами `C=0.1` и `solver='liblinear'`, достигла более высокой точности (`accuracy=1.0`), чем модель с параметрами `C=1.0` и `solver='lbfgs'` (`accuracy=0.8444`).
*   **Время обучения:** Модель с параметрами `C=1.0` и `solver='lbfgs'` обучалась значительно дольше (`Duration=9.0s`), чем модель с параметрами `C=0.1` и `solver='liblinear'` (`Duration=2.8s`).

## Визуализация результатов

**Первая модель:**


![Скриншот MLflow UI](https://github.com/kcherenkovv/ml_ops3/blob/main/screens/mlflow1.png)


**Вторая модель:**
![Скриншот MLflow UI](https://github.com/kcherenkovv/ml_ops3/blob/main/screens/mlflow2.png)


## Выводы

Эксперимент 1 (с параметрами `C=0.1` и `solver='liblinear'`) показал себя лучше с точки зрения точности и времени обучения, чем эксперимент 2. Модель обученная с данными параметрами показывает себя значительно лучше в данной постановке задачи.

## Рекомендации

*   Провести дополнительные эксперименты с разными параметрами и данными, чтобы убедиться в стабильности результатов.
*   Включить в отчет больше метрик, таких как F1-score, recall и precision.
*   Использовать графики из MLflow UI для более наглядного представления результатов.

## Зависимости:

* Python
* MLflow
* scikit-learn
* pandas
