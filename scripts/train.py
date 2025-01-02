import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pandas as pd


def load_data(input_path):
    # Загрузить данные
    df = pd.read_csv(input_path)
    return df

def train(C, solver):
      # Запуск MLflow
      with mlflow.start_run():
        # Логгирование параметров
        mlflow.log_param("C", C)
        mlflow.log_param("solver", solver)
        # Загружаем данные
        df = load_data("data/proc_iris_dataset.csv")
        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(df.drop("target", axis=1), df["target"], test_size=0.3, random_state=42)
        # Обучение модели
        model = LogisticRegression(C=C, solver=solver)
        model.fit(X_train, y_train)
        # Предсказания
        y_pred = model.predict(X_test)
        # Логгирование метрик
        accuracy = accuracy_score(y_test, y_pred)
        mlflow.log_metric("accuracy", accuracy)
        # Логгирование модели
        mlflow.sklearn.log_model(model, "model")
        
if __name__ == "__main__":
    # Пример запуска эксперимента с разными параметрами
    train(C=0.1, solver='liblinear')
    train(C=1.0, solver='lbfgs')