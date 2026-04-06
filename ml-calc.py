import math
from typing import List, Tuple

class MLCalculator:
    """Калькулятор для базовых операций машинного обучения."""

    @staticmethod
    def mean(data: List[float]) -> float:
        """Среднее арифметическое."""
        if not data:
            raise ValueError("Список пуст")
        return sum(data) / len(data)

    @staticmethod
    def variance(data: List[float], sample: bool = True) -> float:
        """
        Дисперсия.
        sample=True - выборочная дисперсия (деление на n-1)
        sample=False - генеральная дисперсия (деление на n)
        """
        if len(data) < 2 and sample:
            raise ValueError("Для выборочной дисперсии нужно минимум 2 элемента")
        mu = MLCalculator.mean(data)
        n = len(data)
        ss = sum((x - mu) ** 2 for x in data)
        return ss / (n - 1 if sample else n)

    @staticmethod
    def std(data: List[float], sample: bool = True) -> float:
        """Стандартное отклонение."""
        return math.sqrt(MLCalculator.variance(data, sample))

    @staticmethod
    def normalize(data: List[float]) -> List[float]:
        """Z-нормализация: (x - mean) / std."""
        mu = MLCalculator.mean(data)
        sigma = MLCalculator.std(data, sample=False)
        if sigma == 0:
            return [0.0 for _ in data]
        return [(x - mu) / sigma for x in data]

    @staticmethod
    def min_max_scale(data: List[float], feature_range: Tuple[float, float] = (0, 1)) -> List[float]:
        """Мин-макс масштабирование в заданный диапазон."""
        if not data:
            return []
        min_val, max_val = min(data), max(data)
        if min_val == max_val:
            return [feature_range[0]] * len(data)
        a, b = feature_range
        return [a + (x - min_val) * (b - a) / (max_val - min_val) for x in data]

    @staticmethod
    def mse(y_true: List[float], y_pred: List[float]) -> float:
        """Среднеквадратичная ошибка (MSE)."""
        if len(y_true) != len(y_pred):
            raise ValueError("Длины списков не совпадают")
        n = len(y_true)
        return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n

    @staticmethod
    def mae(y_true: List[float], y_pred: List[float]) -> float:
        """Средняя абсолютная ошибка (MAE)."""
        if len(y_true) != len(y_pred):
            raise ValueError("Длины списков не совпадают")
        n = len(y_true)
        return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n

    @staticmethod
    def r2_score(y_true: List[float], y_pred: List[float]) -> float:
        """Коэффициент детерминации R²."""
        ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
        y_mean = MLCalculator.mean(y_true)
        ss_tot = sum((yt - y_mean) ** 2 for yt in y_true)
        if ss_tot == 0:
            return 1.0
        return 1 - ss_res / ss_tot

    @staticmethod
    def linear_regression(X: List[float], y: List[float]) -> Tuple[float, float]:
        """
        Простая линейная регрессия y = slope * x + intercept.
        Возвращает (slope, intercept).
        """
        if len(X) != len(y):
            raise ValueError("X и y должны быть одинаковой длины")
        n = len(X)
        x_mean = MLCalculator.mean(X)
        y_mean = MLCalculator.mean(y)

        numerator = sum((X[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((X[i] - x_mean) ** 2 for i in range(n))

        if denominator == 0:
            slope = 0.0
        else:
            slope = numerator / denominator
        intercept = y_mean - slope * x_mean
        return slope, intercept

    @staticmethod
    def predict_lr(X: List[float], slope: float, intercept: float) -> List[float]:
        """Предсказание по обученной линейной регрессии."""
        return [slope * x + intercept for x in X]


# Демонстрация
if __name__ == "__main__":
    ml = MLCalculator()
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print("Данные:", data)
    print(f"Среднее: {ml.mean(data):.2f}")
    print(f"Дисперсия (выборочная): {ml.variance(data):.2f}")
    print(f"Стандартное отклонение: {ml.std(data):.2f}")
    print(f"Нормализованные данные: {ml.normalize(data)}")
    print(f"Мин-макс (0-1): {ml.min_max_scale(data)}")

    # Пример метрик
    y_true = [3, -0.5, 2, 7]
    y_pred = [2.5, 0.0, 2, 8]
    print(f"\nMSE: {ml.mse(y_true, y_pred):.3f}")
    print(f"MAE: {ml.mae(y_true, y_pred):.3f}")
    print(f"R²: {ml.r2_score(y_true, y_pred):.3f}")

    # Простая линейная регрессия
    X = [1, 2, 3, 4, 5]
    y = [2, 4, 6, 8, 10]  # идеальная зависимость y = 2x
    slope, intercept = ml.linear_regression(X, y)
    print(f"\nЛинейная регрессия: y = {slope:.2f} * x + {intercept:.2f}")
    predictions = ml.predict_lr(X, slope, intercept)
    print(f"Предсказания: {predictions}")