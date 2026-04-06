
class Calculator:
    """Класс калькулятора с поддержкой истории операций."""

    def __init__(self):
        self.history = []   # список кортежей (операция, аргументы, результат)

    def _add_to_history(self, operation, args, result):
        """Добавляет запись в историю."""
        self.history.append((operation, args, result))

    def add(self, a, b):
        """Возвращает сумму a + b."""
        result = a + b
        self._add_to_history("add", (a, b), result)
        return result

    def subtract(self, a, b):
        """Возвращает разность a - b."""
        result = a - b
        self._add_to_history("subtract", (a, b), result)
        return result

    def multiply(self, a, b):
        """Возвращает произведение a * b."""
        result = a * b
        self._add_to_history("multiply", (a, b), result)
        return result

    def divide(self, a, b):
        """
        Возвращает результат деления a / b.
        При делении на ноль возвращает None и записывает ошибку в историю.
        """
        if b == 0:
            self._add_to_history("divide (error)", (a, b), "ZeroDivisionError")
            return None
        result = a / b
        self._add_to_history("divide", (a, b), result)
        return result

    def power(self, a, b):
        """Возведение в степень: a ** b."""
        result = a ** b
        self._add_to_history("power", (a, b), result)
        return result

    def show_history(self):
        """Выводит историю всех операций."""
        if not self.history:
            print("История пуста.")
            return
        print("=== История операций ===")
        for idx, (op, args, res) in enumerate(self.history, 1):
            if "error" in op:
                print(f"{idx}. {op} {args} -> {res}")
            else:
                print(f"{idx}. {op}{args} = {res}")

    def clear_history(self):
        """Очищает историю."""
        self.history.clear()
        print("История очищена.")


if __name__ == "__main__":
    # Демонстрация работы класса
    calc = Calculator()

    print("Калькулятор готов (более сложная версия)")
    print(f"5 + 3 = {calc.add(5, 3)}")
    print(f"10 - 4 = {calc.subtract(10, 4)}")
    print(f"7 * 6 = {calc.multiply(7, 6)}")
    print(f"20 / 4 = {calc.divide(20, 4)}")
    print(f"2 ** 10 = {calc.power(2, 10)}")

    # Проверка деления на ноль
    print(f"10 / 0 = {calc.divide(10, 0)}")

    # Вывод истории
    calc.show_history()

    # Очистка и повторный вывод
    calc.clear_history()
    calc.show_history()