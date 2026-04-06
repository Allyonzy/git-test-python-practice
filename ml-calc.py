import math

class MLCalculator:
    @staticmethod
    def mean(d): return sum(d)/len(d) if d else 0

    @staticmethod
    def var(d, sample=True):
        mu = MLCalculator.mean(d)
        ss = sum((x-mu)**2 for x in d)
        return ss/(len(d)-1 if sample else len(d))

    @staticmethod
    def std(d, sample=True): return math.sqrt(MLCalculator.var(d, sample))

    @staticmethod
    def norm(d):
        mu, sigma = MLCalculator.mean(d), MLCalculator.std(d, sample=False)
        return [(x-mu)/sigma if sigma else 0 for x in d]

    @staticmethod
    def scale(d, a=0, b=1):
        mn, mx = min(d), max(d)
        if mn == mx: return [a]*len(d)
        return [a + (x-mn)*(b-a)/(mx-mn) for x in d]

    @staticmethod
    def mse(yt, yp): return sum((a-b)**2 for a,b in zip(yt,yp))/len(yt)
    @staticmethod
    def mae(yt, yp): return sum(abs(a-b) for a,b in zip(yt,yp))/len(yt)

    @staticmethod
    def r2(yt, yp):
        ss_res = sum((a-b)**2 for a,b in zip(yt,yp))
        ss_tot = sum((a-MLCalculator.mean(yt))**2 for a in yt)
        return 1 - ss_res/ss_tot if ss_tot else 1

    @staticmethod
    def linreg(X, y):
        xm, ym = MLCalculator.mean(X), MLCalculator.mean(y)
        num = sum((X[i]-xm)*(y[i]-ym) for i in range(len(X)))
        den = sum((X[i]-xm)**2 for i in range(len(X)))
        slope = num/den if den else 0
        return slope, ym - slope*xm

    @staticmethod
    def predict(X, slope, intercept): return [slope*x + intercept for x in X]


if __name__ == "__main__":
    ml = MLCalculator()
    data = [1,2,3,4,5,6,7,8,9,10]
    print("Среднее:", ml.mean(data))
    print("Дисперсия:", ml.var(data))
    print("Ст.откл.:", ml.std(data))
    print("Нормализация:", ml.norm(data))
    print("Мин-макс:", ml.scale(data))

    y_true, y_pred = [3,-0.5,2,7], [2.5,0,2,8]
    print("MSE:", ml.mse(y_true, y_pred))
    print("MAE:", ml.mae(y_true, y_pred))
    print("R2:", ml.r2(y_true, y_pred))

    X, y = [1,2,3,4,5], [2,4,6,8,10]
    slope, intercept = ml.linreg(X, y)
    print(f"y = {slope}*x + {intercept}")
    print("Предсказания:", ml.predict(X, slope, intercept))