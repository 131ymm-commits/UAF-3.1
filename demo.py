import numpy as np
from sklearn.preprocessing import StandardScaler
from numba import jit

@jit(nopython=True)
def _gamma_fast(values):
    if len(values) < 30:
        return 0.5
    base_std = np.std(values)
    if base_std < 1e-6:
        return 0.5
    entr = np.abs(np.diff(values)).mean() / base_std
    chaos = np.var(values[-20:]) / max(base_std, 0.01)
    gamma = entr / max(entr + chaos, 1e-6)
    return max(0.0, min(1.0, gamma))

def get_uaf_scores(series, window=30):
    scores = np.zeros_like(series, dtype=np.float64)
    if len(series) < 2 * window:
        return scores
    scaler = StandardScaler()
    norm_series = scaler.fit_transform(series.reshape(-1, 1)).ravel()
    for i in range(window, len(series)):
        curr = _gamma_fast(norm_series[i-window:i])
        if i >= 2 * window:
            prev = _gamma_fast(norm_series[i-2*window:i-window])
            drop = (prev - curr) / max(prev, 0.05)
            scores[i] = max(0.0, min(1.0, drop * 1.5))
    return scores

# Пример с искусственным рядом
np.random.seed(42)
t = np.linspace(0, 10, 500)
signal = np.sin(t) + 0.1 * np.random.randn(500)
# Вставляем аномалию
signal[300:320] += 2.0

scores = get_uaf_scores(signal)

print("UAF anomaly scores:", scores[290:330:5])
print("Max score:", np.max(scores))
