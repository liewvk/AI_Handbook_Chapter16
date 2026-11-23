import numpy as np

scores = np.array([1.0, 0.0])
d_k = 2
scaled = scores / np.sqrt(d_k)
shifted = scaled - scaled.max()
weights = np.exp(shifted) / np.exp(shifted).sum()

print("Weights:", np.round(weights, 3))
print("Total:", round(weights.sum(), 3))

