import numpy as np

query = np.array([1.0, 0.0])
keys = np.array([[1.0, 0.0], [0.0, 1.0]])
values = np.array([[10.0, 0.0], [0.0, 20.0]])

scores = (keys @ query) / np.sqrt(query.size)
shifted = scores - scores.max()
weights = np.exp(shifted) / np.exp(shifted).sum()
result = weights @ values

print("Weights:", np.round(weights, 3))
print("Output:", np.round(result, 3))

