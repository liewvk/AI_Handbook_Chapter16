import numpy as np

query = np.array([1.0, 0.0])
keys = np.array([[1.0, 0.0], [0.0, 1.0]])
scores = keys @ query

print("Attention scores:", scores)

