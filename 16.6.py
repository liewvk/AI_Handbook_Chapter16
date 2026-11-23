import numpy as np

x = np.array([[1.0, 0.0],
              [0.0, 1.0],
              [1.0, 1.0]])

scores = (x @ x.T) / np.sqrt(x.shape[1])
scores -= scores.max(axis=1, keepdims=True)
weights = np.exp(scores)
weights /= weights.sum(axis=1, keepdims=True)
output = weights @ x

print("Weights:")
print(np.round(weights, 3))
print("Contextual vectors:")
print(np.round(output, 3))

