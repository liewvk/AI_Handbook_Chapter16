import numpy as np

x = np.array([[1.0, 2.0], [3.0, 1.0]])

def attend(data):
    scores = data @ data.T / np.sqrt(data.shape[1])
    scores -= scores.max(axis=1, keepdims=True)
    weights = np.exp(scores)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights @ data

head1 = attend(x)
head2 = attend(x[:, ::-1])
combined = np.concatenate([head1, head2], axis=1)
print("Head 1 shape:", head1.shape)
print("Head 2 shape:", head2.shape)
print("Combined shape:", combined.shape)

