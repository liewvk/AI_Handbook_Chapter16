import numpy as np

length = 4
allowed = np.tril(np.ones((length, length), dtype=int))
print(allowed)

