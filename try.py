import numpy as np
layers = [2 ** i for i in range(0, 3)] *2
num_receptive_fields = np.sum(layers)
print(num_receptive_fields)