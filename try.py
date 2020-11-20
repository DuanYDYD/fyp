import numpy as np
layers = [2 ** i for i in range(0, 2)] * 8
num_receptive_fields = np.sum(layers)
print(num_receptive_fields)

#45
#3 12?
#2 28?