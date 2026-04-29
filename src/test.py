import numpy as np
a = np.array([[[1,2],[3,4]],[[5,6],[7,8]]])
print(a)
print()
b = a.transpose(1,0,2)
print(b)