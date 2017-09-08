import numpy as np

x_data=np.linspace(-1,1,300)[:,np.newaxis]
a=np.array([1,2,3,4,5])
b=a[np.newaxis,:]
print x_data
print b
