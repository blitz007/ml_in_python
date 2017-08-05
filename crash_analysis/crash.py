import numpy as np
filename='Airplane_Crashes_and_Fatalities_Since_1908.csv'
raw_data=open(filename,'rt')
data = np.loadtxt(raw_data, delimiter=",")
print (dataset.shape)
