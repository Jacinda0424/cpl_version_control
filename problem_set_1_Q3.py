import numpy as np

data = np.array([58.9,7.8,108.6,44.8,11.1,19.2,61.9,30.5,12.7,35.8,
                7.4,39.3,24,62.1,24.3,55.3,32.7,65.3,-19.3,7.6,-5.2,
                -2.1,31,69,88.6,39.5,20.7,89,69,64.9,64.8])

Mean = np.mean(data)
print("Mean :", Mean)

Standard_Deviation =  np.std(data, ddof=1)
print("Standard Deviation :", Standard_Deviation)

n = len(data)
Sample_Standard_Error = np.std(data, ddof=1)/np.sqrt(n)
print("Sample Standard Error :", Sample_Standard_Error)

Confidence_interval_1 = Mean - 2.042*Sample_Standard_Error
Confidence_interval_2 = Mean + 2.042*Sample_Standard_Error
print("Confidence_interval : [", Confidence_interval_1, Confidence_interval_2, "]")

print("Degree of Freedom :", n-1)

print("Null hypothesis : The mean elevation shift is zero.")
print("Alternative hypothesis : The mean elevation shift is not zero.")

print("Test Statistic t :", (np.mean(data)-0)/(np.std(data, ddof=1)/np.sqrt(n)))
