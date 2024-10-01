import pandas as pd
from One_RKM import One_RKM
import numpy as np

Train_data = pd.read_csv("./Train_data.csv")
Train_data = np.array(Train_data)

Test_data = pd.read_csv("./Test_data.csv")
Test_data = np.array(Test_data)

c_1=1
c_2=1
sig = 1

accuracy, time = One_RKM(Train_data, Test_data, c_1, c_2,sig)
print(accuracy)
                        