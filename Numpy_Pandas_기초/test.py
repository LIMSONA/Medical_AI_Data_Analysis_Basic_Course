# 1번
import numpy as np

array = np.arange(8)
matrix=array.reshape((2,4))
print(matrix)

# 2번
import numpy as np
A = np.random.randint(0,100,[1,10])
B= np.random.randint(0,100,[10,1])
print(A+B)

#6번
import pandas as pd
df=pd.read_csv("./data/tree_data.csv")
# ./data/tree_data.csv 파일을 읽어서 작업해보세요!
print(df.sort_values("circumference",ascending=False).iloc[0,:])

#7번
df=pd.read_csv("./data/the_pied_piper_of_hamelin.csv")
print(pd.pivot_table(df[df["구분"]=="Rat"], index="일차", columns="성별", values="나이", aggfunc="mean"))
