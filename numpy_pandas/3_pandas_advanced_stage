import numpy as np
import pandas as pd

print("Masking & query")
df = pd.DataFrame(np.random.rand(5, 2), columns=["A", "B"])
print(df, "\n")

#데이터 프레임에서 A컬럼값이 0.5보다 작고 B컬럼 값이 0.3보다 큰값들을 구해봅시다.
df[ (df["A"]<0.5) & (df["B"]>0.3) ]
df.query("A<0.5 and B>0.3")

#===============================
df = pd.DataFrame(np.arange(5), columns=["Num"])
print(df, "\n")

# 값을 받으면 제곱을 해서 돌려주는 함수
def square(x):
    return x**2


# apply로 컬럼에 함수 적용
df["Square"] = df["Num"].apply(square)


# 람다 표현식으로도 적용하기
df["Square"]= df["Num"].apply(lambda x: square(x))
df["Square"]= df["Num"].apply(lambda x: x**2)

#===============================
df = pd.DataFrame({
    'key': ['A', 'B', 'C', 'A', 'B', 'C'],
    'data1': [1, 2, 3, 1, 2, 3],
    'data2': [4, 4, 6, 0, 6, 1]
})
print("DataFrame:")
print(df, "\n")

# groupby 함수를 이용해봅시다.
# key를 기준으로 묶어 합계를 구해 출력해보세요.
print(df.groupby("key").sum())



# key와 data1을 기준으로 묶어 합계를 구해 출력해보세요.
print(df.groupby(["key","data1"]).sum())

#===============================
df = pd.DataFrame({
    'key': ['A', 'B', 'C', 'A', 'B', 'C'],
    'data1': [0, 1, 2, 3, 4, 5],
    'data2': [4, 4, 6, 0, 6, 1]
})
print("DataFrame:")
print(df, "\n")

# aggregate를 이용하여 요약 통계량을 산출해봅시다.
# 데이터 프레임을 'key' 칼럼으로 묶고, data1과 data2 각각의 최솟값, 중앙값, 최댓값을 출력하세요.
print(df.groupby("key").aggregate([min,np.median,max]))

# 데이터 프레임을 'key' 칼럼으로 묶고, data1의 최솟값, data2의 합계를 출력하세요.
print(df.groupby("key").aggregate({"data1":min, "data2":sum}))

#===============================
df = pd.DataFrame({
    'key': ['A', 'B', 'C', 'A', 'B', 'C'],
    'data1': [0, 1, 2, 3, 4, 5],
    'data2': [4, 4, 6, 0, 6, 1]
})
print("DataFrame:")
print(df, "\n")

# groupby()로 묶은 데이터에 filter를 적용해봅시다.
# key별 data2의 평균이 3이 넘는 인덱스만 출력해봅시다.

print("filtering : ")

def filter_by_mean (x):
    return x['data2']. mean () > 3
print(df.groupby('key').mean())
print(df.groupby('key').filter (filter_by_mean))
print(df.groupby("key").filter(lambda x: x["data2"].mean()>3))

# groupby()로 묶은 데이터에 apply도 적용해봅시다.
# 람다식을 이용해 최댓값에서 최솟값을 뺀 값을 적용해봅시다.

print("applying : ")
print(df.groupby("key").apply(lambda x: x.max()-x.min()))

#===============================
# 파일을 읽어서 코드를 작성해보세요
# 경로: "./data/the_pied_piper_of_hamelin.csv"


df=pd.read_csv("./data/the_pied_piper_of_hamelin.csv")

#print(df.describe)

#child, 남자와 여자, 평균 연령
print(pd.pivot_table(df[df["구분"]=="Child"], index="일차", columns="성별", values="나이"))

#===============================
