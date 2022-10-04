import numpy as np
import pandas as pd

# 두 개의 시리즈 데이터가 있습니다.
print("Population series data:")
population_dict = {
    'korea': 5180,
    'japan': 12718,
    'china': 141500,
    'usa': 32676
}
population = pd.Series(population_dict)
print(population, "\n")

print("GDP series data:")
gdp_dict = {
    'korea': 169320000,
    'japan': 516700000,
    'china': 1409250000,
    'usa': 2041280000,
}
gdp = pd.Series(gdp_dict)
print(gdp, "\n")


# 이곳에서 2개의 시리즈 값이 들어간 데이터프레임을 생성합니다.
print("Country DataFrame")
country = pd.DataFrame({'population':population, 'gdp': gdp})

print(country)


# 데이터 프레임에 gdp per capita 칼럼을 추가하고 출력합니다.
country["gdp per capita"]=country["gdp"]/country["population"]


# 데이터 프레임을 만들었다면, index와 column도 각각 확인해보세요.
print(country.index)
print(country.columns)

#===============================

# 첫번째 컬럼을 인덱스로 country.csv 파일 읽어오기.
print("Country DataFrame")
country = pd.read_csv("./data/country.csv", index_col=0)
print(country, "\n")

# 명시적 인덱싱을 사용하여 데이터프레임의 "china" 인덱스를 출력해봅시다.
print(country.loc["china"])


# 정수 인덱싱을 사용하여 데이터프레임의 1번째부터 3번째 인덱스를 출력해봅시다.
print(country.iloc[1:4])
#===============================

print("DataFrame: ")
df = pd.DataFrame({
    'col1' : [2, 1, 9, 8, 7, 4],
    'col2' : ['A', 'A', 'B', np.nan, 'D', 'C'],
    'col3': [0, 1, 9, 4, 2, 3],
})
print(df, "\n")


# 정렬 코드 입력해보기    
# Q1. col1을 기준으로 오름차순으로 정렬하기.
sorted_df1 = df.sort_values('col1', ascending = True)
# sorted_df1 = df.sort_values('col1') 동일함

# Q2. col2를 기준으로 내림차순으로 정렬하기.
sorted_df2 = df.sort_values('col2', ascending = False)


# Q3. col2를 기준으로 오름차순으로, col1를 기준으로 내림차순으로 정렬하기.
sorted_df3 = df.sort_values(['col2', 'col1'], ascending=[True, False])

#===============================

df= pd.read_csv('./data/tree_data.csv')
# ./data/tree_data.csv 파일을 읽어서 작업해보세요!

print(df.sort_values("height",ascending=False).iloc[0,2])
