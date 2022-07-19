# NumPy_Pandas_Matplotlib

## 01_Numpy

**numpy library에서 자주 사용되는 함수y**
```
np.array - 배열생성
np.zeros - 0이 들어있는 배열 생성
np.ones - 1이 들어있는 배열 생성
np.empty - 초기화가 없는 값으로 배열을 반환
np.arange(시작,끝미만,간격) - 배열 버전의 range 함수
np.linspace(시작,끝이하,갯수) - 갯수만큼 시작과 끝의 일정한 간격 배열생성
np.random - 다양한 난수가 들어있는 배열 생성
```
**array 속성 값 관련 함수**
```
arr.ndim - 배열 차원 확인
arr.shape - 행,열 확인
arr.size - 행*열 값 확인
arr.dtype - 배열데이터 타입 확인
```
**array 조작 함수**
```
arr.reshape(열,행) - 열,행 shape 변경
np.concatenate - array 이어붙임
np.split - array 분할
```
**집계함수**
```
np.sum - 합계
np.min - 최소
np.max - 최대
np.mean - 평균
```
<br>

## 02_Pandas
**pandas library에서 자주 사용되는 함수**
```
pd.series - 시리즈 데이터 생성
pd.DataFrame - 데이터 프레임 생성
pd.map -
pd.apply - 함수로 데이터 처리하기
```
**dataframe**
```
df.index - 인덱스 확인
df.columns - 열 확인
df.to_csv - csv파일 저장
df.to_excel - xls / xlsx파일 저장
df.read_csv - csv파일 열기
df.read_excel - xls / xlsx파일 열기
```
**df indexing/slicing 함수**
```
df.loc - 명시적인 인덱스 참조
df.iloc - 파이썬 스타일 정수 인덱스 참조
```
**df 함수**
```
df.isnull() - null값? T/F
df.notnll() - not null값? T/F
df.dropna() - null값 삭제
df.fillna() - null값 채움
df.sort_values() - 값으로 정렬하기
```

<br>

## 03_Pandas_advanced_stage
```
df.query - 조건으로 검색
```
**그룹**
```
df.groupby - 그룹으로 묶기
df.groupby(조건).sum - 그룹 기준 합계
df.groupby(조건).aggregate - 여러 집계를 한번에 계산
df.groupby(조건).mean - 그룹 기준 평균
df.groupby(조건).aggregate - 그룹 기준 필터링
df.groupby(조건).get_group - 그룹 안에서 key값으로 데이터 조회
```
**피벗**
```
pd.pivot_table(df,index= , columns= , values= , aggfunc= )
```
<br>

# 04_Matplotlib
**plot 기본**
```
plt.plot
plt.title - 제목
plt.xlable - x축 제목
plt.ylable - y축 제목
```
**fig, ax**
```
fig,ax=plt.subplots() - 그래프 1개
ax.plot
ax.set_title - 제목
ax.set_xlabel - x축 제목
ax.set_ylabel - y축 제목
ax.set_xlim - x축 최소,최대
ax.set_ylim - y축 최소,최대
fig.set_dip - 그래프그림 크기
fig.savefig - 그래프그림 저장
```
```
(ex. 2*1 그래프 그리기)
fig,axes=plt.subplots(2,1)
axes[0].plot(~~)
axes[1].plot(~~)
```
**lineplot**
```
ax.plot(x,y,
        linestyle= 라인 표시 종류   - -- -. :
        marker= 마커 표시 종류   . o v s *
        color= 라인 색   r green 0.8 #524FA1)
```
**legend**
```
ax.legend(loc= 위치 upper right / upper left / ~~~ / best
          shadow= 그림자 True / False
          fancybox= 둥근네모 박스모양 True / False
          boarderpad= 안쪽 여백
          facecolor= 박스배경색)
```
<br>

## 05_데이터 분석이란
```
주제 선정 > 데이터 구조 파악 > 데이터 전처리 > 데이터 분석
```
