# ML_Math

## 01_Math. for ML
**keyword**
```
AI  ML
vector matrix
```
**Python**
```
* 들여쓰기 + 세미콜론 없음
* 별도의 컴파일 필요 없음. Interpreter 방식으로 결과 바로 확인 가능
* 여러가지자료형
```
**Numpy&Pandas**
```
np.array
np.zeros
pd.Series
pd.DataFrame
```
**Matplotlib.pyplot**
```
plt.plot
plt.axis
plt.show
plt.xlabel
plt.ylabel
plt.title
plt.legend
```
<br>

---

<br>

## 02_기초 선형대수학

<br>

### 01.벡터의 정의와 의미
```
* 크기와 방향을 모두 고려함
* 리스트형태로 저장되어 있음
```

### 02.벡터 연산, 단위 벡터, 직교성
```
* 벡터에서의 음수 = 180도 반대방향
* 영벡터 :  자기자신 2차원 (0,0)  /  3차원 (0,0,0)
* 벡터 덧셈 : 평행이동해서 더하기
* 벡터 뺄셈 : 덧셈을 응용
* 벡터 곱셈 : 각 차원별로 대응되는 원소 간의 곱하여 계산
* 𝜃: 𝐴와𝐵사이의 사이각
* cosine similarity : cosine 값을 내적과 단위 벡터의 활용으로 구할 수 있음, cosine
값을 알면 두 벡터 사이의 각도도 역함수를 통해 알 수 있음
* 두 벡터가 직교함 : 𝑐𝑜𝑠𝜃=0인 경우를 의미함
```


### 03.행렬의 정의와 의미
```
* 벡터를 합치면 행렬의 형태
* 행과 열로 벡터를 분리가능함
```


### 04.행렬 연산과 역행렬
```S
* 영행렬 : 자기자신 행렬
* 행렬 덧뺄셈 : 각 원소에 대응되는 것끼리
* 교환법칙: 덧셈은 성립가능. 곱셈은 안됨
* 주대각 : 정사각행렬의 대각선에 위치한 성분. diag
* 대각합 : 정사각행렬의 주대각 원소의 합. Trace
* 단위행렬 : 주대각선 원소가 모두 1이고, 나머지가 0인 정사각행렬
* 역행렬 : 정사각행렬A 에 대해 곱해서 단위행렬이 나오게하는 행렬
```

## 03_ 미분법
**keyword**
```
함수의 극한
미분과 도함수
지수함수, 로그함수의 미분
편미분
```

## 04_ Gradient Descent
```
* 최적화 문제는 목적함수의 출력값을 최대 혹은 최소화하는 파라미터를 찾는 문제이다
* 경사하강법은 접선의 기울기를 이용하는 방법으로, 목적함수의 최소화가 목표일 때 사용한다.
```
```
<Gradient Descent를 사용하는 이유>
1. 데이터가 많을 때 계산량 측면에서 효율적이다.
2. 일상생활 속의 복잡한 고차원 함수의 학습에도 용이하다.
3. 미분 계수 구하는 과정에서 컴퓨터를 이용한 gradient descent가 더 편하다.
```