# ML_Basic

## Numpy
```
스칼라, 벡터, 행렬, 전치행렬
```
<br>

## Linear Regression

### 선형회귀분석
```
데이터를 가장 잘 설명하는 선(함수)를 찾아 새로운 데이터가 어떤 결과값을 가질지 예측
y= a*x + b
```
![Loss Function](https://uiuidev.com/static/036fba17ca53cf66530612c426bd625a/c1b63/loss.png)
### 다중회귀분석
![다중회귀분석](https://uiuidev.com/static/6f2534c3474fd2d4ef0192721c3287f9/c1b63/multi_loss.png)
```
다중 회귀 분석(Multiple Linear Regression)은 데이터의 여러 변수(features) XXX를 이용해 결과 YYY를 예측하는 모델입니다
```
### 다항식 회귀분석
```
* MSE란 평균제곱오차를 의미하며, 통계적 추정에 대한 정확성의 지표로 널리 이용됩니다.

* 교차 검증이란 모델이 결과를 잘 예측하는지 알아보기 위해 전체 데이터를 트레이닝(training) 세트와 테스트(Test) 세트로 나누어 모델에 넣고 성능을 평가하는 방법입니다. 트레이닝 데이터는 모델을 학습시킬 때 사용되고, 테스트 데이터는 학습된 모델을 검증할 때 사용됩니다.
* random_state는 트레이닝과 테스트 데이터 그룹을 나눌 때 사용되는 난수 시드

* 모델을 복잡하게 만들면 트레이닝 데이터에 대한 정확도를 높일 수 있지만, 동일한 모델을 테스트 데이터에 적용하면 과적합(Overfitting) 현상이 일어나게 됩니다.
과적합 현상은 트레이닝 데이터에만 적합하게끔 모델이 작성되어, 테스트 데이터 또는 실제 데이터에 대해 모델이 잘 적용되지 않는 현상입니다.
```
<br>

## Naive Bayes Classifier

### 베이즈법칙 
![베이즈법칙](https://uiuidev.com/static/62c9252a353fc6000c812847cc37e80a/c1b63/base.png)

### 나이즈 베이즈 분류기
```
<ML> 
- 지도학습- 회귀분석 / 분류
- 비지도학습- 클러스터링
- 강화학습

* 지도학습이란 얻고자 하는 답으로 구성된 데이터
* Classification: 주어진 데이터가 어떤 클래스에 속하는지 알아내는 방법을 자동으로 학습하는 알고리즘
```
![나이즈베이즈분류기](https://uiuidev.com/static/20a928b8e6e86aaa5f60da8689cb3f31/3e096/naivebayes.png)
```
* Likelihood: prior probability P(A)가 주어졌을 때, P(X|A)를 likelihood라 함
테스트하고 싶은 모델이 데이터를 얼마나 잘 표현하는지 알 수 있음
```

### Bag of words와 감정분석
```
Bag of Words
* Bag안에 word를 넣고 종류랑 빈도를 정리한 것
* 자연어 텍스트 문장에서 특수 무자를 제거한 후 토크나이즈 Tokenize
* 단어 순서는 중요하지 않고, 빈도가 중요함
```

## K-Means 클러스터링
### 비지도학습
```
* 비지도학습이란 답이 정해져있지 않은 데이터에서 숨겨진 구조를 파악
- 차원축소(PCA), 클러스터링

*hard clustering : 데이터 포인트들은 비슷한 것들끼리 뭉쳐있다
*soft clustering : 한 개의 데이터 포인트는 숨겨진 클러스터들의 결합이다

* PCA 차원축소
: 주성분 분석(Principal Component Analysis, PCA)은 고차원 데이터를 저차원의 데이터로 변환하여 데이터를 정제하는 알고리즘입니다.
```
### K- Means 클러스터링
```
* 클러스터링이란 클러스터링, 또는 클러스터 분석은 주어진 개체에서 비슷한 개체를 선별하고 묶는(grouping) 작업입니다.

* K-Means 클러스터링은 주어진 데이터를 K개의 클러스터로 묶는 알고리즘입니다
=> K의 개수를 조정하면 클러스터의 일반도를 조정할 수 있습니다.
```