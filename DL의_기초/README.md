# 텐서플로우와 딥러닝 학습방법

## Gradient descent 알고리즘
Gradient descent 알고리즘은 손실 함수(loss function)의 미분값인 gradient를 이용해 모델에게 맞는 최적의 가중치(weight), 즉 손실 함수의 값을 최소화 하는 가중치를 구할 수 있는 알고리즘입니다.  

**f_x = w0 + w1 * X**
<br>  

## 손실함수 (loss function)
손실 함수(loss function)는 실제값과 모델이 예측한 값 간의 차이를 계산해주는 함수입니다. 손실 함수의 값은 가중치와 편향을 업데이트하는 데에 사용됩니다.

* MSE는 평균 제곱 오차 함수입니다.  
<br>  

## 역전파 (Back propagation)
역전파(Back propagation)는 다층 퍼셉트론 모델을 이루는 가중치들을 개선하기 위해 개발된 여러 알고리즘들 중 가장 유명하고 널리 쓰이는 방법입니다.
<br>

## 텐서(Tensor) 데이터 생성 - 텐서플로우 자료형
* tf.float32 : 32-bit float
* tf.float64 : 64-bit float
* tf.int8 : 8-bit integer
* tf.int16 : 16-bit integer
* tf.int32 : 32-bit integer
* tf.uint8 : 8-bit unsigned integer
* tf.string : String
* tf.bool : Boolean
<br>

## 텐서(Tensor) 연산
* tf.add(x, y) : x 텐서와 y 텐서를 더합니다.
* tf.subtract(x, y) : x 텐서에서 y 텐서를 뺍니다.
* tf.multiply(x, y) : x 텐서와 y 텐서를 곱합니다.
* tf.truediv(x, y) : x 텐서를 y 텐서로 나눕니다.
* tf.square(): 제곱
* tf.reduce_mean(): 평균
<br>

## 텐서플로우&케라스를 이용한 MLP 모델 함수/메서드

* tf.keras.models.Sequential() : 연속적으로 층을 쌓아 만드는 Sequential 모델을 위한 함수
* model.complie() : 학습 방법 설정
* model.fit() : 모델 학습
* model.predict() : 학습된 모델로 예측값 생성
* tf.keras.layers.Dense(units, activation): 신경망 모델의 레이어를 구성하는데 필요한 keras 함수
* units: 레이어 안의 노드 수
* activation: 적용할 activation function