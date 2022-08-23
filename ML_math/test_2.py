# x라는 값이 입력되면 'ax+b'라는 계산식을 통해 값을 산출하는 예측 함수를 정의합니다.
# 예측 함수를 통해 예측값과 실제값 y 간의 차이를 계산합니다.
# a와 b를 업데이트 하는 규칙을 정의하고 이를 바탕으로 a와 b의 값을 조정합니다. (alpha 값을 이용하여 규제 항을 설정합니다.)
# 위의 과정을 특정 반복횟수 만큼 반복합니다.
# 반복적으로 수정된 a와 b를 바탕으로 'y=ax+b'라는 회귀식을 정의합니다.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("Agg")

# 학습률(learning rate)를 설정한다. (권장 : 0.0001 ~ 0.01)
learning_rate = 0.05
# 반복 횟수(iteration)를 설정한다. (자연수)
iteration = 100
# 릿지회귀에 사용되는 알파(alpha) 값을 설정한다. (권장 : 0.0001 ~ 0.01)
alpha = 0.05

def prediction(a,b,x):
    pred = a * x + b
    return pred
    
def update_ab(a,b,x,error,lr, alpha):
    #alpha와 a의 곱으로 regularization을 설정한다.  
    regularization = alpha * a
    delta_a = -(lr*(2/len(error))*(np.dot(x.T, error)) + regularization)
    delta_b = -(lr*(2/len(error))*np.sum(error))
    return delta_a, delta_b
    
def gradient_descent(x, y, iters, alpha):
    #초기값 a= 0, a=0
    a = np.zeros((1,1))
    b = np.zeros((1,1))    
    
    for i in range(iters):
        error = y - prediction(a, b, x)
        a_delta, b_delta = update_ab(a,b,x,error,lr=learning_rate, alpha=alpha)
        a -= a_delta
        b -= b_delta
    
    return a, b

def plotting_graph(x,y,a,b):
    y_pred=a[0,0]*x+b
    plt.scatter(x, y)
    plt.plot(x, y_pred)
    plt.savefig("test.png")
    eu.send_image("test.png")

def main():
    #x, y 데이터를 생성한다.
    x = 5*np.random.rand(100,1)
    y = 10*x**4 + 2*x + 1+ 5*np.random.rand(100,1)
    # a와 b의 값을 반복횟수만큼 업데이트하고 그 값을 출력한다. 
    a, b = gradient_descent(x,y,iters=iteration, alpha=alpha)
    print("a:",a, "b:",b)
    #회귀 직선과 x,y의 값을 matplotlib을 통해 나타낸다.
    plotting_graph(x,y,a,b)
    
main()