import numpy as np
from sympy import Derivative

def main():
    x = 5
    print(derivative(x))
    
    
def derivative(x):
    '''
    먼저 다항함수 $x^2+1$을 선언하세요.그리고 선언한 다항함수의 미분 함수를 정의하고, 이것에  x를 넣은 값을 출력하세요.
    '''
    p = np.poly1d([1,0,1])
    print(p)
    
    q = p.deriv()
    print(q)
    
    return q(x)

if __name__ == "__main__":
    main()
    
    
#==============================================    
# d_fun(x) 함수 안에 x2+1x^2+1x^2 +1의 다항함수를 선언합니다. 그리고 함수의 극한에 활용할 분모값 h=1e-5를 활용해 미분값을 근사합니다.
import numpy as np
from sympy import symbols, Limit

def main():
    x = 5
    print(d_fun(x))
    
def d_fun(x):
    '''
    함수 안에 x^2+1의 다항함수를 선언합니다. 그리고 함수의 극한에 활용할 분모값 h=1e-5를 활용해 미분값을 근사합니다. 미분값의 근사는 미분계수를 구할 때 활용한 극한식을 참고하세요.
    '''
    p = np.poly1d([1,0,1])
    print(p)
    h = 1e-5

    return (p(x+h)-p(x))/h


if __name__ == "__main__":
    main()

#==============================================
#d_fun(x,y, respect = 'y') x,y의 점에서 y에 대한 편미분 값을 구하는 함수를 구현합니다.
#주어진 h값을 활용하여 respect=’x’일때, x에 대한 편미분, 그리고 respect=’y’일 때 y에 대한 편미분을 구합니다.

import numpy as np

def main():
    x = 5
    y = 2
    print(d_fun(x,y, respect = 'x'))
    print(d_fun(x,y, respect = 'y'))

def fun(x,y):
    
    return x**3 * y

def d_fun(x,y, respect = 'y'):
    '''
     x,y의 점에서 y에 대한 편미분 값을 구하는 함수입니다. 함수의 극한을 응용합니다. 주어진 h값을 활용해, x에 대한 미분은 respect='x'일때, 그리고 y에 대한 미분은 respect='y'일 때입니다.
    '''
    h = 1e-5
    
    if respect == 'x':
        answer = (fun(x+h,y)-fun(x,y))/h
    elif respect == 'y':
        answer = (fun(x,y+h)-fun(x,y))/h
    else:
        raise NotImplementedError
    return answer