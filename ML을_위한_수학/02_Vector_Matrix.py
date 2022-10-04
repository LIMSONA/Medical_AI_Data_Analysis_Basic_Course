# 참고:  https://www.delftstack.com/ko/howto/numpy/python-numpy-unit-vector/#%25EC%259E%2590%25EC%25B2%25B4-%25EC%25A0%2595%25EC%259D%2598-%25EC%25A0%2591%25EA%25B7%25BC-%25EB%25B0%25A9%25EC%258B%259D%25EC%259C%25BC%25EB%25A1%259C-numpy-%25EB%25B0%25B0%25EC%2597%25B4%25EC%2597%2590%25EC%2584%259C-%25EB%258B%25A8%25EC%259C%2584-%25EB%25B2%25A1%25ED%2584%25B0-%25EA%25B0%2580%25EC%25A0%25B8-%25EC%2598%25A4%25EA%25B8%25B0
import numpy as np

def main():
    data = [1,2,3]
    
    print(calculate_norm(data))
    print(unit_vector(data))
    

def calculate_norm(data):
    '''
    지시사항1: 함수 안에 주어진 데이터 data의 크기(scale) 값을 출력하는 함수를 채우시오.
    '''
    
    
    return np.linalg.norm(data)

def unit_vector(data):
    '''
    지시사항2: 함수 입력으로 주어진 data 벡터를 단위 벡터(unit vector)로 바꾸어 출력하세요.
    '''
    vector=np.array(data)
    return vector / (vector**2).sum()**0.5
    
#==========================================

np.linalg.norm()

def normalize():
    '''
    지시사항: 함수 안에서 먼저 (5,5)의 랜덤 행렬을 선언하세요.그리고 선언한 행렬의 최대값과 최소값을 기억하고, 이것을 통해 선언했던 랜덤 행렬을 normalize하세요.
    '''
    
    Z = np.random.random((5,5))
    Zmax, Zmin = Z.max(), Z.min()
    Z = Z/np.sum(Z)
    
    return Z

# ??? 최대 최소 이용 잘모르겠엉....................ㅋㅋㅋㅋ
# ==============================


def multiply():
    '''
    함수 안에 먼저 1으로 채워진 행렬 2개를 선언합니다. 하나는 (5,3)의 크기, 다른 하나는 (3,2)로 합니다. 그리고 앞서 선언한 두 행렬의 곱을 합니다.
    '''
    A= np.ones((5,3))
    B= np.ones((3,2))
    
    Z = np.dot(A,B)
    
    return Z