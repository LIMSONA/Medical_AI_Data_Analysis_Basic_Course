# 실습: 나이브 베이즈 분류기

import re
import math
import numpy as np

def main():
    M1 = {'r': 0.7, 'g': 0.2, 'b': 0.1} # M1 기계의 사탕 비율
    M2 = {'r': 0.3, 'g': 0.4, 'b': 0.3} # M2 기계의 사탕 비율
    
    test = {'r': 4, 'g': 3, 'b': 3}

    print(naive_bayes(M1, M2, test, 0.7, 0.3))

def naive_bayes(M1, M2, test, M1_prior, M2_prior):
    
    R1 = M1['r']**test['r'] * M1['g']**test['g'] * M1['b']**test['b'] * M1_prior
    R2 = M2['r']**test['r'] * M2['g']**test['g'] * M2['b']**test['b'] * M2_prior
   
    # 두 값의 합이 1이되어야하기 때문에
    R1,R2 = R1/(R1+R2), R2/(R1+R2)
    return [R1,R2]

if __name__ == "__main__":
    main()


