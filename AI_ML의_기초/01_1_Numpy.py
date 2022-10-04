import numpy as np
A= np.array([1,2],
            [3,4])
print(A)

#산술연산
print(A**2) # 제곱
print(3**A) # 곱셈
print(A*A) # 제곱


#행렬곱셈
x= np.array([1,2],
            [3,4])
y= np.array([3,4],
            [3,2])

print(np.dot(x,y)) #행렬곱셈
print(x*y) #행렬 숫자 간 곱셈


#비교연산
a= np.array([1,2,3,4])
b= np.array([4,2,2,4])

print(a==b)
print(a>b)

#논리연산
c= np.array([1,1,0,0], dtype=bool)
d= np.array([1,0,1,0], dtype=bool)
np.logical_or(c,d)
np.logical_and(c,d)


#reductions
e=np.array([1,2,3,4,5])
np.sum(e) #!5
e.sum() #15

e.min()
e.max()
#자리 
e.argmin() #0
e.argmax() #4


#logical reductions
f=np.array([True,True,True])
g=np.array([True,True,False])

np.all(f) #True
np.all(g) #False

np.any(f) #True
np.any(g) #True


#Statistical Reductions
x=np.array([1,2,3,1])
print(np.mean(x)) #평균값
print(np.median(x)) #중간값
print(np.std(x)) #표준편차
