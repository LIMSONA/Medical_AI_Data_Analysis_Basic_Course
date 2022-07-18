#0부터 5사이 랜덤값 3x5 array
import numpy as np
array1= np.random.randint(0,5,(3,5))
print(array1)

array2= np.random.randint(5, size=(3,5))
print(array2)

#===============================
print("1차원 array")
array = np.arange(10)
print(array)

# Q1. array의 자료형을 출력해보세요.
print(type(array))

# Q2. array의 차원을 출력해보세요.
print(array.ndim)

# Q3. array의 모양을 출력해보세요.
print(array.shape)

# Q4. array의 크기를 출력해보세요.
print(array.size)

# Q5. array의 dtype(data type)을 출력해보세요.
print(array.dtype)

# Q6. array의 인덱스 5의 요소를 출력해보세요.
print(array[5])

# Q7. array의 인덱스 3의 요소부터 인덱스 5 요소까지 출력해보세요.
print(array[3:6])

#==============================
print("2차원 array")
matrix = np.arange(1, 16).reshape(3,5)  #1부터 15까지 들어있는 (3,5)짜리 배열을 만듭니다.
print(matrix)

# Q1. matrix의 자료형을 출력해보세요.
print(type(matrix))

# Q2. matrix의 차원을 출력해보세요.
print(matrix.ndim)

# Q3. matrix의 모양을 출력해보세요.
print(matrix.shape)

# Q4. matrix의 크기를 출력해보세요.
print(matrix.size)

# Q5. matrix의 dtype(data type)을 출력해보세요.
print(matrix.dtype)

# Q6. matrix의 (2,3) 인덱스의 요소를 출력해보세요.
print(matrix[2,3])

# Q7. matrix의 행은 인덱스 0부터 인덱스 1까지, 열은 인덱스 1부터 인덱스 3까지 출력해보세요.
print(matrix[0:2, 1:4])

#==============================
print("array")
array = np.arange(8)
print(array)
print("shape : ", array.shape, "\n")

# Q1. array를 (2,4) 크기로 reshape하여 matrix에 저장한 뒤 matrix와 그의 shape를 출력해보세요.
print("# reshape (2, 4)")
matrix = array.reshape(2,4)


print(matrix)
print("shape : ", matrix.shape)

#==============================
print("matrix")
matrix = np.array([[0,1,2,3],
                   [4,5,6,7]])
print(matrix)
print("shape : ", matrix.shape, "\n")

# (아래의 배열 모양을 참고하세요.)
# Q1. matrix 두 개를 세로로 붙이기 
'''
[[0 1 2 3]
 [4 5 6 7]
 [0 1 2 3]
 [4 5 6 7]]
'''
m=np.concatenate([matrix,matrix],axis=0)
print(m)

# Q2. matrix 두 개를 가로로 붙이기
'''
[[0 1 2 3 0 1 2 3]
 [4 5 6 7 4 5 6 7]]
'''

n=np.concatenate([matrix,matrix],axis=1)
print(n)

#===========================print("matrix")
matrix = np.array([[ 0, 1, 2, 3],
                   [ 4, 5, 6, 7],
                   [ 8, 9,10,11], 
                   [12,13,14,15]])
print(matrix, "\n")

# Q1. matrix를 [3] 행에서 axis 0으로 나누기
'''
[[0  1   2  3]
 [4  5   6  7]
 [8  9  10 11]],

 [12 13 14 15]
'''
a, b = np.split(matrix, [3], axis=0)

print(a, "\n")
print(b, "\n")


# Q2. matrix를 [1] 열에서 axis 1로 나누기
'''
[[ 0]
 [ 4]
 [ 8]
 [12]],

[[ 1  2  3]
 [ 5  6  7]
 [ 9 10 11]
 [13 14 15]]
'''

c, d = np.split(matrix,[1], axis=1)

print(c, "\n")
print(d)

#==============================
array = np.array([1,2,3,4,5])
print(array)

# Q1. array에 5를 더한 값을 출력해보세요.
print(array+5)

# Q2. array에 5를 뺀 값을 출력해보세요.
print(array-5)

# Q3. array에 5를 곱한 값을 출력해보세요.
print(array*5)

# Q4. array를 5로 나눈 값을 출력해보세요.
print(array/5)

# Q5. array에 array2를 더한 값을 출력해보세요.    
array2 = np.array([5,4,3,2,1])
print(array+array2)

# Q6. array에 array2를 뺀 값을 출력해보세요.
print(array-array2)

#==============================
'''
[[0]
 [1]
 [2]
 [3]
 [4]
 [5]] 배열 A와

 [0 1 2 3 4 5] 배열 B를 선언하고, 덧셈 연산해보세요.
'''

A=np.arange(6).reshape(6,1)
B=np.arange(6)

print(A)
print(B)

print(A+B)

#==============================
matrix = np.arange(8).reshape((2, 4))
print(matrix)

# Q1. sum 함수로 matrix의 총 합계를 구해 출력해보세요.
np.sum(matrix)

# Q2. max 함수로 matrix 중 최댓값을 구해 출력해보세요.
np.max(matrix)

# Q3. min 함수로 matrix 중 최솟값을 구해 출력해보세요.
np.min(matrix)

# Q4. mean 함수로 matrix의 평균값을 구해 출력해보세요.
np.mean(matrix)

# Q5. sum 함수의 axis 매개변수로 각 열의 합을 구해 출력해보세요.
np.sum(matrix,axis=1)

# Q6. sum 함수의 axis 매개변수로 각 행의 합을 구해 출력해보세요.
np.sum(matrix,axis=0)

# Q7. std 함수로 matrix의 표준편차를 구해 출력해보세요.
np.std(matrix)

# Q8. 마스킹 연산을 이용하여 matrix 중 5보다 작은 수들만 추출하여 출력해보세요.
print(matrix[matrix<5])

#==============================
#거짓말 0 
daily_liar_data = [0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]


# 양치기 소년이 거짓말을 몇 번 했는지 구하여 출력해주세요.
#print(np.sum(daily_liar_data))

#print(np.size(daily_liar_data))

print(np.size(daily_liar_data) - np.sum(daily_liar_data))
#==============================
