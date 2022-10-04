import numpy as np

def main():
    print(matrix_tutorial())

def matrix_tutorial():
    # Create the matrix A here...
    a=[
        [1,4,5,8],
        [2,1,7,3],
        [5,4,5,9]
    ]
    A= np.array(a)
    
    return A

if __name__ == "__main__":
    main()
    
#============================================
def main():
    print(matrix_tutorial())

def matrix_tutorial():
    A = np.array([[1,4,5,8], [2,1,7,3], [5,4,5,9]])

    # 아래 코드를 작성하세요.
    A = A/np.sum(A)
    
    return np.var(A)

if __name__ == "__main__":
    main()
    
#============================================
def main():
    A = get_matrix()
    print(matrix_tutorial(A))

def get_matrix():
    mat = []
    [n, m] = [int(x) for x in input().strip().split(" ")]
    for i in range(n):
        row = [int(x) for x in input().strip().split(" ")]
        mat.append(row)
    return np.array(mat)

def matrix_tutorial(A):
    
    # 아래 코드를 완성하세요.
    B = A.transpose()
    try:
        C = np.linalg.inv(B)
    except: 
        return "not invertible"
    return np.sum(C>0)
if __name__ == "__main__":
    main()