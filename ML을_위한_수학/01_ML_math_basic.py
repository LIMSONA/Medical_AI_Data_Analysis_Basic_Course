import numpy as np

def main():
    print(tutorial_1st())
    print(tutorial_2nd())
    
def tutorial_1st():
    """
    지시사항 1.
    tutorial_1st() 함수 안에 5번째 값만 1로 가지고 이외의 값은 0을 가지는 
    길이 10의 벡터를 선언하세요.
    """
    A= np.zeros(10)
    A[4]=1   
    return A
    
def tutorial_2nd():
    """
    지시사항 2.
    tutorial_2nd() 함수 안에 10~49의 range를 가지는 벡터를 선언하세요.
    """    
    B = np.arange(10,50)
    return B

if __name__ == "__main__":
    main()
    
    
#===============================
import pandas as pd

def main():
    print(pandas_tutorial())
    
def pandas_tutorial():
    '''
    지시사항: `[2,4,6,8,10]`의 리스트를 pandas의 Series 자료구조로 선언하세요.
    '''
    A=pd.Series([2,4,6,8,10])
    return A

if __name__ == "__main__":
    main()
    

#============================
import matplotlib.pyplot as plt
def main():
    matplotlib_tutorial()

    
def matplotlib_tutorial():
    '''
    지시사항: data [1,2,3,4]를 정의하여서, plt.plot을 활용해 데이터를 시각화해보세요.
    '''
    data= [1,2,3,4]
    plt.plot(data)
    
    # 엘리스에서 이미지를 출력하기 위해서 필요한 코드입니다.
    plt.savefig("plot.png")
    elice_utils.send_image("plot.png")

if __name__ == "__main__":
    main()