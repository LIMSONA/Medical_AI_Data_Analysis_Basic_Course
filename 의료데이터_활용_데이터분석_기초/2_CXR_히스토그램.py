# 흉부 X선 이미지 히스토그램 그리기

import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import elice_utils


def create_histogram(images, channels, mask, histSize, ranges):   
    histr = cv.calcHist(images, channels, mask, histSize, ranges) 
    return histr

def create_mask(img, x_range, y_range):
    mask = np.zeros(img.shape[:2], np.uint8)
    #x축의 x1~x2 구간, y축의 y1~y2 구간에 255의 값을 할당합니다.
    mask[x_range[0]:x_range[1], y_range[0]:y_range[1]] = 255
    #cv.bitwise_and()함수를 활용하여 masked_imag를 구현합니다.
    masked_img = cv.bitwise_and(img,img,mask = mask)
    return mask, masked_img

    
def main():
    img = cv.imread('normal.png')
    '''지시사항 1번과 2번에 따라 아래 적절한 값을 입력하세요.'''
    # 현재 제공하는 이미지는 gray scale입니다.
    channels = [0] # 흑백 이미지의 경우 [0]
    mask = None # 전체 영역에 대한 계산을 원할 경우
    histSize = [256]
    ranges = [0, 256] # 계산하고자하는 그레이 레벨(명암)의 범위입니다. 일반적으로 [0, 256]을 사용
    #원래 이미지의 히스토그램을 출력합니다.
    hist_full = create_histogram([img],channels,mask,histSize,ranges)
    
    '''지시사항 3번에 따라 아래 적절한 값을 입력하세요.'''
    # 전체 이미지의 오른쪽 아래 1/4에 대한 히스토그램을 구하세요. (x, y 값 모두 100~250을 지정하면 됩니다.)
    x_range = [100, 250]
    y_range = [100, 250]
    mask, masked_img = create_mask(img, x_range, y_range)

    # 마스크를 포함한 히스토그램과 제외한 히스토그램을 출력합니다.
    #이 줄의 None은 문제가 아닙니다.
    hist_mask = create_histogram([img],channels,mask,histSize,ranges)
    plt.subplot(221), plt.imshow(img, 'gray')
    plt.subplot(222), plt.imshow(masked_img,'gray')
    plt.subplot(223), plt.hist(hist_full,256)
    plt.xlim([0,256])
    plt.subplot(224), plt.hist(hist_mask,256)
    plt.xlim([0,256])
    plt.show()


    # 엘리스 화면에 그래프를 표시합니다.
    plt.savefig('masked_graph.png')
    elice_utils.send_image('masked_graph.png')
    plt.close()
    
    return channels, ranges, x_range, y_range
    
if __name__ == '__main__':
    main()