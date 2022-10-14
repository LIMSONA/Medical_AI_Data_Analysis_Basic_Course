
# 01. Padding, Stride와 Layer size
`Convolutional Layer`는 커널을 이용하여 이미지에서 feature를 추출하는 Layer입니다.

* 이미지는 Convolutional Layer를 통과할 때 padding을 따로 추가하지 않았다면 사이즈가 점점 줄어듭니다. 따라서 이를 방지하고자 이미지의 테두리에 padding을 추가하게 됩니다.
* 그 외에도 Convolutional Layer에서 조절할 수 있는 hyperparameter로는 커널의 개수, 커널의 크기, stride 등이 있습니다. 이들 모두 결과 이미지의 크기와 형태에 영향을 미치기 때문에 이들을 설정했을 때 결과 feature map이 어떻게 변할지 아는 것이 모델 구성의 첫걸음입니다.

# 02. MLP로 이미지 데이터 학습하기
Fully-connected Layer를 쌓아 만든 Multilayer Perceptron(MLP) 모델


# 04. VGG16 모델 구현하기
VGGNet부터는 Layer 개수가 많이 늘어남에 따라 Block 단위로 모델을 구성하게 됩니다. 각 Block은 2개 혹은 3개의 Convolutional Layer와 Max Pooling Layer로 구성되어 있습니다.

# 05. ResNet 구현하기
ResNet에 처음 소개된 Residual Connection은 모델 내의 지름길을 새로 만든다고도 하여 Skip Connection이라고도 불리며, 레이어 개수가 매우 많은 경우에 발생할 수 있는 기울기 소실(Vanishing Gradient) 문제를 해결하고자 등장하였습니다.

![image](https://cdn-api.elice.io/api-attachment/attachment/e5c43920e91946aca57c07b304146057/Residual%20Block.JPG)
`Residual Connection`은 보통 ResNet의 각 Block 단위로 들어가 있습니다. 따라서 일반적으로 Residual Connection을 가지는 부분을 Residual Block이라 하여 Block 단위로 구현한 후에 이들을 연결하는 식으로 모듈화 하여 전체 모델을 구현하게 됩니다.