# 1. 의료 인공지능 개요
## 01. 의료서비스패러다임의변화

의료환경의 변화로 인해 데이터기반 및 예방 중심 패러다임으로 변화되어 감
* 의료환경의 변화 
1) IoT, 웨어러블 기기, 클라우드
2) 의료 빅데이터·인공지능도입 : EMR, 보건의료데이터, 딥러닝활용도증가
3) 데이터 3법통과 

* 최근 헬스케어 예방구조(4P)
: 치료 중심에서 **예방(Prevention)**으로, 환자가 직접 의료과정에 **참여(Participation)**하고,
개인의 데이터에 따라 분류하고 맞춤화된 s의학적인 결정을내리거나 처방하는 **맞춤(Personalization)**의료, 
질병의 예후를 **예측(Prediction)** 하는 예측중심의 의료로 변화하고있다.  
<br>

## 02. 의료진에게 딥러닝이란
1. 인공지능이 탑재된 진단지원 시스템 등은 환자의 진단을 더 편하고 더 쉽게 도움을 줄 수 있음
2. 인홈(in home) 모니터링 시스템을 통해 환자의 상태를 데이터를 수집하여 진단 및 재활 정확성 향상
3. 인공지능을 통한 시뮬레이션으로 비숙련 의료진들의 숙련도 향상
<br>

## 03. 의료데이터의 특성
1. 복잡한 의료데이터 : EMR, Chart, 진료비 청구 데이터 등
2. 이미지형식의 의료데이터 : MRI, PET, X-ray, CT 등 
    PACS(디지털의료영상저장전송시스템)
3. 연속적 의료데이터 : EKG, V/S
<br>

## 04. 의료데이터 관리의 중요성
<br>

## 05. 인공신경망의 의료데이터를 학습하는 방법
<br>

## 06. 의료인공지능 활용사례
<br>

## 07. 의료인공지능의 장점과 한계
<br>


# 2. 의료영상 이미지를 활용한 DL실습


## 02. CXR 히스토그램
```
matplotlib 라이브러리를 이용해 이미지의 히스토그램을 계산하고 시각화합니다.
이미지 히스토그램은 디지털 이미지에서 색 분포를 그래픽으로 표현하는 역할을 합니다. 
이를 위해 각 색 값(혹은 범주)에 대한 픽셀 수 를 표시합니다. 

X축은 그레이 레벨(대체로 0에서 256까지)을 Y축은 이미지의 해당 그레이 레벨에 해당하는 픽셀의 수를 나타내게 됩니다.

특정 이미지에 대한 히스토그램을 통해 우리는 전체 색 분포를 한 눈에 판단할 수 있습니다.
```

`라이브러리`
* cv2(OpenCV): 인텔에서 개발한 실시간 이미지 처리를 위한 라이브러리
* matplotlib: 시각화를 위한 ㄴ종합적인 라이브러리

`함수`
* cv.calcHist(images, channels, mask, histSize, ranges): 이미지의 히스토그램을 계산하여 주는 함수입니다.
    * images: 히스토그램을 계산하고자 하는 이미지 입니다. 여러 개의 이미지일 경우 리스트(list) 형식으로 입력하세요.
    * channels: 히스토그램을 계산할 채널의 인덱스를 지정합니다. 흑백 이미지의 경우 [0]을, 컬러(빨강, 초록 및 파랑) 이미지에 대한 히스토그램을 계산하려면 [0, 1, 2]를 입력하세요.
    * mask: 계산하고자하는 이미지 영역을 지정합니다. 전체 영역에 대한 계산을 원할 경우 None을 입력하세요.
    * histSize: 히스토그램 계산 시 사용하는 빈(bin)의 수를 의미합니다.
    * ranges: 계산하고자하는 그레이 레벨(명암)의 범위입니다. 일반적으로 [0, 256]을 사용합니다.

* cv.bitwise_and(image1, image2, mask): 두 이미지의 서로 공통으로 겹치는 부분 출력하는 함수입니다. 
    * image1, image2: 사용할 두 개의 이미지를 입력하세요.
    * mask: 계산하고자하는 이미지 영역을 지정합니다. 전체 영역에 대한 계산을 원할 경우 None을 입력하세요.


## 03. 이미지 증강 기법(Data Augmentation)을 이용하여 이미지 수 늘리기
```
데이터 증강은 이러한 문제를 완화하기 위해 기존 데이터를 이용하여 새로운 데이터를 생성하는 기술입니다. 여기에는 기존 데이터를 약간 변형하는 것이 포함됩니다.
```
* 일반적으로 이미지는 0-255의 값을 가진 픽셀로 구성됩니다. 이러한 픽셀 값은 모델 학습 속도 저하와 같은 문제를 발생시킬 수 있습니다. 우리는 픽셀 값을 0-1 범위로 조정하여 표준화함으로써 이러한 문제를 완화할 수 있습니다.
* horizontal flip을 사용할 경우 50%의 확률로 이미지를 뒤집습니다. 우리는 단순한 뒤집기 만으로 이미지 데이터를 두 배로 늘릴 수 있습니다.
* 이외에도 rotation_range를 사용하여 이미지를 회전시키거나 width_shift 혹은 height_shift를 사용하여 이미지를 수평 혹은 수직으로 이동시킬 수 있습니다.

## 04. VGG16 구현하기
```
VGGNet은 ILSVRC 2014년도에 2위를 차지한 모델로, 모델의 깊이에 따른 변화를 비교할 수 있게 만든 이미지 분류 모델입니다. 간단한 구조와 높은 성능을 가지고 있어 많이 사용되는 모델입니다.
VGGNet은 모든 Convolution Layer에 3 x 3 convolution filter를 사용한 것이 특징입니다.

<구조>
VGGNet은 VGG16, VGG19 2개의 버전이 있습니다. 숫자는 모델이 가진 레이어의 개수를 의미합니다. 예를 들어 VGG16은 13개의 합성곱 레이어(convolution layer)와 3개의 완전 연결 레이어(fully connected layer)로 구성되어 있습니다.
```
![image](https://elice-api-cdn.azureedge.net/api-attachment/attachment/a2c34b385fa8478eb6552b893d4b08b8/vgg16.png)
>VGG16의 요약정보를 보면 파라미터가 굉장히 많은 것을 확인할 수 있습니다. 
모델이 커짐에 따라 더 많은 파라미터가 필요하기 때문에 Computation Power가 굉장히 커집니다.

>또한, 모델이 깊어짐에 따라 아래로 갈수록 기울기가 소실되어 학습이 되지 않는 기울기 소실 문제가 발생합니다. 위의 문제로 VGGNet은 16 혹은 19 레이어 모델만을 만들었으며, 후에 기울기 소실 문제를 완화한 152층의 ResNet과 같은 모델이 등장합니다.

<br>  
  
![image](https://cdn-api.elice.io/api-attachment/attachment/cb4a626dcd8540b78847e78683ce06a8/Conv_filter.png)
>3 x 3 Conv filter를 두번 사용하면 (5 x 5)와 같고, 세 번 사용하면 (7 x 7) 과 같아집니다. 그러나 3 x 3을 여러번 사용하게 되면, 연산에 드는 비용이 더 적어지기 때문에 (ex, 3 x 3 x 2 = 18 vs 5 x 5 = 25) 더 높은 성능을 낼 수 있습니다.

## 05. 딥러닝 학습을 위한 데이터셋 분리
```
미지의 데이터에서의 모델 성능을 향상시키기 위해 데이터를 학습, 검증 및 테스트로 나누는 것입니다.

* Train 데이터: 모델이 데이터의 숨겨진 기능/패턴을 학습하도록 하는 데 사용되는 데이터 세트입니다.
* Validation 데이터: 검증 세트는 훈련 중 모델 성능을 검증하는 데 사용되는 훈련 세트와 별개의 데이터 세트입니다. 이 검증 프로세스는 모델의 하이퍼파라미터와 구성을 적절하게 조정하는 데 도움이 되는 정보를 제공합니다.
* Test 데이터: 테스트 세트는 훈련 완료 후 모델을 테스트하는 데 사용되는 별도의 데이터 세트입니다. 간단히 말해서 “ 모델이 얼마나 잘 수행됩니까? “ 라는 질문에 답합니다.
```

## 07. Confusion Matrix를 통한 딥러닝 성능 평가하기

![image](https://2.bp.blogspot.com/-EvSXDotTOwc/XMfeOGZ-CVI/AAAAAAAAEiE/oePFfvhfOQM11dgRn9FkPxlegCXbgOF4QCLcBGAs/s1600/confusionMatrxiUpdated.jpg)

`TP, FP, TN, FN란?`
```
TP(True Positive): 1(정답)인 레이블을 1이라고 예측한 경우
FP(False Positive): 0(오답)인 레이블을 1이라고 예측한 경우
TN(True Negative): 0(오답)인 레이블을 0이라고 예측한 경우
FN(False Negative): 1(정답)인 레이블을 0이라고 예측한 경우
```

`정확도, 정밀도, 재현율, F1 score`
```
* 정확도 (accuracy): 얼마나 1을 1로, 0을 0으로 정확하게 분류해내는지
accuracy= TP+TN / TP+FP+FN+TN 

* 정밀도 (precision): 모델이 1이라고 분류해냈을 때 실제 얼마나 1인지. 즉, 얼마나 믿을만하게 1로 분류해내는지 평가하는 지표
precision=TP / TP+FP​
 
* 재현율 (recall=specificity): 정밀도와 비교되는 척도. 전체 예측 중에서 TP가 얼마나 많은가에 관한 척도.
recall=TP / TP+FN
​
* F1 score: F1 score는 precision과 recall의 균형을 원할 때 사용합니다. 일반적으로 클래스 분포가 고르지 않은 경우 더 나은 측정 방법으로 많이 사용됩니다.
F1= 2 ∗ Precision∗Recall / Precision+Recall​
```

## 08. 분류(Classification) 평가 지표

> 다중 클래스 분류 모델의 성능을 확인하거나 시각화할 때 사용하는 AUC(Area Under the Curve)와 ROC(Receiver Operating Characteristic curve)

`AUC - ROC Curve`
```
> AUC - ROC Curve는 모델의 판단 기준점을 연속적으로 바꾸면서 성능을 측정하였을 때 FPR과 TPR의 변화를 나타낸 것으로, (0,0)과 (1,1)을 잇는 곡선입니다.

> ROC (Receiver Operating Characteristic curve)는 FPR (False positive rate)과 TPR (True Positive Rate)을 각각 x, y축으로 놓은 그래프입니다.

 * TPR (True Positive Rate): 레이블이 1인 케이스에 대해 1로 바르게 예측하는 비율
 * FPR (False positive rate): 레이블이 0인 케이스에 대해 1로 틀리게 예측하는 비율
 * AUC (Area Under the Curve)는 ROC 곡선 아래의 면적을 나타냅니다. AUC 면적의 크기가 클수록 혹은 FPR이 0일 때 TPR이 1에 가까울수록 좋은 모델이라 할 수 있습니다.
```
![image](https://cdn-api.elice.io/api-attachment/attachment/c1e81cbc84184383b30c612cbd94a407/AUC-ROC%20curve.png)

`sklearn.metrics 서브 패키지`
* roc_curve(y_true, y_pred) -> fpr, tpr
: 실제 테스트 데이터의 y값과 모델로 예측하여 얻은 y값 사이의 관계를 파악하기 위해 값을 넣어주면, FPR(False Positive Rate), TPR(True Positive Rate)를 리턴해줍니다.

* auc(fpr, tpr) : fpr, tpr로부터 계산된 결과를 ROC Curve에 그릴 수 있도록 영역을 지정합니다.