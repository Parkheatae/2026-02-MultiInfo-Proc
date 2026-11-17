# 2026-11-17 멀티미디어 정보처리 수업

- Sobel 마스크를 이용한 edge 검출
  - sobel은 x, y축의 1차 미분을 계산
  - magnitude 또는 addWeighted를 이용해서 x, y축을 결합
- Laplacian Of Gaussian, Difference Of Gaussian을 통해서 2차 미분의 약점 (잡음에 취약한 edge검출 보완)
  - Gaussian 블러를 적용
  - DoG인 경우 가우시안 차를 적용할 때 cv2.subtract를 이용하는 것이 Underflow를 예방 가능
- Canny Edge 검출
  - 가우시안 블러를 통해서 잡음을 축소
  - Sobel을 이용해 미분 계산
  - non-max suppression을 통해 edge 후보 검출
  - hysteresis를 이용해서 edge 연결 (확정)
    - high threshold를 넘어가면 edge 후보가 되고, low가 되기전까지 edge로 판별
    - low threshold를 넘어가지만, high를 넘어가지 않으면 edge 후보에서 탈락
  - 검출된 edge가 255 pixel 값을 가지므로, mask 형식으로 적용 가능
    - LoG, DoG, Roberts, Prewitt, Laplacian은 mask 형식이 아니라 edge가 강화된 이미지 (영상)과 혼동하지 말것
