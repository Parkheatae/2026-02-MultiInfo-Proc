# 2026-10-22 수업

## opencv를 notebook에서 사용하기

- 제공한 cvnb.py를 import
  - cv2.imshow()나 cv2.waitKey() 를 실행할 때, notebook 안의 display에서 결과를 보여주기 위함
- 이벤트 처리 등은 생략

## 직선, 사각형, 원, 타원형을 그려서 이미지에 직접 표시 가능

- cv2.line, cv2.rectangle, cv2.circle, cv2.ellipse 를 이용
- image는 numpy 의 ndarray이기 때문에 (row, col) 형식이고, 이미지는 (x, y)의 형식이기 때문에 이미지 처리할 때 혼동해서는 안됨

## 다음 주에는 imread()와 imwrite()

- 이미지를 읽어서, 저장하는 함수를 공부
