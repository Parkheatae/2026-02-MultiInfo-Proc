# 2026-11-03 멀티미디어 정보처리

### opencv 기본 배열 연산 후반부

- mask 의 역할과 적용 방법

  - mask 인자를 적용 : 1, 0으로 논리적인 값을 부여
  - mask 인자 없이 적용 : image 처럼 uint8에 대해서 적용하기 때문에 (1이 아니라 255, 비트단위 정의)
- threshold 적용

  - binary/ binary_inv : 정해진 threshold를 초과 또는 미만인 경우에 0, 255 (maxval)로 지정
  - tozero/ tozero_inv : threshold를 초과 또는 미만인 경우에 threshold로 trunc, 또는 0으로 지정
  - trunc : threshold를 초과하는 경우 threshold로 cut (trunc) 적용
- 이미지의 차분 현상

  - subtract : 처음이미지를 기준으로 처음이미지가 값이 클 경우 확인
  - absdiff : 두 이미지의 차이를 확인, 어떤 이미지가 큰지는 확인 불가능
- 통계 함수

  - reduce : 행, 열의 정보를 축소
  - sort / sortIdx : 행, 열에 따라서 정렬 또는 정렬된 후의 인덱스 반환
- gemm

  - 일반화된 행렬 연산을 처리
  - 회전 행렬 적용시 transpose 유의
  - point 열의 경우 tranpose 고려되어야 함!
