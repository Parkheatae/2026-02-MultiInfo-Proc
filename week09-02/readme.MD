# 2026-10-29 수업

- flip, repeat, transpose 학습

  - flip : 0 (상하), 1(좌우), -1 (둘다) 대칭
  - repeat : ny, nx 순으로 정수배로 확장
  - transpose : numpy의 transpose와 달리 차원을 고려하지 않고 이미지 (흑백, 컬러)의 행과 열을 치환
- split, merge

  - split : color image를 gray image로 나눔 (b, g, r) 3채널의 ndarray 형식
  - merge : split의 반대
- add, subtract, multiply, divide는 원소간 연산

  - opencv의 연산 saturate가 포함 (0보다 작은 0, 255보다 크면 255로 고정)
- bitwise 연산

  - and, or, xor, not 연산
  - mask 역할, 이미지끼리 비트와이즈 연산
