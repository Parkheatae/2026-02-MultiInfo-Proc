# 2026-10-27 멀티미디어 정보처리 수업

1. 영상 읽기/쓰기 (쓰기는 하지 않음)

   1. cv2.imread()는 cv2.IMREAD_COLOR, cv2.IMREAD_GRAYSCALE 은 8bit unsigned int (byte) 행렬을 반환
   2. cv2.IMREAD_ANYDEPTH나 IMREAD_UNCHANGED의 경우에는 파일 저장 정보에 따라 16비트, 32비트 실수형 자료를 반환
2. 동영상 읽기/쓰기

   1. 동영상은 프레임 단위로 영상 처리 가능
   2. notebook에서는 cvnb 를 사용하는 경우 RUN_GUI=True 하면 연속된 영상을 출력할 수 있다.
   3. cvnb 사용없이 opencv를 직접 사용하는 경우, 동일한 window title에 영상을 겹쳐 출력하므로 동영상 출력을 할 수 있다.
   4. opencv를 직접사용하는 항상 waitKey()와 destroyAllWindows() 또는 destroyWindow()를 같이 사용해야 한다.
3. matplotlib를 이용해서 영상 출력

   1. figure로 그림의 크기 지정
   2. subplot을 이용하면 여러개의 그림 또는 그래프를 그릴 수 있다.
      1. subplot( 행의 크기, 열의 크기, 그림 순서 )
      2. 그림 순서는 그림이 표시될 위치를 가리키기 때문에, 중복을 피해야 한다.
   3. subplot 없이 plot을 반복적으로 호출하는 경우
      1. 동일한 좌표축을 이용해서 여러개의 그래프를 그릴 수 있다.
      2. 이미지와 그래프를 동시에 그리는 경우, 좌표 방향이 달라질 수 있으니 주의
