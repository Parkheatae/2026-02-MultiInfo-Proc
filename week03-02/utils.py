# 전역변수
M_PI = 3.141592


def calc_area(shape, w, h=None, scale=1.0):
    # 지역변수
    area = None
    if shape == 1:  # 사각형
        area = w * h
        msg = "사각형"
    elif shape == 2:  # 삼각형
        area = w * h / 2
        msg = "삼각형"
    elif shape == 3:  # 원
        area = M_PI * (w/2)**2  # pi * r ** 2
        msg = '원'
    else:  # 지원하지 않음
        print("지원하지 않습니다.")
        msg = 'None'
    area = area * scale if 1<=shape<=3 else area
    return area, msg


def say():
    print("면적을 계산합니다.")
    return True