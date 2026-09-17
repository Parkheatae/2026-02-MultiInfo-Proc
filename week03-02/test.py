# 1) utils 모듈 그대로 사용
# import utils
# print(utils.M_PI)

# 2) utils 모듈을 ut 라는 이름으로 사용
# import utils as ut
# print(ut.M_PI)
# print(ut.say())
# print(ut.calc_area(1, 10, 20, 2.0))

# 3) utils의 변수, 함수 등을 모듈명 없이 그래도 사용
from utils import M_PI, say
import utils

print(M_PI)
print(say())
print(utils.calc_area(2, 10, 10))