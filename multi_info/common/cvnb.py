"""cvnb — OpenCV 예제를 VS Code 노트북에서 '원본 코드 그대로' 실행하기 위한 도우미

사용법 (노트북 맨 위 [공통] 셀에서 한 번):
    import cvnb

하는 일
  - cv2.imshow(창이름, 영상)   → 창을 띄우는 대신 노트북 안에 그림으로 모아서 표시
  - cv2.waitKey(...)          → 모아 둔 그림을 한 번에 표시하고 바로 다음 줄로 진행 (키 입력 대기 없음)
  - cv2.namedWindow / createTrackbar / getTrackbarPos / setTrackbarPos / setMouseCallback
                              → 실제 창 없이 값만 저장하는 '가상 창'으로 동작
  - matplotlib 한글 글꼴 설정

진짜 OpenCV 창으로 보고 싶으면:  cvnb.use_gui(True)
  → 창이 뜨면 아무 키나 눌러야 다음 셀로 진행됩니다.
"""
import logging

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ─── matplotlib 한글 글꼴 ───
plt.rcParams["font.family"] = "Malgun Gothic"        # Windows 기본 한글 글꼴 (macOS: AppleGothic)
plt.rcParams["axes.unicode_minus"] = False
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

_orig = {k: getattr(cv2, k) for k in ("imshow", "waitKey", "waitKeyEx", "namedWindow", "createTrackbar",
                                       "getTrackbarPos", "setTrackbarPos", "setMouseCallback",
                                       "destroyAllWindows", "destroyWindow", "moveWindow", "resizeWindow")}
_windows = {}        # 창 이름 → 영상 (같은 이름으로 다시 imshow 하면 교체)
_trackbars = {}      # (트랙바 이름, 창 이름) → [값, 콜백]
COLS = 3             # 한 줄에 표시할 창 개수
WIDTH = 4.2          # 창 하나의 표시 폭 (inch)


def _to_rgb(img):
    """cv2.imshow 와 같은 규칙으로 표시용 영상 만들기 (BGR → RGB, float 은 0~1 → 0~255)"""
    a = np.asarray(img)
    if a.dtype == np.bool_:
        a = a.astype(np.uint8) * 255
    elif a.dtype in (np.float32, np.float64):
        a = np.clip(a * 255, 0, 255).astype(np.uint8)
    elif a.dtype == np.uint16:
        a = (a / 256).astype(np.uint8)
    elif a.dtype != np.uint8:
        a = np.clip(a, 0, 255).astype(np.uint8)
    if a.ndim == 3 and a.shape[2] == 3:
        return cv2.cvtColor(a, cv2.COLOR_BGR2RGB), None
    if a.ndim == 3 and a.shape[2] == 4:
        return cv2.cvtColor(a, cv2.COLOR_BGRA2RGBA), None
    if a.ndim == 3:
        a = a[:, :, 0]
    return a, "gray"


def show_windows(cols=None):
    """모아 둔 '창'들을 한 그림으로 표시하고 비움 (그림 제목 = imshow 의 창 이름)"""
    if not _windows:
        return
    items = list(_windows.items())
    _windows.clear()
    cols = min(cols or COLS, len(items))
    rows = (len(items) + cols - 1) // cols
    ratios = [np.asarray(im).shape[0] / max(1, np.asarray(im).shape[1]) for _, im in items]
    h = WIDTH * min(1.25, max(0.12, max(ratios)))
    fig, axes = plt.subplots(rows, cols, figsize=(WIDTH * cols, (h + 0.45) * rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (name, im) in zip(axes.ravel(), items):
        rgb, cmap = _to_rgb(im)
        ax.imshow(rgb, cmap=cmap, vmin=0, vmax=255, interpolation="nearest")
        ax.set_title(name, fontsize=11)
    plt.tight_layout()
    plt.show()


def _imshow(winname, mat):
    _windows[str(winname)] = np.array(mat, copy=True)


def _waitKey(delay=0):
    show_windows()
    return -1


def _namedWindow(winname, flags=None):
    return None


def _createTrackbar(trackbarName, windowName, value, count, onChange=None):
    _trackbars[(trackbarName, windowName)] = [value, onChange]


def _getTrackbarPos(trackbarName, windowName):
    return _trackbars[(trackbarName, windowName)][0]


def _setTrackbarPos(trackbarName, windowName, pos):
    """트랙바를 움직인 것처럼 값을 바꾸고 콜백 호출 → 노트북에서 '트랙바 조작' 흉내"""
    tb = _trackbars[(trackbarName, windowName)]
    tb[0] = pos
    if tb[1]:
        tb[1](pos)


def _setMouseCallback(windowName, onMouse, param=None):
    _trackbars[("__mouse__", windowName)] = [param, onMouse]


def mouse(windowName, event, x, y, flags=0):
    """마우스 이벤트 흉내: cvnb.mouse("창", cv2.EVENT_LBUTTONDOWN, x, y)"""
    param, cb = _trackbars[("__mouse__", windowName)]
    cb(event, x, y, flags, param)


def _destroy(*args):
    _windows.clear()


def _noop(*args, **kwargs):
    return None


_patched = {"imshow": _imshow, "waitKey": _waitKey, "waitKeyEx": _waitKey, "namedWindow": _namedWindow,
            "moveWindow": _noop, "resizeWindow": _noop,
            "createTrackbar": _createTrackbar, "getTrackbarPos": _getTrackbarPos,
            "setTrackbarPos": _setTrackbarPos, "setMouseCallback": _setMouseCallback,
            "destroyAllWindows": _destroy, "destroyWindow": _destroy}


def use_gui(on=True):
    """True: 진짜 OpenCV 창 사용 (키 입력 대기) / False: 노트북 안에 표시 (기본)"""
    for k, f in (_orig if on else _patched).items():
        setattr(cv2, k, f)


use_gui(False)

# 셀이 waitKey 없이 끝나도 모아 둔 창을 표시
try:
    get_ipython().events.register("post_run_cell", lambda *a: show_windows())   # noqa: F821
except NameError:
    pass
