# 음성정보처리 Chapter 2 — 기본 표현(Basic Representations) × librosa 실습

### 2026-02 멀티미디어정보처리
#### 한남대학교 정보통신공학과 윤영선교수 
**교재 대응**: Chapter 2 「기본 표현」 전체 + librosa 라이브러리 소개

---

## 오늘의 목표

1. 소리가 컴퓨터 안에서 어떤 숫자로 저장되는지 설명하고 확인한다 (PCM, 샘플링, 양자화)
2. 음성을 20~30 ms 조각으로 자르고 창(window)을 씌우는 이유를 설명한다
3. 스펙트로그램을 그리고, 거기서 F₀·포먼트·마찰음·파열음을 찾아낸다
4. 에너지/ZCR/자기상관으로 유성음·무성음을 구분하고 음높이를 추정한다
5. 멜 스케일과 MFCC가 무엇을 압축한 결과인지 설명하고 추출한다

> **실습 진행 방식**: 셀을 위에서부터 차례대로 `Shift + Enter` 로 실행합니다.
> 각 실습 끝의 **🔍 관찰 포인트**는 옆 사람과 말로 답해 보고, **✏️ 직접 해보기**는 직접 코드를 고쳐 봅니다.

---
# 0교시 (20분)

## 0.1 왜 librosa인가?

| 라이브러리 | 잘하는 일 |
|---|---|
| `soundfile`, `scipy.io.wavfile` | 파일 읽기/쓰기만 |
| `pydub` | 자르기·붙이기 등 편집 |
| **`librosa`** | **분석**: STFT, 멜, MFCC, 피치, 시각화까지 한 번에 |

librosa는 BSD 라이선스 오픈소스이고, numpy / scipy / matplotlib 생태계 위에서 동작합니다.
즉 **librosa가 돌려주는 것은 전부 numpy 배열**입니다. 이것만 기억하면 절반은 이해한 겁니다.


```python
# 처음 한 번만 실행하세요 (이미 설치했다면 건너뛰기)
# !pip install librosa soundfile matplotlib numpy scipy
```


```python
import numpy as np
import scipy.signal as sps
import matplotlib as mpl
import matplotlib.pyplot as plt

import librosa
import librosa.display
from IPython.display import Audio, display

print("librosa 버전:", librosa.__version__)
print("numpy  버전:", np.__version__)

# ── 그래프 한글 폰트 설정 (운영체제에 맞는 것이 자동 선택됩니다) ──────────
# for font in ["Malgun Gothic", "AppleGothic", "NanumGothic", "DejaVu Sans"]:
#    if font in {f.name for f in mpl.font_manager.fontManager.ttflist}:
#        mpl.rcParams["font.family"] = font
#        break
mpl.rcParams["font.family"] = ["Malgun Gothic","DejaVu Sans"]
mpl.rcParams["axes.unicode_minus"] = False   # 마이너스 기호 깨짐 방지
mpl.rcParams["figure.figsize"] = (11, 3.5)
mpl.rcParams["figure.max_open_warning"] = 0
print("사용 폰트:", mpl.rcParams["font.family"])
```

## 0.2 오디오 파일 불러오기

```python
y, sr = librosa.load(path, sr=16000, mono=True)
```

| 변수 | 의미 |
|---|---|
| `y` | **파형**. 1차원 numpy 배열. 각 값은 그 순간의 (상대적) 기압 = 교재의 $x_n$ |
| `sr` | **샘플링 레이트**(Hz). 1초에 숫자를 몇 개 기록했는가 |

교재 2.1의 핵심 문장 — *"음성 신호 = 공기 중 압력 변화 → 마이크 → AD 변환기 → 수열 $x_n$ (PCM)"* —
이 한 줄이 곧 `y, sr = librosa.load(...)` 입니다.

⚠️ `sr=None` 으로 주면 **원본 샘플링 레이트 유지**, 숫자를 주면 그 값으로 **자동 리샘플링**됩니다.
오늘은 음성 표준인 **16 kHz(광대역)** 로 통일합니다.


```python
# ── 오늘 사용할 음성 파일 ─────────────────────────────────────────────
# 방법 A: librosa 내장 예제 (영어 낭독 음성, 인터넷 필요)
# 방법 B: 직접 녹음한 wav 파일 경로를 AUDIO_PATH 에 적기  (권장! 자기 목소리가 제일 재미있습니다)
# 방법 C: 인터넷이 안 되면 맨 아래 [부록 A]의 합성 음성 코드를 먼저 실행

AUDIO_PATH = None          # 예: AUDIO_PATH = "my_voice.wav"
SR = 16000                 # 광대역 음성 표준

if AUDIO_PATH:
    y, sr = librosa.load(AUDIO_PATH, sr=SR, mono=True)
else:
    y, sr = librosa.load(librosa.example("libri1"), sr=SR, mono=True)

y = y[:int(4.0 * sr)]      # 너무 길면 앞 4초만 사용

print("y 타입      :", type(y))
print("y shape     :", y.shape, "  ← 샘플 개수")
print("y dtype     :", y.dtype, " ← 실수(float32), 보통 -1.0 ~ +1.0 범위")
print("sr          :", sr, "Hz")
print("길이        : %.2f 초" % librosa.get_duration(y=y, sr=sr))
print("샘플 수 검산: %d x %.2f = %.0f" % (sr, len(y)/sr, len(y)))
print("최대 진폭   : %.3f" % np.max(np.abs(y)))
print("\ny 의 앞 10개 값:", np.round(y[:10], 4))
```


```python
# 들어보기 (Jupyter 전용 플레이어)
display(Audio(data=y, rate=sr))
```


```python
# 파형 그리기 — 교재 2.1의 첫 그림과 같은 것
fig, ax = plt.subplots()
librosa.display.waveshow(y, sr=sr, ax=ax)
ax.set(title="파형 (Waveform)", xlabel="시간 (s)", ylabel="진폭")
plt.tight_layout(); plt.show()
```

### 🔍 관찰 포인트
- 진폭이 큰 구간과 거의 0인 구간이 번갈아 나옵니다. 큰 곳은 **모음**, 0에 가까운 곳은 **휴지(pause)** 또는 **파열음 직전의 정지 구간**입니다.
- 파형은 **영평균(zero-mean)** 입니다. 0은 "소리 없음"이 아니라 **주변 기압 수준**을 뜻합니다.

### ✏️ 직접 해보기
아래 셀에서 `start`, `dur` 을 바꿔 0.05초(50 ms)만 확대해 보세요. 파형이 **주기적으로 반복**되는 모양이 보이면 그 구간은 유성음입니다.


```python
start, dur = 1.0, 0.05          # ← 값을 바꿔가며 관찰해 보세요
seg = y[int(start*sr):int((start+dur)*sr)]

fig, ax = plt.subplots(figsize=(11, 3))
ax.plot(np.arange(len(seg))/sr*1000, seg)
ax.set(title=f"{start}s 부터 {dur*1000:.0f} ms 확대", xlabel="시간 (ms)", ylabel="진폭")
ax.grid(alpha=.3)
plt.tight_layout(); plt.show()
```

---
# 1교시 (40분) — 파형 · 샘플링 레이트 · 양자화  〈교재 2.1〉

## 1.1 이론 5분 요약

디지털 음성은 **두 번 잘라서** 만듭니다.

| 축 | 자르는 행위 | 결정하는 값 | 잘못 고르면 |
|---|---|---|---|
| 시간축 | **샘플링(sampling)** | 샘플링 레이트 $F_s$ | 고주파가 사라지거나 엉뚱한 주파수로 접힘(앨리어싱) |
| 진폭축 | **양자화(quantization)** | 비트 깊이 | 지직거리는 양자화 잡음 |

**나이퀴스트 주파수** = $F_s / 2$. 표현 가능한 최대 주파수입니다.
- $F_s = 8000$ Hz → 0~4000 Hz만 표현 가능
- 포먼트는 300~3500 Hz → 최소 7~8 kHz 필요 → **전화망이 8 kHz인 이유**

| 대역폭 | $F_s$ | 주파수 범위 |
|---|---|---|
| 협대역 Narrowband | 8 kHz | 0~3.3 kHz |
| 광대역 Wideband | 16 kHz | 0~7 kHz |
| 초광대역 | 32 kHz | 0~16 kHz |
| 풀밴드 (CD) | 44.1/48 kHz | 0~22 kHz |

## 실습 1-1. 샘플링 레이트를 낮추면 소리가 어떻게 되나

`librosa.resample()` 로 같은 음성을 여러 샘플링 레이트로 바꿔 **직접 들어 봅니다**.


```python
for target in [16000, 8000, 4000, 2000]:
    y_rs = librosa.resample(y, orig_sr=sr, target_sr=target)
    print(f"Fs = {target:5d} Hz   나이퀴스트 = {target//2:5d} Hz   샘플 수 = {len(y_rs):6d}")
    display(Audio(data=y_rs, rate=target))
```

### 🔍 관찰 포인트
- 8 kHz까지는 말은 알아들을 수 있지만 /s/, /f/ 같은 **마찰음이 먹먹**해집니다 (교재: 4 kHz 이상 에너지 손실).
- 4 kHz 이하로 가면 **웅얼거림**이 되고, 2 kHz에서는 포먼트 F₂가 잘려 모음 구분이 무너집니다.
- 실제 AD 변환기에는 나이퀴스트 이상을 미리 잘라내는 **저역통과(anti-aliasing) 필터**가 반드시 들어갑니다. `librosa.resample` 도 내부에서 같은 일을 합니다.

## 실습 1-2. 진폭 양자화: 비트 수를 줄여 보기

교재의 **선형 양자화**: $\hat{x} = \Delta q \cdot \mathrm{round}(x / \Delta q)$

16비트면 진폭 레벨이 $2^{16}=65{,}536$ 개, 4비트면 겨우 16개입니다.
얼마나 망가지는지 **SNR(dB)** 로 재 봅시다.

$$\text{SNR} = 10\log_{10}\frac{\sum x_n^2}{\sum (x_n-\hat{x}_n)^2}$$


```python
# 선형 양자화: x를 2^bits 단계로 반올림 (교재 2.1.4)
def linear_quantize(x, bits):
    levels = 2 ** (bits - 1)          # 부호 있는 정수 범위의 절반
    dq = 1.0 / levels                 # 양자화 스텝 크기 Δq
    return np.clip(np.round(x / dq) * dq, -1.0, 1.0)


# 원본 대비 양자화 오차의 신호대잡음비(dB)
def snr_db(x, x_hat):
    noise = x - x_hat
    return 10 * np.log10(np.sum(x**2) / (np.sum(noise**2) + 1e-20))


for bits in [16, 8, 6, 4, 3, 2]:
    yq = linear_quantize(y, bits)
    print(f"{bits:2d} bit  →  레벨 {2**bits:6d}개,  SNR = {snr_db(y, yq):6.2f} dB")
    display(Audio(data=yq, rate=sr))
```


```python
# 양자화 계단을 눈으로 확인 (교재 2.1.4 그림과 같은 형태)
seg = y[int(1.0*sr):int(1.0*sr)+200]
t_ms = np.arange(len(seg)) / sr * 1000

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(t_ms, seg, label="원본 x", lw=1.5)
ax.step(t_ms, linear_quantize(seg, 4), where="mid", label="4비트 양자화 x̂", lw=1.5)
ax.set(title="선형 양자화 — 계단 모양이 곧 양자화 오차", xlabel="시간 (ms)", ylabel="진폭")
ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.show()
```

### 🔍 관찰 포인트
- 비트가 1개 줄 때마다 SNR이 **약 6 dB씩** 떨어집니다. (1 bit ≈ 6 dB, 외워두면 유용)
- 작은 소리 구간이 더 심하게 망가집니다. **선형 양자화의 약점**입니다.

## 실습 1-3. 로그 양자화 (뮤-법칙, μ-law)

교재 2.1.4: 작은 신호와 큰 신호 모두 **균일한 상대 정확도**를 주려면 로그 양자화가 좋다.
$x=0$에서 $\log$ 가 $-\infty$ 로 발산하는 문제는 뮤-법칙이 해결합니다.

$$F(x) = \mathrm{sign}(x)\,\frac{\log(1+\mu|x|)}{\log(1+\mu)},\qquad \mu=255$$

librosa에 그대로 들어 있습니다: `librosa.mu_compress` / `librosa.mu_expand`


```python
# 8비트 선형 vs 8비트 뮤-법칙 비교
y_lin8 = linear_quantize(y, 8)

y_mu   = librosa.mu_compress(y, mu=255, quantize=True)   # 8비트 정수로 압축
y_mu8  = librosa.mu_expand(y_mu,  mu=255, quantize=True) # 다시 복원

print("── 신호 전체 기준 ──")
print("  8비트 선형    SNR = %6.2f dB" % snr_db(y, y_lin8))
print("  8비트 뮤-법칙 SNR = %6.2f dB" % snr_db(y, y_mu8))

# 진폭이 작은 구간(조용한 부분)만 따로 떼어 비교해 봅시다
amp = np.abs(y)
quiet = (amp > 0) & (amp < np.percentile(amp[amp > 0], 30))
print("\n── 조용한 구간(하위 30%)만 기준 ──")
print("  8비트 선형    SNR = %6.2f dB" % snr_db(y[quiet], y_lin8[quiet]))
print("  8비트 뮤-법칙 SNR = %6.2f dB  ← 뮤-법칙의 진짜 이점" % snr_db(y[quiet], y_mu8[quiet]))
print("\n뮤-법칙 압축 결과의 값 범위:", y_mu.min(), "~", y_mu.max(), "(8비트 정수)")

display(Audio(data=y_lin8, rate=sr))
display(Audio(data=y_mu8,  rate=sr))
```


```python
# 뮤-법칙의 압축 곡선 자체를 그려 보기
x_axis = np.linspace(-1, 1, 1000)
mu_curve = np.sign(x_axis) * np.log1p(255*np.abs(x_axis)) / np.log1p(255)

fig, ax = plt.subplots(figsize=(6, 5))
ax.plot(x_axis, x_axis,   "--", label="선형 (y = x)")
ax.plot(x_axis, mu_curve,       label="뮤-법칙 F(x)")
ax.set(title="뮤-법칙(μ-law) 압축 곡선", xlabel="입력 x", ylabel="출력 F(x)")
ax.legend(); ax.grid(alpha=.3); ax.set_aspect("equal")
plt.tight_layout(); plt.show()
```

### 🔍 관찰 포인트
- **신호 전체 SNR은 선형 양자화가 더 높게 나올 수도 있습니다.** 그런데도 뮤-법칙을 쓰는 이유는 **조용한 구간의 SNR**이 훨씬 좋기 때문입니다. 사람 귀는 큰 소리 속 잡음은 잘 못 듣고, 조용할 때의 잡음은 잘 듣습니다. 교재의 *"SNR로 측정한 품질 ≠ 지각적 품질"* 이 바로 이 이야기입니다.
- 곡선이 **0 근처에서 가파릅니다** → 작은 신호에 더 많은 레벨을 배정 → 조용한 부분이 살아납니다.
- 이처럼 압축(compress)+복원(expand)을 묶은 기법을 **컴팬딩(companding)** 이라 부릅니다. 전화망 G.711이 바로 이것입니다.

### 💡 한 장 정리 (교재 2.1.6~2.1.8)
> PCM(선형) → APCM(스텝 크기를 시간에 따라 적응) → DPCM(예측 오차만 양자화) → ADPCM(둘 다 적응)
> **신호에 대해 아는 게 많을수록 같은 비트로 더 좋은 품질**을 얻습니다. 6교시의 선형예측(LPC)이 그 "아는 것"의 정체입니다.

### ✏️ 직접 해보기
`linear_quantize` 를 4비트로, `mu_compress`를 `quantize=True`로 각각 적용해 SNR을 비교해 보세요. 어느 쪽이 더 유리한가요?

---
# 2교시 (40분) — 윈도잉 · 프레이밍 · Overlap-Add  〈교재 2.2 / 2.4〉

## 2.1 이론 5분 요약

문제: 음성은 **고도로 시변(time-variant)** 인데, 푸리에 변환 같은 분석 도구는 **정상(stationary) 신호**를 가정합니다.
→ 문장 전체를 한 번에 분석하면 **모든 음소의 평균**이 나와서 쓸모가 없습니다.

해결: 신호를 **20~30 ms 짧은 조각(프레임)** 으로 자릅니다. 그 정도면 음소 하나가 거의 변하지 않습니다.

그런데 그냥 자르면 **경계에서 불연속**이 생깁니다 → 스펙트럼이 번져 보입니다(누설, leakage).
그래서 경계에서 0으로 부드럽게 수렴하는 **윈도우 함수**를 곱합니다.

| 윈도우 | 식 | 쓰는 곳 |
|---|---|---|
| 사각(rectangular) | 1 | 그냥 자르기 (누설 최대) |
| Hann | $w_n=\sin^2(\pi n/L)$ | **분석**에 가장 널리 쓰임 |
| Hamming | $0.54-0.46\cos(\cdot)$ | 분석 (MFCC 관례) |
| 반사인(half-sine) | $\sin(\pi n/L)$ | **처리(OLA)** — Princen-Bradley 만족 |

용어 정리 (librosa 인자 이름과 1:1 대응):

| 교재 용어 | librosa 인자 | 오늘 쓸 값 (16 kHz) |
|---|---|---|
| 윈도우 길이 $L$ | `win_length` / `frame_length` | 400 샘플 = **25 ms** |
| 스텝(hop) | `hop_length` | 160 샘플 = **10 ms** |
| DFT 길이 $N$ | `n_fft` | 512 (2의 거듭제곱, 영 확장 포함) |


```python
# 이 노트북 전체에서 쓸 분석 파라미터 (교재 2.4.3 권장값)
FRAME_LEN = 400    # 25 ms  @16 kHz
HOP_LEN   = 160    # 10 ms  @16 kHz  → 오버랩 60%
N_FFT     = 512    # 400을 담을 수 있는 2의 거듭제곱 (나머지는 0으로 채움 = 영 확장)

print(f"윈도우 길이 : {FRAME_LEN} 샘플 = {FRAME_LEN/sr*1000:.1f} ms")
print(f"스텝(hop)   : {HOP_LEN} 샘플 = {HOP_LEN/sr*1000:.1f} ms")
print(f"오버랩      : {(1-HOP_LEN/FRAME_LEN)*100:.0f} %")
print(f"주파수 해상도: {sr/N_FFT:.1f} Hz/bin,  bin 개수 = {N_FFT//2+1}")
```

## 실습 2-1. 윈도우 함수 모양과 그 스펙트럼

윈도우를 곱한다는 것은 **시간 영역의 곱셈** = **주파수 영역의 컨볼루션**(교재 2.2.1).
즉 진짜 스펙트럼이 윈도우의 스펙트럼으로 **번집니다**. 어떻게 번지는지 봅시다.


```python
L = FRAME_LEN
windows = {
    "사각 (rectangular)": np.ones(L),
    "Hann":               sps.get_window("hann",    L, fftbins=True),
    "Hamming":            sps.get_window("hamming", L, fftbins=True),
    "반사인 (half-sine)": np.sin(np.pi * (np.arange(L) + 0.5) / L),
}

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for name, w in windows.items():
    axes[0].plot(np.arange(L)/sr*1000, w, label=name)
    W = np.abs(np.fft.rfft(w, n=4096))
    W_db = 20*np.log10(W/W.max() + 1e-12)
    axes[1].plot(np.fft.rfftfreq(4096, 1/sr), W_db, label=name)

axes[0].set(title="① 윈도우 함수 모양", xlabel="시간 (ms)", ylabel="w[n]")
axes[1].set(title="② 윈도우의 스펙트럼 (번짐 정도)", xlabel="주파수 (Hz)",
            ylabel="크기 (dB)", xlim=(0, 600), ylim=(-100, 5))
for a in axes: a.legend(fontsize=9); a.grid(alpha=.3)
plt.tight_layout(); plt.show()
```

### 🔍 관찰 포인트
- **사각창**은 가운데 봉우리는 제일 좁지만 옆의 **사이드로브가 -13 dB로 매우 큽니다** → 멀리까지 번짐.
- **Hann/Hamming**은 봉우리는 두 배 넓지만 사이드로브가 훨씬 작습니다 → 스펙트럼이 깨끗합니다.
- 그래서 교재가 말하는 *"주요 최적화 기준: 스펙트럼 왜곡 최소화"* 의 답이 Hann 창입니다.

## 실습 2-2. 프레이밍: 신호를 조각내기

`librosa.util.frame(y, frame_length, hop_length)` → shape **(frame_length, 프레임 개수)** 의 2차원 배열


```python
frames = librosa.util.frame(y, frame_length=FRAME_LEN, hop_length=HOP_LEN)
print("frames.shape =", frames.shape, " → (윈도우 길이, 프레임 개수)")

# 교재 2.2.1의 길이 공식과 대조해 봅시다
n_frames = 1 + (len(y) - FRAME_LEN) // HOP_LEN
print("공식으로 계산한 프레임 개수:", n_frames)

w = sps.get_window("hann", FRAME_LEN, fftbins=True)
frames_win = frames * w[:, np.newaxis]     # 각 열(프레임)에 윈도우를 곱함 (브로드캐스팅)

# 이웃한 세 프레임이 어떻게 겹치는지 그림으로
idx = 120
fig, ax = plt.subplots(figsize=(11, 4))
for k, color in zip([idx, idx+1, idx+2], ["C0", "C1", "C2"]):
    t0 = k * HOP_LEN / sr * 1000
    t  = t0 + np.arange(FRAME_LEN)/sr*1000
    ax.plot(t, frames_win[:, k] + 0, color=color, alpha=.8, label=f"프레임 {k}")
    ax.plot(t, w * np.max(np.abs(frames_win[:, k])), color=color, ls="--", alpha=.4)
ax.set(title="이웃 프레임은 서로 겹친다 (hop 10 ms, 윈도우 25 ms)",
       xlabel="시간 (ms)", ylabel="진폭")
ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.show()
```

## 실습 2-3. 같은 프레임, 다른 창 → 스펙트럼이 달라진다

유성음 한 프레임을 골라 사각창과 Hann창으로 각각 스펙트럼을 그려 비교합니다.


```python
frame_idx = 150                                  # ← 유성음 구간이 나올 때까지 바꿔 보세요
x_frame = frames[:, frame_idx]

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(np.arange(FRAME_LEN)/sr*1000, x_frame, label="원본 프레임")
axes[0].plot(np.arange(FRAME_LEN)/sr*1000, x_frame*w, label="Hann 창을 곱한 뒤")
axes[0].set(title=f"프레임 #{frame_idx} (25 ms)", xlabel="시간 (ms)"); axes[0].legend(); axes[0].grid(alpha=.3)

freqs = np.fft.rfftfreq(N_FFT, 1/sr)
for label, xx in [("사각창", x_frame), ("Hann창", x_frame*w)]:
    X = np.abs(np.fft.rfft(xx, n=N_FFT))
    axes[1].plot(freqs, 20*np.log10(X + 1e-10), label=label, alpha=.85)
axes[1].set(title="로그 크기 스펙트럼 20·log₁₀|X_k|", xlabel="주파수 (Hz)", ylabel="dB")
axes[1].legend(); axes[1].grid(alpha=.3)
plt.tight_layout(); plt.show()
```

### 🔍 관찰 포인트
- 유성음 프레임이라면 **일정 간격의 봉우리(빗살, comb)** 가 보입니다. 그 간격이 바로 **기본주파수 F₀** 입니다.
- 사각창 쪽은 골짜기가 메워져 빗살이 흐릿합니다. Hann창 쪽이 훨씬 또렷합니다.

## 실습 2-4. Overlap-Add: 잘랐다가 다시 붙이면 원본이 되는가?

교재 2.2.1의 **완전 재구성(Perfect Reconstruction)** 과
**Princen-Bradley 조건** $w_n^2 + w_{n+L/2}^2 = 1$ 을 코드로 확인합니다.


```python
# 반사인 창은 50% 오버랩에서 제곱합이 1이 된다 → 완전 재구성
L2 = 512
w_hs = np.sin(np.pi * (np.arange(L2) + 0.5) / L2)     # 반사인 창
check = w_hs[:L2//2]**2 + w_hs[L2//2:]**2             # w²_n + w²_{n+L/2}

print("Princen-Bradley 조건 w²_n + w²_{n+L/2} = 1 인가?")
print("  최소값 = %.6f,  최대값 = %.6f" % (check.min(), check.max()))

fig, ax = plt.subplots(figsize=(9, 3.5))
ax.plot(w_hs**2, label="w²  (윈도우 1)")
ax.plot(np.roll(w_hs**2, L2//2), label="w²  (윈도우 2, L/2 이동)")
ax.plot(w_hs**2 + np.roll(w_hs**2, L2//2), "k", lw=2, label="두 개의 합 = 1")
ax.set(title="50% 오버랩에서 창의 제곱합이 상수 1", xlabel="샘플", ylim=(0, 1.3))
ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.show()
```


```python
# librosa로 STFT → 아무 처리도 안 하고 → ISTFT (내부가 바로 Overlap-Add 입니다)
D      = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LEN, win_length=FRAME_LEN, window="hann")
y_rec  = librosa.istft(D, n_fft=N_FFT, hop_length=HOP_LEN, win_length=FRAME_LEN, length=len(y))

err = y - y_rec
print("최대 오차      : %.3e" % np.max(np.abs(err)))
print("오차 에너지 비 : %.2f dB" % (10*np.log10(np.sum(err**2)/np.sum(y**2) + 1e-30)))

fig, axes = plt.subplots(2, 1, figsize=(11, 5), sharex=True)
axes[0].plot(np.arange(len(y))/sr, y,     label="원본",     lw=.8)
axes[0].plot(np.arange(len(y))/sr, y_rec, label="재구성",   lw=.8, alpha=.7)
axes[0].legend(); axes[0].set(title="원본 vs Overlap-Add 재구성")
axes[1].plot(np.arange(len(y))/sr, err, color="C3", lw=.8)
axes[1].set(title="차이(원본 - 재구성)", xlabel="시간 (s)")
for a in axes: a.grid(alpha=.3)
plt.tight_layout(); plt.show()
```

### 🔍 관찰 포인트
- 오차가 $10^{-7}$ 수준, 즉 **수치 오차를 빼면 완전히 같습니다**.
- 단, 교재대로 **맨 앞과 맨 뒤 경계**에서만 오차가 보입니다. 실무에서는 허용 가능한 수준입니다.
- 이것이 잡음 제거·음성 변환 등 **모든 처리 알고리즘의 뼈대**입니다: 자르고 → 고치고 → 다시 더한다.

### ✏️ 직접 해보기
`window="hann"` 을 `"boxcar"`(사각창)로 바꾸고 다시 실행해 보세요. 재구성 오차가 어떻게 변하나요?

---
# 3교시 (50분) — STFT와 스펙트로그램  〈교재 2.3 / 2.4.5〉

## 3.1 이론 5분 요약

**DFT**: 길이 $N$ 신호 $x_n$ → 복소수 $X_k$ ($N$개 계수).
복소수는 그대로 볼 수 없으므로 세 가지로 시각화합니다.

| 표현 | 식 | 특징 |
|---|---|---|
| 크기 스펙트럼 | $\lvert X_k\rvert$ | 값의 범위가 너무 넓어 해석이 어려움 |
| 전력 스펙트럼 | $\lvert X_k\rvert^2$ | 에너지 개념 |
| **로그(dB) 스펙트럼** | $20\log_{10}\lvert X_k\rvert$ | **인간의 크기 지각과 비슷** → 실무 표준 |

**STFT**: 프레임마다 DFT를 하는 것.

$$X(h,k)=\sum_{n=0}^{N-1} x_{n+h}\,w_n\,e^{-i2\pi kn/N}$$

- $h$: 시간 인덱스(윈도우 위치), $k$: 주파수 인덱스
- 출력은 **복소수 행렬** → `20log₁₀|X(h,k)|` 를 히트맵으로 그린 것이 **스펙트로그램**

librosa에서는 딱 두 줄입니다.

```python
D    = librosa.stft(y, n_fft=512, hop_length=160, win_length=400)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
```


```python
D = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LEN, win_length=FRAME_LEN, window="hann")

print("D.shape  =", D.shape, " → (주파수 bin 수, 프레임 수)")
print("D.dtype  =", D.dtype, " ← 복소수!")
print("주파수 bin 수 검산: n_fft//2 + 1 =", N_FFT//2 + 1)
print("각 bin의 폭      : %.1f Hz" % (sr / N_FFT))
print("프레임 간 시간 간격: %.1f ms" % (HOP_LEN / sr * 1000))
```

## 실습 3-1. 한 프레임의 스펙트럼 — 세 가지 표현 비교


```python
frame_idx = 150
X = D[:, frame_idx]
freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)

fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
axes[0].plot(freqs, np.abs(X));            axes[0].set(title="크기 |X_k|")
axes[1].plot(freqs, np.abs(X)**2);         axes[1].set(title="전력 |X_k|²")
axes[2].plot(freqs, 20*np.log10(np.abs(X)+1e-10)); axes[2].set(title="로그 20·log₁₀|X_k|  (dB)")
for a in axes:
    a.set_xlabel("주파수 (Hz)"); a.grid(alpha=.3)
plt.tight_layout(); plt.show()
```

### 🔍 관찰 포인트
- 크기/전력 표현은 저주파의 큰 값 때문에 **고주파가 바닥에 깔려 안 보입니다**.
- dB로 바꾸면 전 대역의 구조가 한눈에 들어옵니다. 교재: *"dB 스케일 ≈ 인간의 크기 지각 스케일"*.

## 실습 3-2. 스펙트로그램 그리기


```python
# 앞으로 계속 쓸 간단한 헬퍼
def show_spec(D_complex, sr, hop_length, title, y_axis="hz", ax=None, fmax=None):
    S_db = librosa.amplitude_to_db(np.abs(D_complex), ref=np.max)
    created = ax is None
    if created:
        fig, ax = plt.subplots(figsize=(11, 4))
    img = librosa.display.specshow(S_db, sr=sr, hop_length=hop_length,
                                   x_axis="time", y_axis=y_axis, ax=ax, cmap="magma")
    if fmax:
        ax.set_ylim(0, fmax)
    ax.set(title=title)
    if created:
        ax.figure.colorbar(img, ax=ax, format="%+2.0f dB")
        plt.tight_layout(); plt.show()
    return img


show_spec(D, sr, HOP_LEN, "스펙트로그램 (윈도우 25 ms, hop 10 ms)")
show_spec(D, sr, HOP_LEN, "같은 스펙트로그램을 0~4 kHz로 확대 — 포먼트가 보인다", fmax=4000)
```

### 🔍 스펙트로그램에서 찾아야 할 4가지 (교재 2.3.1)

| 무엇 | 어떻게 보이나 | 정체 |
|---|---|---|
| **F₀ (기본주파수)** | 저주파의 **수평 빗살 무늬** | 성대 진동. 간격 = F₀, 그 정수배가 고조파 |
| **포먼트 (F₁,F₂,F₃)** | 가로로 이어지는 **밝은 띠** | 성도의 공명. 모음을 결정 |
| **마찰음 /s/ /f/ /h/** | **고주파의 밝은 뭉게구름** | 성대 진동 없는 잡음 |
| **파열음 /p/ /t/ /k/** | **어두운 정지 구간 + 세로줄** | 막았다가 터뜨리는 소리 |

지금 화면에서 네 가지를 각각 하나씩 손가락으로 찾아보세요.

## 실습 3-3. 시간 해상도 vs 주파수 해상도 — 가장 중요한 트레이드오프

교재 2.4.7: *"더 긴 윈도우 → 스펙트럼 해상도 향상, 시간 해상도 감소"*.
윈도우 길이만 바꿔가며 같은 음성을 세 번 그려 봅니다.


```python
configs = [
    (128,  32, "짧은 창 8 ms — 광대역(wideband): 시간 ↑, 주파수 ↓"),
    (400, 160, "중간 창 25 ms — 표준"),
    (1024, 160, "긴 창 64 ms — 협대역(narrowband): 주파수 ↑, 시간 ↓"),
]

fig, axes = plt.subplots(3, 1, figsize=(11, 10))
for (wl, hl, title), ax in zip(configs, axes):
    nfft = int(2 ** np.ceil(np.log2(wl)))
    Dx = librosa.stft(y, n_fft=nfft, hop_length=hl, win_length=wl, window="hann")
    show_spec(Dx, sr, hl, title, ax=ax, fmax=4000)
plt.tight_layout(); plt.show()
```

### 🔍 관찰 포인트 (시험에 반드시 나옵니다)
- **짧은 창**: 세로줄(파열음, 성대 펄스 하나하나)이 선명. 대신 빗살(F₀)이 뭉개짐 → **포먼트 관찰에 유리**
- **긴 창**: 수평 빗살이 또렷 → **F₀·고조파 관찰에 유리**. 대신 파열음이 번짐
- 둘 다 좋게 할 수는 없습니다. 교재의 *"20~30 ms 최적 균형점"* 이 나온 이유입니다.

## 실습 3-4. 유성음 vs 무성음 한 프레임씩 비교

교재 심화: 유성음은 준주기적·빗살 구조, 무성음은 잡음성·고주파 집중.
**에너지와 영교차율을 이용해 자동으로** 유성/무성 구간을 하나씩 골라 봅니다.


```python
rms = librosa.feature.rms(y=y, frame_length=FRAME_LEN, hop_length=HOP_LEN)[0]
zcr = librosa.feature.zero_crossing_rate(y, frame_length=FRAME_LEN, hop_length=HOP_LEN)[0]

loud = rms > np.percentile(rms, 60)                       # 소리가 충분히 큰 프레임만 후보
voiced_idx   = int(np.argmax(np.where(loud, -zcr, -9e9))) # 큰데 ZCR이 낮다 → 유성음
unvoiced_idx = int(np.argmax(np.where(loud,  zcr, -9e9))) # 큰데 ZCR이 높다 → 무성음
print(f"유성음 후보 프레임 #{voiced_idx}  (t={voiced_idx*HOP_LEN/sr:.2f}s, ZCR={zcr[voiced_idx]:.3f})")
print(f"무성음 후보 프레임 #{unvoiced_idx} (t={unvoiced_idx*HOP_LEN/sr:.2f}s, ZCR={zcr[unvoiced_idx]:.3f})")

fig, axes = plt.subplots(2, 2, figsize=(13, 7))
for row, (name, idx) in enumerate([("유성음 (Voiced)", voiced_idx),
                                   ("무성음 (Unvoiced)", unvoiced_idx)]):
    seg = y[idx*HOP_LEN: idx*HOP_LEN + FRAME_LEN]
    axes[row, 0].plot(np.arange(len(seg))/sr*1000, seg)
    axes[row, 0].set(title=f"{name} — 파형", xlabel="시간 (ms)")
    spec = 20*np.log10(np.abs(np.fft.rfft(seg*sps.get_window('hann', len(seg)), n=1024)) + 1e-10)
    axes[row, 1].plot(np.fft.rfftfreq(1024, 1/sr), spec)
    axes[row, 1].set(title=f"{name} — 로그 스펙트럼", xlabel="주파수 (Hz)", ylabel="dB")
for a in axes.ravel(): a.grid(alpha=.3)
plt.tight_layout(); plt.show()
```

### 🔍 관찰 포인트
- 유성음: 파형이 **반복**되고, 스펙트럼에 **빗살**이 있으며 에너지가 저주파에 몰립니다.
- 무성음: 파형이 **불규칙**하고, 스펙트럼이 평평하며 고주파에 에너지가 있습니다.

### ✏️ 직접 해보기
두 프레임을 각각 `Audio(seg, rate=sr)` 로 들어보세요 (25 ms라 아주 짧습니다. `np.tile(seg, 50)` 으로 반복하면 잘 들립니다).


```python
seg_v = y[voiced_idx*HOP_LEN: voiced_idx*HOP_LEN+FRAME_LEN]
seg_u = y[unvoiced_idx*HOP_LEN: unvoiced_idx*HOP_LEN+FRAME_LEN]
print("유성음 프레임 50번 반복");   display(Audio(np.tile(seg_v, 50), rate=sr))
print("무성음 프레임 50번 반복");   display(Audio(np.tile(seg_u, 50), rate=sr))
```

---
# 4교시 (25분) — 에너지 · 데시벨 · ZCR · 자기상관  〈교재 2.6 / 2.7〉

## 4.1 이론 5분 요약

- **신호 에너지** = 분산 $\mathrm{Var}(x)=E[(x-\mu)^2]$. 진동 신호라 **순간값은 의미가 없고 윈도우 평균**을 씁니다 → RMS
- **데시벨** $10\log_{10}\sigma^2$ (전력 기준) = $20\log_{10}\sigma$ (크기 기준)
- **dBov**: 클리핑 한계 대비 dB. 항상 음수이며 **입력 음성은 보통 -26 dBov로 정규화**합니다
- **ZCR(영교차율)**: 신호가 0을 가로지르는 횟수 → 유성음 낮음, 무성음 높음
- **자기상관** $r_k=E[x_n x_{n-k}]$: 지연 $k$만큼 민 자기 자신과 얼마나 닮았는가 → **피치(F₀) 추정의 고전 도구**

## 실습 4-1. RMS 에너지와 dB, 그리고 간단한 VAD(음성 구간 검출)


```python
rms_db = 20*np.log10(rms + 1e-10)
t_frames = librosa.times_like(rms, sr=sr, hop_length=HOP_LEN)

# 간단한 VAD: 최대 에너지보다 25 dB 이상 낮으면 '비음성'으로 판정
threshold_db = rms_db.max() - 25
is_speech = rms_db > threshold_db

fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
librosa.display.waveshow(y, sr=sr, ax=axes[0], alpha=.7)
axes[0].fill_between(t_frames, -1, 1, where=is_speech, color="C2", alpha=.2, label="음성 구간")
axes[0].set(title="파형 + VAD 결과", ylim=(-1, 1)); axes[0].legend()

axes[1].plot(t_frames, rms_db, label="RMS 에너지 (dB)")
axes[1].axhline(threshold_db, color="C3", ls="--", label=f"임계값 {threshold_db:.1f} dB")
axes[1].set(title="단시간 에너지", xlabel="시간 (s)", ylabel="dB"); axes[1].legend(); axes[1].grid(alpha=.3)
plt.tight_layout(); plt.show()

print("음성으로 판정된 비율: %.1f %%" % (100*is_speech.mean()))
```


```python
# dBov: 클리핑 한계(±1.0) 대비 에너지. 교재 2.6.2
def dbov(x):
    p  = np.mean(x**2)          # 신호 전력
    p0 = 1.0 ** 2               # 클리핑 한계(최대 진폭)의 전력 = 0 dBov 기준
    return 10*np.log10(p/p0 + 1e-20)


sine = np.sin(2*np.pi*440*np.arange(sr)/sr)     # 최대 진폭 정현파
print("최대 진폭 정현파 : %.2f dBov  ← 교재의 -3.01 dBov 와 일치" % dbov(sine))
print("현재 신호        : %.2f dBov" % dbov(y))

# 실무 표준인 -26 dBov로 정규화하기
target = -26.0
y_norm = y * 10 ** ((target - dbov(y)) / 20)
print("정규화 후        : %.2f dBov,  최대 진폭 = %.3f" % (dbov(y_norm), np.max(np.abs(y_norm))))
print("→ 이렇게 여유를 두면 이후 처리에서 진폭이 커져도 클리핑되지 않습니다.")
```

## 실습 4-2. 자기상관으로 F₀ 직접 구해 보기

유성음 프레임의 자기상관을 구하면 **한 피치 주기만큼 밀었을 때 가장 크게 닮습니다**.
그 지연(lag) $k$ 를 찾으면 $F_0 = F_s / k$.


```python
frame_v = seg_v * sps.get_window("hann", FRAME_LEN)
ac = librosa.autocorrelate(frame_v)
ac = ac / (ac[0] + 1e-12)                   # r_k / r_0 = 자기상관 c_k (교재 2.7)

fmin, fmax = 70, 400                        # 사람 음성 F0 범위
lag_min, lag_max = int(sr/fmax), int(sr/fmin)
peak_lag = lag_min + int(np.argmax(ac[lag_min:lag_max]))
f0_est = sr / peak_lag

fig, ax = plt.subplots(figsize=(11, 3.8))
ax.plot(np.arange(len(ac))/sr*1000, ac)
ax.axvline(peak_lag/sr*1000, color="C3", ls="--",
           label=f"첫 피크 lag={peak_lag} 샘플 → F₀ = {f0_est:.1f} Hz")
ax.axvspan(lag_min/sr*1000, lag_max/sr*1000, color="C2", alpha=.1, label="탐색 범위 70~400 Hz")
ax.set(title="유성음 프레임의 자기상관", xlabel="지연 lag (ms)", ylabel="c_k", xlim=(0, 25))
ax.legend(); ax.grid(alpha=.3)
plt.tight_layout(); plt.show()
```


```python
# librosa의 전문 피치 추정기와 비교 (YIN 계열 — 자기상관을 개량한 알고리즘)
f0, voiced_flag, voiced_prob = librosa.pyin(y, fmin=70, fmax=400, sr=sr,
                                            frame_length=1024, hop_length=HOP_LEN)
t_f0 = librosa.times_like(f0, sr=sr, hop_length=HOP_LEN)

print("내가 손으로 구한 F₀ : %.1f Hz" % f0_est)
print("pyin 이 같은 위치에서 준 F₀ : %.1f Hz" % f0[min(voiced_idx, len(f0)-1)])
print("전체 구간 F₀ 중앙값 : %.1f Hz" % np.nanmedian(f0))

fig, ax = plt.subplots(figsize=(11, 4))
show_spec(D, sr, HOP_LEN, "스펙트로그램 위에 F₀ 궤적 겹쳐 그리기", ax=ax, fmax=1500)
ax.plot(t_f0, f0, color="cyan", lw=2, label="pyin F₀")
ax.legend()
plt.tight_layout(); plt.show()
```

### 🔍 관찰 포인트
- F₀ 곡선이 **스펙트로그램 빗살의 첫 번째 줄**과 일치합니다.
- 무성음·휴지 구간에서는 `NaN` 이 나옵니다 (`voiced_flag` 가 False).
- **F₀와 포먼트(F₁,F₂)는 전혀 다른 개념**입니다. F₀는 성대 진동 속도, 포먼트는 성도의 공명. 교재가 "혼동 주의"라고 강조한 부분입니다.

### ✏️ 직접 해보기
`fmin`, `fmax` 를 남성 음성(80~200)과 여성 음성(150~350)에 맞춰 바꿔 보고, F₀ 중앙값으로 화자의 성별을 추측해 보세요.

---
# 5교시 (35분) — 켑스트럼 · 멜 스케일 · MFCC  〈교재 2.8〉

## 5.1 이론 5분 요약

로그 스펙트럼은 **두 가지가 겹쳐진 것**입니다.

| 성분 | 모양 | 정체 | 담고 있는 정보 |
|---|---|---|---|
| **포락선 (envelope)** | 천천히 변하는 큰 곡선 | 성도(vocal tract)의 공명 | **모음이 무엇인가** (언어 내용) |
| **고조파 구조 (harmonic)** | 촘촘한 빗살 | 성대 진동 | **음높이 F₀** (화자·억양) |

이 둘을 분리하려면? → **로그 스펙트럼을 한 번 더 변환**합니다. 그 결과가 **켑스트럼(cepstrum)**.
- spectrum → **cepstrum**, frequency → **quefrency**(케프렌시, 단위는 초)
- **느린 성분(포락선)은 낮은 케프렌시**, **빠른 빗살(F₀)은 높은 케프렌시**에 모입니다

```
파형 → [윈도잉] → [DFT] → [log|·|] → [역변환] → 켑스트럼
```

## 실습 5-1. 켑스트럼을 직접 계산하기 (4줄이면 끝납니다)


```python
NF = 1024
frame_v = seg_v * sps.get_window("hann", FRAME_LEN)

# ① DFT → ② 로그 크기 → ③ 역변환
spec     = np.fft.rfft(frame_v, n=NF)
log_mag  = np.log(np.abs(spec) + 1e-10)
cepstrum = np.fft.irfft(log_mag, n=NF)

quefrency_ms = np.arange(NF) / sr * 1000     # 케프렌시 축(ms)

# F0에 해당하는 케프렌시 피크 찾기 (2.5~14 ms ≈ 70~400 Hz)
q_lo, q_hi = int(sr/400), int(sr/70)
peak_q = q_lo + int(np.argmax(cepstrum[q_lo:q_hi]))
print("켑스트럼 피크 케프렌시 = %.2f ms → F₀ = %.1f Hz" % (peak_q/sr*1000, sr/peak_q))

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(np.fft.rfftfreq(NF, 1/sr), log_mag)
axes[0].set(title="① 로그 스펙트럼 (포락선 + 빗살이 겹쳐 있음)", xlabel="주파수 (Hz)")
axes[1].plot(quefrency_ms[:NF//2], cepstrum[:NF//2])
axes[1].axvline(peak_q/sr*1000, color="C3", ls="--", label=f"F₀ 피크 → {sr/peak_q:.0f} Hz")
axes[1].set(title="② 켑스트럼", xlabel="케프렌시 (ms)", xlim=(0, 20)); axes[1].legend()
for a in axes: a.grid(alpha=.3)
plt.tight_layout(); plt.show()
```

## 실습 5-2. 리프터링(liftering) — 포락선과 미세구조 분리

켑스트럼에서 **낮은 계수만 남기면 포락선**, **높은 계수만 남기면 고조파 구조**가 됩니다.
교재의 *"고조파를 지워도 언어 내용은 남지만, 포락선을 지우면 톱니파 버징 소리만 남는다"* 를 확인합니다.


```python
def lifter(cep, n_keep, mode="low"):
    # mode='low' → 낮은 케프렌시만 남김(포락선), 'high' → 높은 것만 남김(고조파)
    m = np.zeros_like(cep)
    if mode == "low":
        m[:n_keep] = 1
        m[-n_keep+1:] = 1
    else:
        m[n_keep:-n_keep+1] = 1
    return cep * m


env_log   = np.fft.rfft(lifter(cepstrum, 20, "low")).real    # 포락선만
fine_log  = np.fft.rfft(lifter(cepstrum, 20, "high")).real   # 고조파 구조만
f_axis = np.fft.rfftfreq(NF, 1/sr)

fig, ax = plt.subplots(figsize=(11, 4.5))
ax.plot(f_axis, log_mag,  alpha=.45, label="원래 로그 스펙트럼")
ax.plot(f_axis, env_log,  lw=2.5,    label="포락선 (낮은 켑스트럼 20개) — 모음 정보")
ax.plot(f_axis, fine_log - 3, alpha=.7, label="고조파 구조 (높은 켑스트럼) — F₀ 정보 (보기 좋게 -3 이동)")
ax.set(title="켑스트럼 리프터링으로 포락선과 고조파 분리", xlabel="주파수 (Hz)", ylabel="log |X|")
ax.legend(fontsize=9); ax.grid(alpha=.3)
plt.tight_layout(); plt.show()
```

### 🔍 관찰 포인트
- 굵은 포락선의 **봉우리 위치가 곧 포먼트 F₁, F₂, F₃** 입니다. 몇 Hz쯤인가요?
- 교재 기준: F₁ 300~800 Hz (혀 높이), F₂ 1300~2100 Hz (혀 앞뒤).

## 실습 5-3. 멜 스케일과 멜 필터뱅크

$$\mathrm{mel}(f)=2595\log_{10}\!\left(1+\frac{f}{700}\right)$$

사람의 귀는 저주파에서 예민하고 고주파에서 둔합니다. 그 지각 특성에 맞춰 주파수 축을 휘는 것이 **멜 스케일**.
멜 축에서 **등간격으로 삼각 필터**를 배치해 전력 스펙트럼을 평균내면 → **적은 계수로 포락선을 요약**할 수 있습니다.


```python
f_hz = np.linspace(0, 8000, 400)
mel  = 2595 * np.log10(1 + f_hz/700)              # 교재의 식 직접 구현
mel_librosa = librosa.hz_to_mel(f_hz, htk=True)   # librosa (htk=True가 같은 식)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(f_hz, mel, lw=2, label="직접 구현")
axes[0].plot(f_hz, mel_librosa, "--", label="librosa.hz_to_mel(htk=True)")
axes[0].set(title="멜 스케일 곡선", xlabel="주파수 (Hz)", ylabel="Mel"); axes[0].legend()

mel_fb = librosa.filters.mel(sr=sr, n_fft=N_FFT, n_mels=26)   # (26, 257)
for i in range(mel_fb.shape[0]):
    axes[1].plot(librosa.fft_frequencies(sr=sr, n_fft=N_FFT), mel_fb[i], lw=1)
axes[1].set(title="멜 삼각 필터뱅크 26개 — 저주파에 촘촘, 고주파에 성기게",
            xlabel="주파수 (Hz)", ylabel="가중치")
for a in axes: a.grid(alpha=.3)
plt.tight_layout(); plt.show()

print("멜 필터뱅크 shape :", mel_fb.shape, " → (필터 개수, 주파수 bin 수)")
print("1000 Hz → %.1f mel,  2000 Hz → %.1f mel  (두 배 주파수인데 멜은 두 배가 아님!)"
      % (librosa.hz_to_mel(1000, htk=True), librosa.hz_to_mel(2000, htk=True)))
```


```python
# 멜 스펙트로그램 = 멜 필터뱅크를 전력 스펙트로그램에 곱한 것
S_mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=N_FFT,
                                       hop_length=HOP_LEN, win_length=FRAME_LEN, n_mels=40)
S_mel_db = librosa.power_to_db(S_mel, ref=np.max)
print("일반 스펙트로그램 :", D.shape, " → 프레임당 %d개 계수" % D.shape[0])
print("멜 스펙트로그램   :", S_mel.shape, " → 프레임당 %d개 계수 (대폭 압축)" % S_mel.shape[0])

fig, axes = plt.subplots(2, 1, figsize=(11, 7))
show_spec(D, sr, HOP_LEN, "① 선형 주파수 스펙트로그램 (257개 계수)", ax=axes[0])
img = librosa.display.specshow(S_mel_db, sr=sr, hop_length=HOP_LEN,
                               x_axis="time", y_axis="mel", ax=axes[1], cmap="magma")
axes[1].set(title="② 멜 스펙트로그램 (40개 계수)")
plt.tight_layout(); plt.show()
```

## 실습 5-4. MFCC — 멜 로그 스펙트럼에 DCT를 걸면 끝

남은 문제: 이웃한 멜 계수끼리 **상관이 높아** 정보가 여러 샘플에 퍼져 있습니다.
→ **DCT(이산 코사인 변환)** 로 비상관화(decorrelate) → **MFCC**

$$\text{파형} \to \text{윈도잉} \to \text{DFT} \to \text{멜 필터뱅크} \to \log \to \text{DCT} \to \text{MFCC}$$

보통 **12~13개 계수**만 남깁니다. 머신러닝 프론트엔드의 사실상 표준입니다.


```python
n_mfcc = 13
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=N_FFT,
                            hop_length=HOP_LEN, win_length=FRAME_LEN, n_mels=40)
print("MFCC shape :", mfcc.shape, " → (계수 13개, 프레임 수)")

# librosa가 내부에서 하는 일을 직접 따라해 보기 (멜 로그 스펙트럼 → DCT)
from scipy.fftpack import dct
mfcc_manual = dct(librosa.power_to_db(S_mel), axis=0, type=2, norm="ortho")[:n_mfcc]
print("직접 구현과 librosa 결과의 최대 차이: %.2e  → 같은 계산입니다" %
      np.max(np.abs(mfcc - mfcc_manual)))
```


```python
# 델타(Δ)와 델타-델타(ΔΔ): 시간에 따른 변화율 / 가속도
mfcc_d1 = librosa.feature.delta(mfcc)
mfcc_d2 = librosa.feature.delta(mfcc, order=2)
print("표준 음성인식 특징 벡터 차원 = 13 + 13 + 13 = %d" %
      (mfcc.shape[0] + mfcc_d1.shape[0] + mfcc_d2.shape[0]))

fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
for ax, M, title in zip(axes, [mfcc, mfcc_d1, mfcc_d2],
                        ["MFCC (정적 특징)", "Δ MFCC (변화율)", "ΔΔ MFCC (가속도)"]):
    img = librosa.display.specshow(M, sr=sr, hop_length=HOP_LEN, x_axis="time", ax=ax)
    ax.set(title=title, ylabel="계수 번호")
    ax.figure.colorbar(img, ax=ax, pad=.01)
plt.tight_layout(); plt.show()
```


```python
# 세 가지 표현을 나란히: 정보가 어떻게 압축되는가
fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
show_spec(D, sr, HOP_LEN, f"① 스펙트로그램 — 프레임당 {D.shape[0]}개", ax=axes[0])
librosa.display.specshow(S_mel_db, sr=sr, hop_length=HOP_LEN, x_axis="time",
                         y_axis="mel", ax=axes[1], cmap="magma")
axes[1].set(title=f"② 로그 멜 스펙트로그램 — 프레임당 {S_mel.shape[0]}개")
librosa.display.specshow(mfcc, sr=sr, hop_length=HOP_LEN, x_axis="time", ax=axes[2])
axes[2].set(title=f"③ MFCC — 프레임당 {mfcc.shape[0]}개", ylabel="계수 번호")
plt.tight_layout(); plt.show()
```

### 🔍 관찰 포인트
- ①→②→③ 으로 갈수록 **계수는 줄지만 음소의 경계는 여전히 뚜렷**합니다. 좋은 특징 추출의 정의입니다.
- MFCC 0번 계수는 전체 에너지(로그 평균)를 나타내서 값이 유독 큽니다. 실무에서는 빼거나 별도 처리합니다.
- 교재의 경고: 멜 스케일이 완벽히 정당화된 선택은 아니지만 **오래 써서 표준이 된 것**입니다. 바꾸려면 강력한 근거가 필요합니다.

### ✏️ 직접 해보기
`n_mels` 를 40 → 13 → 80 으로 바꾸면 멜 스펙트로그램이 어떻게 변하나요? `n_mfcc` 를 40으로 늘리면 아래쪽 계수에 무엇이 나타나나요?

---
# 6교시 (10분) — 선형 예측(LPC)과 마무리  〈교재 2.9〉

## 6.1 이론 3분 요약

음성은 연속적이라 **인접 샘플끼리 강하게 상관**되어 있습니다. 그러면 이전 샘플들로 현재를 예측할 수 있습니다.

$$\hat{x}_n = -\sum_{k=1}^{M} a_k x_{n-k}, \qquad e_n = x_n - \hat{x}_n \;(\text{예측 잔차})$$

- 계수 $a_k$ → 필터 $H(z)=1/A(z)$ → 그 주파수 응답이 곧 **스펙트럼 포락선**(= 포먼트!)
- 잔차 $e_n$ → 성대가 만드는 **여기(excitation) 신호**
- 차수 권장값: $M \approx F_s/1000 + 2$ (16 kHz → 18)
- 이 구조가 CELP 코덱의 핵심: **잔차만 전송해 비트레이트를 크게 줄임**


```python
order = int(sr/1000) + 2
frame64 = frame_v.astype(np.float64)

a = librosa.lpc(frame64, order=order)          # a[0] = 1
print("LPC 차수 M =", order, " / 계수 개수 =", len(a))

# ① LPC 포락선: H(z) = 1/A(z) 의 주파수 응답
w_f, h = sps.freqz([1.0], a, worN=512, fs=sr)
lpc_env_db = 20*np.log10(np.abs(h) + 1e-10)

# ② 예측 잔차 e_n = A(z) 를 신호에 통과시킨 것
resid = sps.lfilter(a, [1.0], frame64)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
Xf = 20*np.log10(np.abs(np.fft.rfft(frame64, n=NF)) + 1e-10)
axes[0].plot(np.fft.rfftfreq(NF, 1/sr), Xf, alpha=.5, label="로그 스펙트럼")
axes[0].plot(w_f, lpc_env_db + (Xf.max() - lpc_env_db.max()), lw=2.5, color="C3",
             label=f"LPC 포락선 (M={order})")
axes[0].set(title="LPC가 추정한 스펙트럼 포락선 = 포먼트", xlabel="주파수 (Hz)", ylabel="dB")
axes[0].legend(); axes[0].grid(alpha=.3)

axes[1].plot(np.arange(len(frame64))/sr*1000, frame64, label="원 신호 x_n")
axes[1].plot(np.arange(len(resid))/sr*1000, resid, label="예측 잔차 e_n", alpha=.8)
axes[1].set(title="잔차에는 성대 펄스만 남는다", xlabel="시간 (ms)")
axes[1].legend(); axes[1].grid(alpha=.3)
plt.tight_layout(); plt.show()

print("원 신호 에너지 / 잔차 에너지 = %.1f 배  → 예측이 이만큼 정보를 걷어냈습니다"
      % (np.sum(frame64**2)/np.sum(resid**2)))
```

### 🔍 관찰 포인트
- 붉은 포락선의 봉우리 = **포먼트**. 5-2의 켑스트럼 포락선과 거의 같은 위치인지 비교해 보세요.
- 잔차는 **뾰족한 펄스의 반복**입니다. 그 간격이 곧 피치 주기(1/F₀)입니다.
- 교재 2.1.7의 *"1차 차분만으로 진폭 범위 59% 감소 ≈ 1 bit/sample 절약"* 이 LPC의 가장 단순한 형태입니다.

---
# 종합 실습 — 한 화면에 전부 모으기

지금까지 배운 표현을 **하나의 그림**으로 쌓아 봅니다. 이것이 실제 음성 연구에서 가장 먼저 그리는 그림입니다.


```python
fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)

librosa.display.waveshow(y, sr=sr, ax=axes[0])
axes[0].set(title="① 파형 (시간 영역)", ylabel="진폭")

axes[1].plot(t_frames, rms_db, label="RMS (dB)")
axes[1].plot(t_frames, zcr*100, label="ZCR ×100", alpha=.8)
axes[1].set(title="② 단시간 에너지와 영교차율", ylabel="값"); axes[1].legend(); axes[1].grid(alpha=.3)

show_spec(D, sr, HOP_LEN, "③ 스펙트로그램 + F₀ 궤적", ax=axes[2], fmax=4000)
axes[2].plot(t_f0, f0, color="cyan", lw=1.8, label="F₀"); axes[2].legend()

librosa.display.specshow(S_mel_db, sr=sr, hop_length=HOP_LEN, x_axis="time",
                         y_axis="mel", ax=axes[3], cmap="magma")
axes[3].set(title="④ 로그 멜 스펙트로그램")

librosa.display.specshow(mfcc, sr=sr, hop_length=HOP_LEN, x_axis="time", ax=axes[4])
axes[4].set(title="⑤ MFCC", xlabel="시간 (s)", ylabel="계수 번호")

plt.tight_layout(); plt.show()
```

---
# 마무리 정리

## 오늘 배운 표현들의 계층 (교재 심화: 음성 신호의 계층적 구조)

| 레벨 | 표현 | librosa 함수 | 프레임당 계수 |
|---|---|---|---|
| 1 | 파형 (PCM) | `librosa.load` | 1 (샘플) |
| 2 | 단시간 특징 | `feature.rms`, `zero_crossing_rate`, `autocorrelate` | 1~2 |
| 3 | 스펙트럼 | `stft`, `amplitude_to_db` | 257 |
| 4 | 포락선 | 켑스트럼 리프터링, `librosa.lpc` | 13~18 |
| 5 | 지각적 특징 | `feature.melspectrogram`, `feature.mfcc` | 13~40 |
| 6 | 언어적 특징 | (다음 챕터: 음소·단어) | — |

## 꼭 기억할 숫자

| 항목 | 값 | 이유 |
|---|---|---|
| 샘플링 레이트 | **16 kHz** | 나이퀴스트 8 kHz, 마찰음까지 포함 |
| 윈도우 길이 | **20~30 ms** | 음소 하나가 정상(stationary)으로 보이는 최대 길이 |
| 스텝(hop) | **10~15 ms** | 50~60% 오버랩 |
| MFCC 계수 | **12~13개** | 포락선 정보를 담기에 충분 |
| 멜 필터 수 | **20~40개** | 해상도와 효율의 균형 |
| LPC 차수 | **Fs/1000 + 2** | 포먼트 개수에 맞춘 경험식 |
| 정규화 레벨 | **-26 dBov** | 처리 후 클리핑 방지 |

---
# 부록 A. 인터넷/마이크 없이 쓰는 합성 음성

예제 다운로드가 막혀 있거나 녹음 파일이 없으면 **아래 셀을 0교시 대신 실행**하세요.
유성음(모음) → 무성음(마찰음) → 유성음 순서의 가짜 음성을 만듭니다.


```python
def make_fake_speech(sr=16000, dur=2.4):
    t = np.arange(int(sr*dur)) / sr
    # ① 성대 진동: F0가 천천히 변하는 톱니파 (고조파가 풍부)
    f0 = 130 + 25*np.sin(2*np.pi*0.8*t)
    glottal = sps.sawtooth(2*np.pi*np.cumsum(f0)/sr)
    # ② 성도 필터: 포먼트 3개를 공명 필터로 구현
    voiced = glottal
    for f_form, q in [(700, 12), (1220, 14), (2600, 16)]:
        b, a_ = sps.iirpeak(f_form/(sr/2), Q=q)
        voiced = sps.lfilter(b, a_, voiced)
    voiced /= np.max(np.abs(voiced))
    # ③ 무성 마찰음: 고역 통과 잡음
    noise = np.random.randn(len(t))
    b, a_ = sps.butter(4, 3500/(sr/2), btype="high")
    fric = sps.lfilter(b, a_, noise) * 0.25
    # ④ 유성 - 무성 - 유성 - 무음 으로 이어 붙이기
    n = len(t)//4
    out = np.concatenate([voiced[:n], fric[:n//2], voiced[n:2*n], np.zeros(n//4),
                          voiced[2*n:3*n], fric[:n//3]])
    env = np.ones_like(out)                      # 경계 클릭 제거용 페이드
    fade = np.hanning(200)
    env[:100], env[-100:] = fade[:100], fade[-100:]
    return (out*env / np.max(np.abs(out)) * 0.9).astype(np.float32)


# 사용법: 아래 두 줄의 주석을 풀고 실행한 뒤, 0교시의 그 다음 셀들을 이어서 실행
# y, sr = make_fake_speech(), 16000
# display(Audio(data=y, rate=sr))
```

# 부록 B. 자주 나는 오류와 해결

| 증상 | 원인 | 해결 |
|---|---|---|
| `ModuleNotFoundError: librosa` | 설치 안 됨 | `pip install librosa` |
| 그래프 한글이 □□□ | 한글 폰트 없음 | 0교시 폰트 설정 셀 실행 / `pip install matplotlib` 후 나눔고딕 설치 |
| `NoBackendError`, mp3 로드 실패 | 디코더 없음 | `pip install soundfile audioread`, 또는 wav로 변환 |
| `librosa.example()` 에서 네트워크 오류 | 인터넷 차단 | 부록 A의 합성 음성 사용 |
| `TypeError: mfcc() takes 0 positional arguments` | librosa는 **키워드 인자 전용** | `librosa.feature.mfcc(y=y, sr=sr)` 처럼 이름을 꼭 붙일 것 |
| 스펙트로그램 x축 시간이 이상함 | `specshow`에 `hop_length`를 안 줌 | `specshow(..., sr=sr, hop_length=HOP_LEN)` |
| `Audio()` 재생기가 안 보임 | 주피터가 아닌 환경 | `soundfile.write("out.wav", y, sr)` 로 저장해서 듣기 |

# 부록 C. 교재 용어 ↔ librosa 함수 대조표

| 교재 (Chapter 2) | librosa |
|---|---|
| 파형 $x_n$, PCM | `librosa.load` → `y` |
| 샘플링 레이트 $F_s$ | `sr`, `librosa.resample` |
| 뮤-법칙 컴팬딩 | `librosa.mu_compress` / `mu_expand` |
| 윈도잉 $w_n$ | `scipy.signal.get_window`, `librosa.util.frame` |
| Overlap-Add, 완전 재구성 | `librosa.stft` ↔ `librosa.istft` |
| STFT $X(h,k)$ | `librosa.stft` |
| 로그 스펙트럼 $20\log_{10}|X|$ | `librosa.amplitude_to_db` (전력이면 `power_to_db`) |
| 스펙트로그램 | `librosa.display.specshow` |
| 신호 에너지, dB | `librosa.feature.rms` |
| 영교차율 ZCR | `librosa.feature.zero_crossing_rate` |
| 자기상관 $r_k$ | `librosa.autocorrelate` |
| 기본주파수 $F_0$ | `librosa.yin`, `librosa.pyin` |
| 멜 스케일, 멜 필터뱅크 | `librosa.hz_to_mel`, `librosa.filters.mel` |
| 켑스트럼 | `np.fft.rfft` → `log` → `np.fft.irfft` |
| MFCC, Δ, ΔΔ | `librosa.feature.mfcc`, `librosa.feature.delta` |
| 선형 예측 LPC | `librosa.lpc` |
| 프리엠퍼시스 | `librosa.effects.preemphasis` |

# 부록 D. 더 공부하려면

- librosa 공식 문서: <https://librosa.org/doc/latest/>
- Rabiner & Schafer, *Introduction to Digital Speech Processing* (2007)
- Jurafsky & Martin, *Speech and Language Processing* (3rd ed. draft)
- Aalto University, *Introduction to Speech Processing* (이 교재의 원본, 온라인 공개)
