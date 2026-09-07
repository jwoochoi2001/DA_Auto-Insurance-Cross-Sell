# -*- coding: utf-8 -*-
"""matplotlib 한글 폰트 설정 (플랫폼 무관, 없으면 경고 후 진행)."""
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

_CANDIDATES = [
    "Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR",
    "Noto Sans KR", "Source Han Sans KR", "Yu Gothic", "MS Gothic",
]
_WIN_PATHS = [r"C:\Windows\Fonts\malgun.ttf", r"C:\Windows\Fonts\malgunsl.ttf"]


def setup_korean_font() -> str:
    installed = {f.name for f in font_manager.fontManager.ttflist}
    for name in _CANDIDATES:
        if name in installed:
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
    for path in _WIN_PATHS:
        try:
            font_manager.fontManager.addfont(path)
            name = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams["font.family"] = name
            plt.rcParams["axes.unicode_minus"] = False
            return name
        except Exception:
            continue
    warnings.warn("한글 폰트를 찾지 못했습니다. 그래프의 한글이 깨질 수 있습니다.")
    plt.rcParams["axes.unicode_minus"] = False
    return "(default)"
