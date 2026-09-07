# -*- coding: utf-8 -*-
"""
전체 파이프라인을 순서대로 실행한다.

    python analysis/run_pipeline.py            # 전처리 + 자산 생성 (대시보드용)
    python analysis/run_pipeline.py --full     # 분석 스크립트(RF/XGB/테스트평가)까지

산출물은 모두 스크립트로 재생성되므로 git 에는 원본 데이터만 두면 된다.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORE = ["split_data.py", "preprocess.py", "build_dashboard_assets.py"]
ANALYSIS = ["train_rf.py", "train_xgb.py", "evaluate_test.py"]


def run(script: str) -> None:
    print(f"\n{'='*60}\n▶ {script}\n{'='*60}")
    r = subprocess.run([sys.executable, str(ROOT / "analysis" / script)])
    if r.returncode != 0:
        sys.exit(f"실패: {script}")


def main() -> None:
    scripts = CORE + (ANALYSIS if "--full" in sys.argv else [])
    for s in scripts:
        run(s)
    print("\n완료. 대시보드 실행:  python -m streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
