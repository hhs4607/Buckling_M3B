# 복합재 빔 좌굴 해석 프로그램 - M3 (Composite Beam Buckling Analysis)

이 프로젝트는 마이크로역학(Micromechanics)과 고전 적층 이론(CLT)을 사용하여 복합재 빔의 좌굴 거동을 해석하는 전문가용 도구입니다. 두 가지 솔버 코어를 제공합니다:
- **M3**: 루트 회전 스프링을 포함한 정밀한 2-term Ritz 모델.
- **M2**: 빠른 연산을 위한 1-term Ritz 모델 (Clamped 조건 가정).

## 시스템 요구사항
- Python 3.8 이상
- `pip` (Python 패키지 관리자)

## 설치 방법

1.  **저장소 다운로드**: 이 저장소를 클론(clone)하거나 다운로드합니다.
2.  **가상환경 설정 (권장)**:
    프로젝트 격리를 위해 가상환경을 사용하는 것을 권장합니다.
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # 윈도우의 경우: venv\Scripts\activate
    ```
3.  **의존성 패키지 설치**:
    ```bash
    pip install -r requirements.txt
    ```

## 실행 방법

### 실행 스크립트 사용 (Mac/Linux)
1.  스크립트에 실행 권한 부여 (최초 1회):
    ```bash
    chmod +x run_gui.sh
    ```
2.  스크립트 실행:
    ```bash
    ./run_gui.sh
    ```
    *스크립트는 `venv` 폴더가 존재하면 자동으로 가상환경을 사용하여 실행합니다.*

### Python 직접 실행
```bash
python3 gui_buckling.py
```

## 사용 설명서 (Operation Manual)

GUI는 세 가지 주요 탭으로 구성되어 있으며, 각각 다른 해석 목적을 가집니다.

### 1. Full Analysis 탭 (상세 해석)
단일 설계 포인트에 대한 정밀 해석을 수행합니다.
- **Inputs (입력)**: 왼쪽 사이드바에서 기하 형상, 적층 정보, 물성치를 설정합니다.
- **Run**: `Run Full Analysis` 버튼을 눌러 임계 좌굴 하중($P_{cr}$)과 모드 형상을 계산합니다.
- **Outputs (출력)**:
    - **Load-Deflection Plot**: 전체, 선형, 비선형 처짐 경로를 그래프로 보여줍니다.
    - **Mode Contour**: 실제 빔 형상(상단 뷰)에 따른 좌굴 모드 형상 $w(x,y)$를 등고선(Contour)으로 시각화합니다. (Y축 대칭 및 실제 비율 반영)

### 2. Sensitivity (OAT) 탭 (민감도 해석)
단일 변수가 안정성에 미치는 영향을 분석합니다 (One-At-a-Time).
- **Configuration (설정)**: `L`, `Ef`, `t_face_total` 등 15개의 수치 변수 목록이 제공됩니다.
    - **Select**: 분석할 변수의 체크박스를 선택합니다.
    - **Range**: 변화 범위 `±%`를 입력합니다 (예: 10 입력 시 ±10%).
    - **Resolution**: `Pts`에 계산할 포인트 수를 입력합니다.
- **Run**: `RUN SENSITIVITY` 버튼을 클릭합니다.
- **Outputs**: 선택한 각 변수에 대해 $P_{cr}$ 변화 그래프가 그리드 형태로 표시됩니다.

### 3. Uncertainty (Sobol) 탭 (불확실성 해석)
전역 민감도 해석(Global Sensitivity Analysis, GSA)을 통해 변수의 중요도를 정량화합니다.
- **Configuration (설정)**:
    - **Select**: 불확실성을 고려할 변수를 선택합니다.
    - **Bounds**: 각 변수의 `Low`(최소) 및 `High`(최대) 절대 범위를 설정합니다. (기본값은 현재 입력의 ±10%로 자동 설정됨)
    - **N_base**: 기본 샘플 수를 설정합니다 (총 계산 횟수 $\approx N(k+2)$).
- **Run**: `RUN SOBOL` 버튼을 클릭합니다.
- **Outputs**: **Sobol 지수(Indices)**가 막대 그래프로 표시됩니다.
    - $S_i$ (1차 지수): 해당 변수 단독의 영향력.
    - $S_{Ti}$ (총 지수): 다른 변수와의 상호작용을 포함한 전체 영향력.

## 결과 내보내기 (Export)
- **Save Config**: 현재 입력 상태를 JSON 파일로 저장하여 나중에 다시 불러올 수 있습니다.
- **Load Config**: 저장된 JSON 설정을 불러옵니다.
- **Excel Results**: 상세 해석 데이터(OAT 결과, Sobol 지수 등)는 실행 폴더 내에 엑셀 파일로 임시 저장되어 추가 분석에 활용할 수 있습니다.
