# SPV Kickboard Project

구성:
- 실시간 카메라 영상에서 킥보드, 사람, 2인 탑승(동승) 등 안전 관련 객체 탐지
- 전처리: 히스토그램 평활화, 컬러필터, 라플라시안 샤프닝, 벡터 필터 + 적응형 선택
- 시프트 레지스터(temporal smoothing)
- YOLOv8 기반 탐지 파이프라인 (CPU 지원)

사용:
1. 가상환경 생성 및 패키지 설치
python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate
pip install -r requirements.txt

2. 데이터셋 준비: `datasets/kick_safety/`에 images/labels 구조로 넣기.
`config/dataset.yaml`을 참고.

3. 웹캠 실행:
python run.py

구성 파일과 주요 모듈은 `spv/` 디렉토리에 있습니다.

4. 