## 공유 킥보드 안전 관리 시스템

## 목표: 공유 킥보드 2인 이상 동승 감지 및 경고 알림 시스템 개발

## 개발 환경: YOLOv8n![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white) ![python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)

## 킥보드 데이터셋 : https://universe.roboflow.com/kts-data/kickboard-data | https://universe.roboflow.com/new-workspace-zuae9/last-jevns

## 프로젝트 폴더 구조

```text
AI_PROJECT/
├── models/
│ ├── yolov8n.pt # YOLOv8n 사전 훈련 가중치 파일
│ └── best.pt # 사용자 정의 훈련 후 저장된 최적 가중치 파일
├── data/
│ ├── images/
│ │ ├── train/ # 훈련 이미지
│ │ └── val/ # 검증 이미지
│ ├── labels/
│ │ ├── train/ # 훈련 라벨 (YOLO 형식 .txt)
│ │ └── val/ # 검증 라벨 (YOLO 형식 .txt)
│ └── dataset.yaml # YOLO 데이터셋 설정 파일 (클래스 이름, 경로 등)
├── src/
│ ├── detect_and_alert.py # 실시간 감지 및 경고 로직 (핵심 실행 파일)
│ ├── train_model.py # 모델 훈련 스크립트 (선택 사항, 필요 시)
│ └── utils.py # 보조 함수/클래스 (예: OpenCV 이미지 처리 함수 등)
├── requirements.txt # 프로젝트에 필요한 라이브러리 목록
└── README.md # 프로젝트 설명 및 실행 방법
```

## 역할 분담

## 데이터셋 - 이지온, 객체 - 이선재, 모델 - 전형규
