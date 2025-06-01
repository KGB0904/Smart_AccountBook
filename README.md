# Smart_AccountBook  
딥러닝 프로젝트: OCR&GPT API를 활용한 1차 분류 + KcELECTRA를 활용한 2차 소비 항목 분류

kc_model 폴더는 학습 이후 생성되는 폴더입니다. 딥러닝 모델 특성상 정확도가 학습마다 달라지기에, 잘 나올 때까지 여러번 시행착오가 필요할 수 있습니다.

폰트파일 없을시 validation 시각화가 불가능합니다 (Arial, Noto-sans 등 사용하면 plt 환경에서 한글 깨짐). 나눔고딕 폰트 인터넷에서 다운로드 받으시고, 동일 디렉터리에 넣어주세요.

## Requirements
**conda 환경 권장**
로컬 환경에서도 물론 가능합니다. 하지만 버전 문제 발생 가능성이 매우 높습니다.

```bash
# 1. Python 3.10 기반 가상환경 생성
conda create -n <your-env-name> python=3.10
conda activate <your-env-name>

# 2. GPU 사용 시 (CUDA 11.8 환경)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 3. GPU 정상 인식 확인
python -c "import torch; print(torch.cuda.is_available())"

# 4. 필수 패키지 설치
pip install transformers datasets scikit-learn pandas tqdm regex protobuf matplotlib google-cloud-vision openai
```

## Usage
```bash
# 1. train_final.py 먼저 실행. 이때, 데이터셋을 동일 디렉터리에 넣어야 함. epoch는 7~9에서 최적.

# 2. 이후 생성된 모델 학습 결과의 성능이 관찰하고 싶다면 validation.py 실행. 모델의 전반적인 성능 및 정확도 측정 가능 (f1-score, confusion matrix 시각화 등이 포함됨)

# 3. 학습이 완료됐다면 ocr.ipynb파일에 본인의 api를 작성 후(과금이 발생하기에 해당 코드에는 포함하지 않았음) 순차적으로 실행. 결과적으로 gpt_receipt_result.json 이라는 ocr 및 항목별 parsing이 완료된 파일 생성.

# 4. 위의 과정이 모두 완료됐다면(2번은 제외해도 무관) infer_final.py 를 실행. 결과적으로 gpt_receipt_with_categories.json 파일이 생성되며, 기존 비어있던 카테고리에 분류 결과가 채워짐.
