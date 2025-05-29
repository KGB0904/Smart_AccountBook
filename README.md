# Smart_AccountBook  
딥러닝 프로젝트: OCR + 소비 항목 분류 (KcELECTRA 기반)

kc_model 폴더는 용량이 크므로 메일로 요청시 전달방법을 찾아보겠습니다.

폰트파일 없을시 validation 시각화 안됩니다. 나눔고딕 폰트 동일 디렉터리에 넣어주세요.

## Requirements
**conda 환경 권장**

```bash
# 1. Python 3.10 기반 가상환경 생성
conda create -n <your-env-name> python=3.10
conda activate <your-env-name>

# 2. GPU 사용 시 (CUDA 11.8 환경)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 3. GPU 정상 인식 확인
python -c "import torch; print(torch.cuda.is_available())"

# 4. 필수 패키지 설치
pip install transformers datasets scikit-learn pandas tqdm regex protobuf matplotlib google-cloud-vision
