# AI 기반 웹 침입 탐지·로그 분석

웹 요청에 위험 점수를 매기고, 관리 화면에서 의심 로그의 입력 원문까지 볼 수 있게 만든 학부 프로젝트입니다. 저는 CNN 모델 구조와 학습 코드를 맡았고, Flask 탐지 API와 MySQL 로그 조회를 JavaScript 화면에 연결했습니다.

## 요청이 들어오면

입력 문자열을 정리하고 CountVectorizer의 단어 순서대로 횟수를 셉니다. 값 4,096개를 `64×64×1` 배열로 바꾸면 CNN이 SQL 인젝션 점수를 냅니다. 관리 화면에서는 소수 둘째 자리까지 표시한 점수가 0.6을 넘는 요청을 위험 목록에 넣었습니다. 목록에서 로그 번호를 선택하면 요청에 들어온 값을 확인할 수 있습니다.

1차 모델은 사전에 있는 단어가 하나도 없는 입력을 받을 때 정상 요청까지 공격으로 보는 일이 많았습니다. 입력 배열이 모두 0일 때 나온 0.5838점이 모델 비교 기준인 0.5를 넘었기 때문입니다. 2차 모델 비교에서는 CountVectorizer와 CNN 특징 추출부를 유지하고, 정상·공격 예시를 보강해 Dense 분류층을 더 학습했습니다.

같은 HttpParamsDataset 평가 5,084건에 저장 모델을 적용했습니다.

| 모델 | 정확도 | F1 | 정상 오탐 | 공격 미탐 |
| --- | ---: | ---: | ---: | ---: |
| 1차 CNN | 41.60% | 54.65% | 2,724건 | 245건 |
| 2차 CNN | 94.75% | 93.02% | 12건 | 255건 |

## 코드

| 파일 | 내용 |
| --- | --- |
| [detector.py](detector.py) | 문자열 전처리, CountVectorizer 입력 배열, CNN 점수 계산 |
| [models](models/) | 1차·2차 가중치와 벡터 변환 정보 |
| [prepare_http.py](prepare_http.py) · [train_head.py](train_head.py) · [evaluate.py](evaluate.py) | 공개 데이터 분할, Dense 분류층 추가 학습, 평가 |
| [web](web/) | 요청 입력 화면과 위험 로그 조회 화면 |
| [app.py](app.py) | Flask 탐지 API, 로그 저장과 조회 |

학부 프로젝트의 웹 코드는 PHP, 로그 저장은 MySQL을 사용했습니다. 이 저장소에서 흐름을 확인할 때는 같은 화면·API 역할을 로컬 Flask와 SQLite로 구성했습니다. 모델 파일은 보관한 CNN의 학습용 옵티마이저 상태를 덜어낸 추론 파일입니다. 벡터라이저의 토큰 항목은 단어의 해시와 배열 위치로 저장했습니다.

## 화면 열기

Python 3.12 환경에서 실행합니다.

```bash
python -m pip install -r requirements.txt
python app.py
```

브라우저에서 `http://127.0.0.1:8000`을 열고 예제 값을 보내면 됩니다. 로그 화면에서 의심 요청과 원문을 확인할 수 있습니다. 모델은 기본으로 2차를 사용합니다. 1차를 보려면 환경 변수 `SQL_DETECTOR_MODEL`을 `baseline`으로 설정합니다. 예제 요청은 로컬 `instance/demo.sqlite3`에 저장됩니다.

## 공개 데이터로 분류층 학습하기

[HttpParamsDataset](https://github.com/Morzeux/HttpParamsDataset)의 CSV를 `data/source.csv`로 둡니다. 라벨은 정상 0, SQL 인젝션 1입니다. 비슷한 형식의 요청이 학습과 평가에 나뉘어 들어가는 일을 줄이기 위해 템플릿 단위로 분할합니다.

```bash
python prepare_http.py data/source.csv data/splits
python train_head.py --train data/splits/train.csv --validation data/splits/validation.csv --output runs/extra_head.h5
python evaluate.py data/splits/test.csv --model baseline
python evaluate.py data/splits/test.csv --model improved
```

학습 코드는 보관한 1차 CNN에서 Dense 분류층을 이어 학습하는 공개 데이터 예제입니다. 새로 학습한 모델은 평가할 때 `--model-file runs/extra_head.h5`로 지정합니다. 위 성능 표는 저장소에 포함된 1차·2차 모델을 기존 공개 평가 분할에서 비교한 값입니다.
