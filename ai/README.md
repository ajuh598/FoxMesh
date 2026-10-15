# FoxMesh AI

FoxMesh 플랫폼 전용 3D 생성 모델 **FoxMesh**와, 이 모델의 데이터셋 구축·학습·평가·버전 관리를 위한 라이브러리
(`foxmesh-ai`)입니다. 학습한 모델은 플랫폼의 자동 선택 엔진에 등록되어 다른 모델과 같은 기준으로 선택됩니다.

```
ai/foxmesh_ai/
  geometry.py        메시 입출력(OBJ/STL/PLY/GLB), 정규화, 표면 샘플링, 복셀화, manifold surface nets
  text.py            한국어·영어 토크나이저 (문자 n-gram 해시, 어휘 파일 없음)
  render.py          조건 이미지 렌더러 (이미지 → 3D 학습용)
  data/dataset.py    데이터셋 관리: 매니페스트, 가져오기, 중복 제거, 분할, 검증, 통계, 버전 지문
  data/synthetic.py  절차적 학습 데이터 16종 + 한국어/영어 설명문
  data/preprocess.py 학습 샘플 생성 (점유도 라벨, 표면 점·색, 렌더 이미지)
  model.py           FoxMesh 모델
  train.py           학습·이어서 학습·파인튜닝
  evaluate.py        생성 품질 평가 (IoU, Chamfer, F-score, 폐쇄성)
  generate.py        추론: 텍스트·이미지 → 색 있는 닫힌 메시
  store.py           모델 버전 관리, 모델 카드, 플랫폼 등록용 설명 파일 내보내기
  serving.py         GPU 추론 서버용 러너
  cli.py             foxmesh-ai 명령
```

## 모델 구조

```
텍스트 ─► TextEncoder (문자 n-gram 해시 임베딩 + Transformer) ─┐
                                                               ├─► 조건 벡터 c
이미지 ─► ImageEncoder (CNN, RGBA: 알파 = 물체 마스크) ────────┘
                                                                     │ FiLM
3D 점 x ─► Fourier 특징 ──► 조건부 MLP (잔차 블록) ──► 점유도(안/밖) 로짓, RGB 색
```

- 텍스트만, 이미지만, 둘 다 입력할 수 있습니다. 학습할 때 표본마다 입력 방식을 무작위로 골라 한 모델이 모두 처리합니다.
- 생성: 격자(기본 64³)에서 점유도를 평가하고 **manifold surface nets**로 메시를 추출합니다. 결과는 항상
  닫혀 있고, 모든 모서리가 정확히 두 면에 속합니다(3D 프린팅 검증 통과). 색은 꼭짓점마다 예측합니다.
- 프리셋: `base`(GPU용, 폭 256·블록 5·이미지 128px), `small`(CPU용, 폭 192·블록 4·이미지 64px).
- 한국어 조사가 붙은 단어도 같은 특징을 공유합니다("검을"과 "검"이 `<검`을 공유). 어휘 파일이 필요 없습니다.

## 빠른 시작

```bash
pip install -e "ai[torch,dev]"

# 1. 데이터셋: 합성 데이터로 시작하고, 실제 에셋을 추가
foxmesh-ai dataset synth data/foxmesh --count 2000
foxmesh-ai dataset import data/foxmesh ~/assets/chairs --license CC-BY-4.0 --tag furniture
foxmesh-ai dataset validate data/foxmesh --deep
foxmesh-ai dataset preprocess data/foxmesh --workers 8

# 2. 학습 (중단해도 --resume으로 이어서)
foxmesh-ai train data/foxmesh --out runs/v1 --preset small --epochs 60

# 3. 등록·평가·운영 지정
foxmesh-ai models register --store models runs/v1/checkpoints/best.pt --eval data/foxmesh --promote
foxmesh-ai models list --store models
foxmesh-ai generate --store models --prompt "빨간 의자" --out chair.glb
foxmesh-ai generate --store models --image photo.png --out from_photo.glb

# 4. 플랫폼 자동 선택 엔진에 등록
foxmesh-ai models export --store models --to /srv/foxmesh/models
FOXMESH_MODELS_DIR=/srv/foxmesh/models foxmesh serve
```

새 데이터로 기존 모델을 이어서 학습할 때는 다음처럼 합니다.

```bash
foxmesh-ai finetune data/new_assets --out runs/v2 --store models --from v1 [--freeze text image]
foxmesh-ai models register --store models runs/v2/checkpoints/best.pt --parent v1 --eval data/new_assets
foxmesh-ai models promote --store models v2
```

## 데이터셋 형식

```
data/foxmesh/
  dataset.json   이름, 설명, 라이선스, 분할 비율
  items.jsonl    항목마다 {id, mesh, captions[], images[], tags[], split, source, license, faces, watertight}
  meshes/        정규화된 메시(GLB, 꼭짓점 색 포함)
  images/        참고 이미지(선택)
  cache/         전처리 결과(.npz), 지워도 다시 만들 수 있음
```

- **항목 ID**는 정규화된 형상의 해시입니다. 같은 에셋을 다시 가져오면 중복으로 처리되고 설명문만 합쳐집니다.
- **분할**(train/val/test)은 ID 해시로 정해지므로, 항목을 추가하거나 지워도 기존 항목의 분할이 바뀌지 않습니다.
- **가져오기**: `chair.glb`와 같은 이름의 `chair.txt`(한 줄에 설명문 하나)나 `chair.json`
  (`{"captions": [...], "tags": [...], "license": "..."}`)을 설명으로 씁니다. 없으면 파일 이름을 씁니다.
  같은 이름의 이미지(`chair.png`, `chair_1.png`)는 이미지→3D 학습에 함께 쓰입니다.
- **버전 지문**(`fingerprint`)은 데이터셋 내용의 해시이며, 학습한 모델마다 기록되어 어떤 데이터로 학습했는지 추적합니다.
- **라이선스**: 항목마다 라이선스를 기록하고, `validate`가 라이선스를 알 수 없는 항목을 경고합니다. 외부 에셋은
  학습·상업적 이용이 허용되는지 확인한 뒤 가져오세요.

## 학습과 평가

- 손실: 표면 근처 점과 공간 전체 점의 점유도 BCE + 표면 점 색 L1
- AdamW, warmup + 코사인 학습률, 기울기 클리핑, EMA 가중치(생성에 사용)
- 에폭마다 검증 IoU로 `best.pt` 저장, `last.pt`로 이어서 학습, 조기 종료, `train_log.jsonl` 로그
- 파인튜닝: 낮은 학습률과 조기 종료가 기본값이며, 인코더나 디코더를 고정할 수 있음
- 평가(`foxmesh-ai eval`): 검증 분할에서 텍스트·이미지 입력별로 IoU, Chamfer-L1, F-score@0.02, 닫힌 메시
  비율, 생성 시간, 태그(형태 종류)별 점수를 측정해 버전과 모델 카드에 기록

## 플랫폼 연동

`models export`가 쓰는 `foxmesh.json`은 플랫폼의 모델 설명 파일입니다. 평가 점수로 초기 품질 추정치를 채우고,
플랫폼은 실제 실행 결과와 벤치마크로 이를 대체합니다. 백엔드의 `foxmesh_ai` 어댑터는 모델을 메모리에 올려 두고
재사용하며, 운영 버전을 바꾸면 다시 불러옵니다. GPU 서버에서 따로 돌리려면 `serving.py`의 `FoxMeshRunner`를
추론 서버에 등록하면 됩니다.

## 한계

- 동봉된 v1은 합성 데이터 16종으로만 학습했으므로 그 16종 밖의 물체는 잘 만들지 못합니다. 평가 점수도 같은 분포의
  검증 데이터 기준이라 실제 요청에서의 품질보다 높게 나옵니다. 실제 에셋을 가져와 파인튜닝해야 범위가 넓어집니다.
- 입력이 같으면 결과도 같습니다(무작위 다양성 없음). 색은 텍스처가 아니라 꼭짓점 색입니다.
- 해상도 64³ 기준의 거친 형상이므로, 정교한 결과보다는 시제품이나 편집의 출발점에 적합합니다.
