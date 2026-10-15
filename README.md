# FoxMesh

Blender 기반 지능형 3D 생성 AI 플랫폼. 사용자의 요청을 분석해 등록된 AI 모델 중 가장 적합한 것을
**규칙과 점수로** 자동 선택하고, 생성 → Blender 후처리 → 자동 검증까지 사람의 개입 없이 실행합니다.
AI끼리 자연어로 대화하거나 토론하는 단계는 없습니다. 모든 단계는 구조화된 데이터(`TaskSpec`,
`RoutingDecision`, Blender 리포트)로 연결됩니다.

```
사용자 요청 ─▶ 요청 분석(규칙) ─▶ 후보 필터링(필수 조건) ─▶ 점수 기반 선택 ─▶ AI 생성
           ─▶ Blender 후처리(필요한 단계만) ─▶ 자동 검증 ─▶ 실패 시 제한된 재시도/대체 모델 ─▶ 결과
```

## 구성

| 경로 | 역할 |
| --- | --- |
| `backend/foxmesh/classifier.py` | 규칙 기반 요청 분석. 한국어·영어 키워드와 정규식으로 작업 유형, 목적(프로필), 출력 형식, 폴리곤 예산, 크기를 추출. LLM 호출 없음, 결과 캐싱 |
| `backend/foxmesh/registry/` | AI 모델 레지스트리. 모델 하나 = JSON 파일 하나(`registry/models/*.json`). 실행 통계(실행 수, 실패율, 평균 시간, 검증 점수)를 실제 로그로 누적 |
| `backend/foxmesh/router.py` | 작업 라우팅 엔진. 필수 조건 필터 후 가중 점수로 선택 |
| `backend/foxmesh/gpu.py` | `nvidia-smi` 기반 VRAM 모니터링과 작업별 VRAM 예약 |
| `backend/foxmesh/adapters/` | 모델 실행 어댑터: 내장 절차적 생성기(CPU), 내장 이미지→3D(CPU), GPU 추론 서버용 HTTP 어댑터, entry point 플러그인 |
| `backend/foxmesh/blender/` | Blender 자동화: 작업별 단계 선택(`planner.py`), Blender 내부 스크립트(`bpy_pipeline.py`), 별도 프로세스 실행(`worker.py`) |
| `backend/foxmesh/validation.py` | 결과 검증: 파일 형식(내용 기반), 메시, 폴리곤 예산, 폐쇄성, 재질, UV, 크기 |
| `backend/foxmesh/pipeline.py` | 실행 엔진: 생성 → 후처리 → 검증, 재시도와 대체 모델 |
| `backend/foxmesh/jobs.py` | 비동기 작업 큐: 중복 요청 방지, 취소, 재시작 복구, 보존 기간 정리 |
| `backend/foxmesh/store.py` | 작업 저장소: SQLite(기본) / PostgreSQL / 메모리 |
| `backend/foxmesh/benchmark.py` | 모델별 성능 평가: 고정 테스트 세트 실행, 결과를 모델 통계에 반영 |
| `backend/foxmesh/api/` | FastAPI HTTP API |
| `backend/foxmesh/inference/` | GPU 추론 서버: 모델을 메모리에 유지(프로세스 재사용), VRAM 예산 내 LRU 언로드 |
| `frontend/` | Next.js 웹: 대시보드, 생성 페이지, 작업 상세(3D 미리보기), 실행 엔진 관리 |

## 모델 자동 선택

### 1) 필수 조건 (하나라도 어기면 점수와 무관하게 제외)

- 활성화 여부, 어댑터 설치 여부, 추론 서버 주소 설정 여부
- 작업 유형 지원 (텍스트→3D, 이미지→3D 등)
- 입력 지원 (이미지 작업은 이미지 입력 필수, 텍스트는 보조 정보)
- 출력 형식: 직접 출력하거나, Blender로 변환 가능한 형식일 것
- VRAM: 로컬 GPU 모델은 확보 가능한 VRAM 이하일 것 (원격/CPU 모델은 제외). FoxMesh 작업이 잠시 쓰고 있는 VRAM은
  곧 돌아오므로, 그 몫까지 합쳐 맞으면 후보에 남기고 실행 시 VRAM이 빌 때까지 기다림. 다른 프로세스가 쓰는 VRAM은 제외
- 라이선스: 상업적 사용이 필요하면(`FOXMESH_COMMERCIAL_USE`, 기본 true) 비상업 라이선스 제외
- 안정성(차단기): 최근 20회 중 5회 이상 실행해 실패율이 50%를 넘거나 연속 3회 실패하면 10분간 제외.
  이후 한 번 시험 실행을 허용하고(성공하면 정상 복귀, 실패하면 다시 10분 제외), 영구 제외되지 않음
- 마감 시간: 예상 실행 시간이 요청한 시간을 넘으면 제외

제외된 모델과 이유는 모두 `RoutingDecision.rejected`에 남고 웹에 표시됩니다.

### 2) 점수 (모든 항목 0~1로 정규화)

```
선택 점수 = w정확도·정확도 + w속도·속도 + w품질·품질 + w안정성·안정성 + w자원·자원효율
```

- **정확도**: 사전 추정치와 실제 검증 점수 평균을 표본 수에 따라 섞은 값(실행이 쌓일수록 실측 위주)
- **속도**: 실측 평균 시간(없으면 추정치)을 프로필 기준 시간 대비 로그 스케일로 환산. 기준 시간이면 0.5
- **품질**: 프로필마다 의미가 다름. 게임용은 토폴로지(폴리곤 효율), 3D 프린팅은 폐쇄성(watertight)과 형상 위주
- **안정성**: 최근 실패율에 베이지안 사전값을 둔 값
- **자원 효율**: 현재 남은 VRAM 대비 요구량

| 프로필 | 우선순위 | 최소 품질 |
| --- | --- | --- |
| 빠른 시제품 `prototype` | 속도 35%, 자원 25% | 0 |
| 균형 `balanced` | 정확도·품질 25%씩 | 0.3 |
| 고품질 `high_quality` | 품질 37%, 정확도 30% | 0.6 |
| 게임용 에셋 `game_asset` | 품질(토폴로지) 35% | 0.4 |
| 3D 프린팅 `print_3d` | 품질(폐쇄성) 35%, 안정성 30% | 0.4 |

최소 품질을 넘는 후보 중에서만 고르며, 점수 차가 0.02 이내면 더 빠르고 VRAM을 덜 쓰는 모델을 우선합니다.
가중치는 웹의 **실행 엔진 관리** 페이지나 `PUT /api/policies/{profile}`로 바꿀 수 있습니다.

## Blender 후처리 (필요한 단계만 실행)

| 작업 | 실행 단계 |
| --- | --- |
| 공통 | 불러오기, 검사, 내보내기, 썸네일 |
| 빠른 시제품 | + 크기·원점 정규화 |
| 게임용 에셋 | + 정규화, 법선 정리, UV 검사(없으면 자동 펼침), 재질 연결, 폴리곤 예산 초과 시 데시메이트 |
| 3D 프린팅 | + 정규화(요청 크기로 스케일), 법선 정리·정점 병합, 폐쇄성 검사와 구멍 메우기, STL은 mm 단위 |
| 고품질 | + 정규화, 법선, UV, 재질, BLEND 프로젝트 저장 |
| 미리보기 / 변환 | 썸네일만 / 형식 변환만 |

Blender는 항상 별도 프로세스(`blender --background`)로 실행되어 충돌·멈춤이 API 서버에 영향을 주지 않고,
작업 취소 시 프로세스를 종료합니다. Blender가 없으면 형식이 같은 파일만 복사하는 기본 처리로 동작합니다.

## 실행

### 백엔드

```bash
cd backend
pip install -e ".[dev]"

# Blender 경로 (없으면 PATH의 blender 사용, 그것도 없으면 기본 처리만)
export FOXMESH_BLENDER=/opt/blender/blender
# GPU를 감지하지 못하는 환경에서 VRAM 크기를 직접 지정 (GB)
export FOXMESH_VRAM_GB=24

foxmesh serve --port 8000                       # API 서버
foxmesh analyze "게임에 사용할 중세풍 검을 만들어 줘"   # 선택 과정만 확인 (실행 안 함)
foxmesh run "3D 프린팅용 10cm 컵"                 # 한 번 실행
foxmesh models                                  # 등록 모델과 실행 가능 여부
foxmesh bench [--model ID] [--repeats N]        # 모델별 성능 평가
```

주요 설정(환경 변수, 접두어 `FOXMESH_`):

| 변수 | 기본값 | 설명 |
| --- | --- | --- |
| `DATA_DIR` | `./var` | 작업 결과, 업로드, 통계, DB 저장 위치 |
| `DATABASE_URL` | `sqlite:///<DATA_DIR>/foxmesh.db` | 작업 저장소. `postgresql://...`(psycopg 필요) 또는 `memory://` |
| `RETENTION_DAYS` | `30` | 끝난 작업·결과 파일·안 쓰는 업로드 보존 기간 (0이면 무기한). 1시간마다 정리 |
| `DEDUP_TTL` | `3600` | 같은 요청에 기존 결과를 재사용하는 시간(초) |
| `WORKERS` | `2` | 동시에 실행할 작업 수 |
| `MAX_ATTEMPTS` / `TRANSIENT_RETRIES` | `3` / `1` | 작업당 최대 시도 수 / 일시 오류 시 같은 모델 재시도 수 |
| `BLENDER`, `BLENDER_MODE` | PATH의 `blender`, `blender` | Blender 실행 파일, 또는 `bpy` 모드의 Python |
| `VRAM_GB` | 자동 감지 | GPU를 감지하지 못할 때 쓸 VRAM 크기 |

pip의 `bpy` 모듈(Blender 4.2는 Python 3.11 필요)을 쓸 경우:
`FOXMESH_BLENDER=/path/to/python3.11 FOXMESH_BLENDER_MODE=bpy`.

### 프론트엔드

```bash
cd frontend
npm install
FOXMESH_API_URL=http://127.0.0.1:8000 npm run dev   # http://localhost:3000
```

`/api/*` 요청은 Next.js가 백엔드로 프록시합니다.

### 테스트

```bash
cd backend
pytest                                            # Blender 없이 전체 테스트
FOXMESH_TEST_BLENDER=/opt/blender/blender FOXMESH_TEST_BLENDER_MODE=blender pytest tests/test_blender.py
FOXMESH_TEST_DATABASE_URL=postgresql://postgres@localhost/foxmesh_test pytest tests/test_store.py
```

## 성능 최적화와 안정성

- **작업 저장과 복구**: 작업은 SQLite(기본) 또는 PostgreSQL에 저장되어 재시작 후에도 기록이 남습니다.
  정상 종료 시 실행 중이던 작업은 대기열로 돌아가 다음 시작 때 이어서 실행되고, 비정상 종료로 중단된 작업은
  한 번만 다시 실행합니다(같은 작업이 계속 서버를 죽이는 상황 방지).
- **중복 작업 방지와 결과 캐시**: 내용이 같은 요청(공백·대소문자 무시, 이미지는 파일 해시)은 진행 중이거나
  `DEDUP_TTL` 안에 성공한 작업을 재사용합니다. 재시작 후에도 적용되며, 결과 파일이 지워졌으면 다시 실행합니다.
  결과 파일과 썸네일은 바뀌지 않으므로 `Cache-Control: immutable`로 내려 브라우저가 다시 받지 않습니다.
- **보존 기간 정리**: `RETENTION_DAYS`가 지난 작업, 결과 파일, 쓰이지 않는 업로드를 1시간마다 삭제합니다.
  `POST /api/maintenance/cleanup`으로 즉시 실행할 수 있습니다.
- **모델별 성능 평가**: `foxmesh bench` 또는 실행 엔진 관리 페이지의 **벤치마크 실행**. 텍스트 4종·이미지 2종 테스트를
  모델마다 1회 시도, 대체 모델 없이 실행해 성공률, 평균/P90 생성 시간, 검증 점수를 측정합니다. 결과는 모델 통계에
  기록되어 자동 선택이 추정치 대신 실측값을 쓰게 됩니다. 모델이 지원하지 않는 작업은 건너뜁니다.
- **VRAM 기반 실행 제한**: 로컬 GPU 모델은 실행 동안 VRAM을 예약하고, 남은 VRAM이 부족하면 빌 때까지 기다립니다.
- **모델 프로세스 재사용**: 추론 서버는 모델을 메모리에 유지하고, VRAM 예산(`FOXMESH_INFERENCE_VRAM_GB`)을 넘으면
  가장 오래 안 쓴 모델부터 내립니다. `FOXMESH_INFERENCE_PRELOAD`로 시작 시 미리 올리고,
  `FOXMESH_INFERENCE_IDLE_S`가 지나도록 안 쓴 모델은 내립니다.
- **모니터링**: `GET /api/metrics`가 Prometheus 형식으로 작업 수, 대기열, 모델별 실행·실패·평균 시간·검증 점수,
  GPU 메모리를 내보냅니다.

## 이미지로 3D 생성

생성 페이지에서 이미지(PNG·JPG·WebP)를 올리면 작업이 `image_to_3d`로 분류되고, 이미지 입력을 받는 모델 중에서
자동 선택됩니다. 프롬프트는 없어도 됩니다.

GPU 모델이 연결되지 않은 기본 상태에서는 내장 **Image Relief** 모델(`adapters/image_relief.py`, CPU, 약 0.3~1초)이
실행됩니다.

1. 배경 제거: 투명 PNG는 알파 채널, 그 외에는 테두리 색을 배경으로 보고 색 거리 + Otsu 임계값으로 분리.
   배경이 단색이 아니면 이미지 전체를 사용(부조 모드)
2. 마스크 정리: 가장 큰 덩어리만 남기고 구멍을 메우고, 꼭짓점으로만 닿는 칸을 없애 메시가 항상 닫히도록 처리
3. 형태: 설명의 단어로 선택
   - **부풀리기** (기본, `인형`·`통통`): 윤곽까지의 거리로 둥근 단면을 만든 입체
   - **부조** (`부조`·`높이맵`·`리토판`): 뒷면은 평평하고 앞면은 이미지 밝기로 높낮이
   - **돌출** (`로고`·`키링`·`간판`·`평면`): 일정 두께의 판
4. 윤곽선을 매끄럽게 다듬고, 이미지를 앞·뒷면 텍스처로 입혀 GLB(텍스처 포함) 또는 OBJ+MTL+PNG로 출력
5. 폴리곤 예산(게임용 기본 2만)과 크기(예: `5cm`)를 반영하고, 이후 일반 작업처럼 Blender 후처리와 검증을 거침

이 방식은 윤곽 기반 2.5D 모델이라 보이지 않는 뒷면을 추정하지 않습니다. 실제 3D 복원이 필요하면 TripoSR·TRELLIS·
Hunyuan3D 같은 이미지→3D 모델을 아래 방법으로 연결하세요. 정확도 점수가 더 높아 연결되면 자동으로 우선 선택됩니다.

## 실제 AI 모델 연결

기본 상태에서는 내장 **절차적 생성기**(키워드로 검·테이블·의자·나무·집 등 기본 형상 조립)와 내장 **Image Relief**
(이미지→2.5D)만 활성화되어 있어 GPU 없이 전체 흐름을 확인할 수 있습니다. Shap-E, TripoSR, TRELLIS, Hunyuan3D 2.0 설명 파일이 함께 등록되어
있지만 비활성 상태입니다. 이 파일들의 실행 시간·VRAM·품질 값은 **초기 추정치**이며, 실제 실행 로그가 쌓이면
라우터는 실측값을 사용합니다. 라이선스는 사용 전에 반드시 직접 확인하세요.

1. GPU 머신에서 추론 서버를 실행합니다. 모델별로 `ModelRunner`를 구현해 등록합니다
   (`backend/foxmesh/inference/server.py` 문서 참고).
   ```bash
   FOXMESH_INFERENCE_RUNNERS=mypkg.runners:TripoSRRunner FOXMESH_INFERENCE_VRAM_GB=24 \
     uvicorn foxmesh.inference.server:app --host 0.0.0.0 --port 9000
   ```
2. 실행 엔진 관리 페이지에서 해당 모델을 활성화하고 추론 서버 주소를 입력합니다
   (또는 `PATCH /api/models/triposr {"enabled": true, "config": {"endpoint": "http://gpu:9000"}}`).
3. 새 모델은 JSON 설명 파일을 `FOXMESH_MODELS_DIR` 디렉터리에 두거나 `POST /api/models`로 등록합니다.
   새로운 실행 방식은 `GenerationAdapter`를 구현해 `foxmesh.adapters` entry point로 배포하면 됩니다.

## API 요약

| 메서드 | 경로 | 설명 |
| --- | --- | --- |
| `POST` | `/api/analyze` | 요청 분석과 모델 선택 결과만 반환 (실행 안 함) |
| `POST` | `/api/jobs` | 작업 등록. 같은 요청이 진행 중이거나 최근 성공했으면 기존 작업 반환 |
| `GET` | `/api/jobs`, `/api/jobs/{id}` | 작업 목록 / 상세(진행 단계, 선택 근거, 시도 기록, 검증 결과) |
| `POST` | `/api/jobs/{id}/cancel` | 작업 취소 |
| `GET` | `/api/jobs/{id}/files/{name}` | 결과 파일 다운로드 |
| `PUT` | `/api/uploads?filename=` | 이미지·3D 파일 업로드 (본문에 파일 바이트, 이미지는 디코딩 검사) |
| `GET` | `/api/uploads/{id}` | 업로드한 이미지 보기 (미리보기용) |
| `GET/POST/PATCH` | `/api/models[/{id}]` | 모델 목록·등록·활성화·버전·설정 변경 |
| `GET/PUT/DELETE` | `/api/policies[/{profile}]` | 자동 선택 가중치 조회·변경·초기화 |
| `GET` | `/api/system`, `/api/dashboard` | GPU·Blender·대기열 상태, 대시보드 요약 |
| `POST` / `GET` | `/api/benchmarks[/{id}]` | 모델 벤치마크 실행 / 결과 조회 (`/api/benchmarks/cases`: 테스트 목록) |
| `GET` | `/api/metrics` | Prometheus 지표 |
| `POST` | `/api/maintenance/cleanup` | 보존 기간 정리 즉시 실행 |

## 개발 단계 진행 현황

- [x] 1단계 자동 선택 엔진: 작업 분류, 모델 레지스트리, 후보 필터링, 점수 기반 선택, 실행 로그(`var/runs.jsonl`)
- [x] 2단계 Blender 연동: 백그라운드 실행, 자동 불러오기, 메시·재질 검사, BLEND/GLB 출력, 썸네일 렌더링
- [x] 3단계 웹 통합: 생성 페이지, 작업 대기열, 진행 상태, 결과 미리보기, 실행 엔진 관리
- [x] 4단계 성능 최적화: VRAM 기반 실행 제한(대기 포함), 모델 프로세스 재사용(미리 올리기·유휴 언로드),
  캐싱과 중복 작업 방지(재시작 후 유지, 보존 기간 정리), 모델별 성능 평가(벤치마크), 오류 복구(작업 저장·재시작
  복구, 차단기)와 대체 실행. 여러 서버가 대기열을 나눠 쓰는 Redis 기반 분산 큐는 아직 없음(현재는 서버 하나에서 처리)
- [ ] 5단계 자체 AI 개발: 자체 모델은 `ModelRunner` + JSON 설명 파일로 등록하면 자동 선택 대상에 포함됨
