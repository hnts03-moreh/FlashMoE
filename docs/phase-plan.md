# FlashMoE ROCm Porting — Phase Plan

## Phase 1: 코드 포팅 (완료)

**목표**: CUDA 코드를 ROCm/HIP 듀얼 플랫폼으로 포팅

**완료 항목**:
- [x] 추상화 레이어 헤더 8개 (`platform/*.h`)
- [x] rocBLASDx 서브프로젝트 7개 파일 (MFMA 기반 header-only GEMM)
- [x] 커널 헤더 37개 `.cuh` 포팅 (`#ifdef` 분기)
- [x] 테스트 파일 10개 포팅
- [x] Python 레이어 7개 파일 포팅
- [x] 빌드 시스템 4개 파일 수정 (CMake, pyproject.toml)
- [x] 문서 5개 생성

**담당**: platform-architect → rocblasdx-developer → hip-kernel-porter + comm-porter + python-porter

**커밋**: `b78bc04` Port FlashMoE to ROCm/HIP with dual-platform abstraction layer

---

## Phase 2: 빌드 검증 (완료)

**목표**: 모든 HIP 테스트 타겟 컴파일 성공

**완료 항목**:
- [x] 8개 HIP 테스트 타겟 빌드 성공 (MI300X, ROCm 7.0, hipcc Clang 20)
  - testScheduler, testGEMM, gemmMNK, testGatedGEMM, testGate, testCombine, testFlashMoE, playground
- [x] testScheduler 데드락 수정
- [x] testFlashMoE E2E HIP stub 활성화

**담당**: qa-validator

**커밋**: `83b4547`, `fa6e9d3`, `e2aa18a`

---

## Phase 3: 기능 검증 (현재)

**목표**: MI300X 하드웨어에서 런타임 테스트 통과, 수치 정확도 확인

**담당**: qa-validator + rocblasdx-developer + hip-kernel-porter

### 체크리스트

#### 3.1 테스트 바이너리 런타임 실행
- [x] `playground` — GPU 없이 기본 동작 확인
- [x] `testScheduler` — 동시성 정확성 (데드락 없음)
- [x] `testGEMM` — MFMA GEMM 실행 및 결과 출력
- [x] `gemmMNK` — 다양한 M/N/K 조합 GEMM
- [x] `testGatedGEMM` — Gated MLP GEMM
- [x] `testGate` — softmax 수치 안정성
- [x] `testCombine` — 토큰 결합 가중치

#### 3.2 MFMA 정확도 검증
- [x] rocBLASDx MFMA 실행 결과 vs naive GEMM 비교
- [ ] FP16: 상대 오차 < 1e-2 (16/32/64 tiles 0% error, 128+ WIP)
- [ ] FP32: 상대 오차 < 1e-5
- [ ] BF16: 상대 오차 < 1e-2
- [x] MFMA 명령어별 정확도 (V_MFMA_F32_16X16X16F16 ✓, V_MFMA_F32_32X32X8F16 ✓ for single-tile)

#### 3.3 수치 검증 파이프라인
- [x] CPU 호스트 사이드 GEMM 레퍼런스 구현 (host_reference.cuh)
- [ ] PyTorch ROCm 비교 스크립트 (선택)
- [ ] 자동화된 정확도 리포트 생성

#### 3.4 발견된 버그 및 수정사항
- [x] MFMA output register layout → copy_fragment mapping 불일치 수정
- [x] SmemTensor LDS bank-conflict padding 미반영 SharedSizeAB 수정
- [x] A/B 글로벌 메모리 로딩 시 SmemLayout padding stride 오적용 수정
- [ ] tiles_per_wave > 1 (bM=128+) 시 multi-tile MFMA 결과 매핑 오류 (WIP)

### 완료 기준
- 7개 단일 GPU 테스트 바이너리 정상 실행 (crash/hang 없음)
- MFMA GEMM 정확도가 epsilon 이내

---

## Phase 4: 통합 검증 (예정)

**목표**: ROCSHMEM 통합, 분산 멀티GPU E2E 테스트

**담당**: comm-porter + python-porter + qa-validator

### 체크리스트
- [ ] ROCSHMEM 빌드 (`scripts/build_rocshmem.sh`)
- [ ] ROCSHMEM 링크된 testFlashMoE 빌드
- [ ] 디바이스 사이드 SHMEM API 동작 확인 (`rocshmem_wg_init` 등)
- [ ] 2+ GPU 분산 testFlashMoE 실행
- [ ] ROCSHMEM Python 바인딩 (pybind11) 개발
- [ ] `cb.py` ROCSHMEM 경로 런타임 검증
- [ ] `flashmoe/__init__.py` 분산 초기화 검증

### 완료 기준
- 분산 E2E 테스트가 2+ GPU에서 정상 실행
- Python에서 ROCSHMEM init/barrier/finalize 동작

---

## Phase 5: 성능 최적화 (예정)

**목표**: MI300X 성능 극대화

**담당**: perf-profiler + rocblasdx-developer + hip-kernel-porter

### 체크리스트
- [ ] rocprof 기반 커널 프로파일링 (HW 카운터, 타임라인)
- [ ] omniperf 심층 분석 (VALU/MFMA 파이프 점유율, LDS 사용률)
- [ ] 성능 병목 식별 및 우선순위 지정
- [ ] MFMA 소프트웨어 파이프라이닝 (GEMM 실행 루프)
- [ ] LDS 이중 버퍼링 (`tile.cuh` 동기식 메인루프 대체)
- [ ] `buffer_load` 프리페칭
- [ ] 벡터화 atomicAdd (`infra/rvt.cuh`)
- [ ] shared memory 뱅크 충돌 최적화 (AMD 32-bank)

### 완료 기준
- CUDA 원본 대비 80%+ 성능 달성 (GEMM TFLOPS 기준)
- 성능 회귀 테스트 기준선 수립

---

## Phase 6: 확장 (예정)

**목표**: 추가 아키텍처 및 기능 지원

**담당**: platform-architect + hip-kernel-porter

### 체크리스트
- [ ] MI250 (gfx90a) MFMA 명령어 선택 및 빌드 분기
- [ ] CDNA4 아키텍처 감지 및 MFMA 래퍼
- [ ] hipGraph 벤치마킹 포팅
- [ ] `experimental/topo.cuh` ROCSHMEM `_wg`/`_wave` 변환
- [ ] composable_kernel 통합 평가

### 완료 기준
- MI250에서 빌드 + 기본 테스트 통과
- hipGraph 캡처 기반 벤치마킹 동작
