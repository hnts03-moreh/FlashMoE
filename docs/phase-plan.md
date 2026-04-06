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

## Phase 3: 기능 검증 (완료)

**목표**: MI300X 하드웨어에서 런타임 테스트 통과, 수치 정확도 확인

**담당**: qa-validator + rocblasdx-developer + hip-kernel-porter

**커밋**: `818472b` (최종), `29377f8`, `a0f701d`, `4471082`

### 체크리스트

#### 3.1 테스트 바이너리 런타임 실행
- [x] `playground` — GPU 없이 기본 동작 확인
- [x] `testScheduler` — 동시성 정확성 (0% error, 데드락 없음)
- [x] `testGEMM` — MFMA GEMM 실행 (FP16, CPU ref 비교)
- [x] `gemmMNK` — 36/36 configurations 0% error (M=32~1024, N=128/256, K=64~256)
- [x] `testGatedGEMM` — Gated MLP GEMM (실행 확인, 대형 사이즈 시 CPU ref 느림)
- [x] `testGate` — softmax 수치 안정성 (E=8~256, 정상 실행)
- [x] `testCombine` — 토큰 결합 가중치 (정상 실행)
- [x] `testPrecision` — FP16/BF16/FP32 30/30 PASS (신규)

#### 3.2 MFMA 정확도 검증
- [x] rocBLASDx MFMA 실행 결과 vs naive GEMM 비교
- [x] FP16: 10/10 PASS, 상대 오차 0%, max_abs_err ≤ 7.8e-3 (16~512 tiles)
- [x] FP32: 10/10 PASS, 상대 오차 0%, max_abs_err ≤ 4.8e-6 (bK=32, LDS 최적화)
- [x] BF16: 10/10 PASS, 상대 오차 0%, max_abs_err ≤ 2.0e-3 (16~512 tiles)
- [x] MFMA 명령어별 정확도: F16 16x16x16/32x32x8 ✓, BF16 16x16x16/32x32x8 ✓, F32 16x16x4/32x32x2 ✓

#### 3.3 수치 검증 파이프라인
- [x] CPU 호스트 사이드 GEMM 레퍼런스 구현 (host_reference.cuh)
- [x] 자동화된 정확도 리포트 생성 (testPrecision CSV 출력 + phase3_validate.sh)
- [ ] PyTorch ROCm 비교 스크립트 (선택 — 불필요, CPU ref로 충분)

#### 3.4 발견된 버그 및 수정사항
- [x] MFMA output register layout → copy_fragment mapping 불일치 수정
- [x] SmemTensor LDS bank-conflict padding 미반영 SharedSizeAB 수정
- [x] A/B 글로벌 메모리 로딩 시 SmemLayout padding stride 오적용 수정
- [x] tiles_per_wave > 1 (bM=128+) multi-tile MFMA — 검증 완료 (0% error)
- [x] FP32 MFMA scalar dispatch: lane group K-offset 누락 수정 (gemm.hpp)
- [x] tfloat32_t explicit 변환: implicit으로 변경 (types.hpp)
- [x] FP32 LDS 초과: 128x128 타일에서 bK=64→32 (float 4B × 2 = FP16 대비 2배)

### 완료 기준
- [x] 7개 단일 GPU 테스트 바이너리 정상 실행 (crash/hang 없음)
- [x] MFMA GEMM 정확도가 epsilon 이내 (FP16/BF16/FP32 30/30 PASS)

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
