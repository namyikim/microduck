# 천천히 앉았다 일어서기 / Gentle sit and stand

## 한국어

다음 목표는 **서기 → 약 2초에 걸쳐 앉기 → 앉아서 유지 → 다시 서서 유지**입니다. 완료된 앉은 인사와는 별도 정책입니다. 우선 전환 동작을 검증하고 나중에 인사와 연결합니다.

기존 `Mjlab-SitStand-Flat-MicroDuck` 환경의 양방향 자세 명령과 2초 목표 높이 전환을 사용합니다. 인사·JumpSpin 모델을 불러오지 않고 처음부터 학습합니다. 기본 예산은 **4,096환경·6,000회**이며 성공 여부는 최종 영상과 지표로 판단합니다.

### 실행

Colab A100에서 저장소 설치와 Drive 연결을 마친 뒤 실행합니다. CLI 런타임은 항상 `colab run --keep`으로 만듭니다.

```bash
uv run python scripts/train_sitstand_colab.py --start-training
```

`--start-training`을 빼면 안내만 출력합니다. 실행 순서는 다음과 같습니다.

1. 명목 물성에서 서기·앉기 자세를 각각 64환경에서 3초 유지해 높이·기울기·머리 접촉 검사. 초기 관절·높이·기울기에 작은 오차를 줍니다.
2. 64환경·5회 GPU 시험 학습 및 공식 ONNX 내보내기·추론 검사.
3. 시험 모델로 영상 2개를 만들어 반복 reset과 영상 저장 경로 검사. 이 초기 모델의 동작 성공은 요구하지 않습니다.
4. 4,096환경 본학습. 같은 실험의 체크포인트가 있으면 남은 횟수만 재개.
5. 최종 모델의 공식 ONNX와 16초 평가 영상 5개·JSON 생성.

본학습 전에 실패하면 `status.json`과 `pipeline.log`를 확인합니다. 새 실행은 평가 폴더를 따로 만들며 기존 결과를 덮어쓰지 않습니다. 이미 다른 학습이 실행 중이라면 중복 실행하지 마세요.

### 물리 사전 검사에서 확인한 점

HOME 제어값 그대로는 명목 물성에서도 서기 유지에 실패했습니다. 좌우 대칭 제어값을 탐색해, 서기 검사에만 왼쪽 hip_pitch +0.12rad / 오른쪽 -0.12rad 제어 보정을 사용했습니다. 관측의 HOME 기준, 학습 보상, 정책 출력에는 이 보정을 넣지 않습니다.

명목 물성·초기 관절 ±0.01rad·기울기 ±0.5°·높이 11.4~11.6cm에서 64/64가 서기를 유지했습니다. 실제 높이는 11.37~11.49cm, 최대 기울기는 3.58°였습니다. 앉기도 64/64 통과했습니다. 물성 무작위화와 더 큰 초기 오차를 함께 준 고정 제어 시험은 48/64만 통과했으므로, 사전 검사는 **평형 가능성** 검사로 한정합니다. 물성 변화에 대한 견고함은 학습된 피드백 정책으로 별도 판단해야 합니다.

### 평가 기준

영상은 0~2초 서기, 2~8초 앉기 명령, 8~16초 서기 명령으로 구성합니다. 머리 명령은 중립이며 외부 밀기·머리 충격 이벤트를 끈 기본 동작 검사입니다.

- 정지 구간 높이: 서기 11.5cm, 앉기 6cm에서 각각 ±1.5cm 이내.
- 정지 구간 기울기 15° 이내, 전환 중 45° 이내, 머리 바닥 접촉 없음.
- 명령 후 목표 높이 구간 도달 시간은 각각 1~4초. 즉시 주저앉거나 움직이지 않으면 실패.
- 전환 중 하강 0.10m/s·상승 0.16m/s 초과를 실패 처리. 잠시 기다리다 갑자기 내려앉는 동작도 걸러냅니다.
- 전체 구간을 빠짐없이 기록하고 NaN, 조기 종료를 실패 처리.

기존 학습 환경은 전환 중 머리를 지지점으로 쓰는 것을 허용합니다. **이번 평가에서는 머리 접촉을 실패로 표시**하므로 학습을 마쳐도 이 기준을 통과하지 못할 수 있습니다. 그 경우 실제 체크포인트의 영상과 접촉 지표를 보고 후속 조정을 결정합니다. 실제 로봇이 없으므로 하드웨어 검증은 하지 않습니다.

### 결과와 재개

Drive 기준 폴더: `MyDrive/microduck-training/sitstand_v1/`

| 파일 | 내용 |
|---|---|
| `status.json` | 현재 단계, 실패 원인, 평가 폴더 |
| `pipeline.log` | 이번 CLI 실행 래퍼가 저장하는 전체 로그. 위 명령을 직접 실행하면 터미널에 출력됩니다. |
| `training.log` | 본학습 iteration, 보상, ETA |
| `logs/rsl_rl/microduck_sitstand/<run>/model_*.pt` | 재개용 체크포인트 |
| `evaluations/<UTC 시각>/preflight.json` | 목표 자세 물리 검사 |
| 같은 폴더의 `policy.onnx` | 관측 정규화를 포함한 공식 내보내기 |
| 같은 폴더의 `sitstand_trial_01.mp4` ~ `05.mp4` | 최종 평가 영상 |
| 같은 폴더의 `evaluation_summary.json`, `*_trace.json` | 판정과 시계열 근거 |

세션이 끊기면 Drive를 다시 연결하고 같은 명령을 실행합니다. 다른 과제 모델에서 재개하지 않습니다. 종료 시 런타임을 자동 삭제하지 않으며 자동 점검 예약도 새로 만들지 않습니다.

## English

The next target is **stand → sit slowly over roughly two seconds → hold → stand and hold**. This is a separate policy from the completed seated greeting; connecting them comes after transition validation.

The existing `Mjlab-SitStand-Flat-MicroDuck` task trains both directions with a two-second target ramp. Training starts from scratch, without greeting or JumpSpin weights. The default budget is 4,096 environments and 6,000 iterations.

After A100 setup and Drive mounting, run `uv run python scripts/train_sitstand_colab.py --start-training`. Always create CLI runtimes with `colab run --keep`. The pipeline checks both rest poses in 64 environments, runs five smoke iterations and official ONNX validation, records two smoke videos to exercise repeated resets, then trains and generates five final videos plus ONNX. Smoke-policy motion quality is not a prerequisite; successful execution is.

An open-loop HOME hold failed even with nominal dynamics. A symmetric hip control offset (+0.12rad left, −0.12rad right) produced a stable equilibrium in 64/64 starts with ±0.01rad joint noise, ±0.5° tilt and 11.4–11.6cm initial height. Measured standing height was 11.37–11.49cm and maximum tilt 3.58°. This offset is used only for the preflight hold; HOME observations, training rewards and policy actions are unchanged. Fixed control passed only 48/64 trials under combined dynamics variation and larger initialization errors, so preflight establishes nominal feasibility, not robustness of the future policy.

The 16-second evaluation commands stand until 2s, sit until 8s, then stand until 16s. It requires stable endpoint heights (11.5/6cm ±1.5cm), endpoint tilt below 15°, transition tilt below 45°, no head contact and arrival within 1–4 seconds. It also rejects descent above 0.10m/s or ascent above 0.16m/s, including waiting before a sudden drop, as well as missing/non-finite evidence and incomplete cycles. Head commands are neutral and external pushes are disabled for this basic assessment.

The inherited training recipe permits head support during transitions; this evaluation deliberately flags that behavior as a failure. Finishing the budget does not guarantee passing. Review the actual videos and measurements before changing the recipe. No physical robot testing is performed.

The CLI launcher for this run saves `pipeline.log`; direct invocation prints that output to the terminal. Results are under `MyDrive/microduck-training/sitstand_v1/`: `status.json`, `pipeline.log`, `training.log`, checkpoints in `logs/rsl_rl/microduck_sitstand/`, and videos/ONNX/JSON in `evaluations/<UTC timestamp>/`. Rerun the same command after mounting Drive to resume only this experiment. No automatic runtime shutdown or new monitoring schedule is created.
