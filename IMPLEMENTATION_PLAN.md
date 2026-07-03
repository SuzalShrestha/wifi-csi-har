# WiFi CSI Human Activity Recognition — Implementation Plan

**Project:** WiFi CSI HAR using Deep Neural Networks (EX 707, Tribhuvan University)
**Hardware:** 3× ESP32-S3 N16R8 (RX) + commodity WiFi router (TX)
**Team:** 4 members · **Timeline:** 8 months (Asar → Chaitra, defense ~March–April 2027)
**Target:** ≥85% accuracy on 5 classes: walking, sitting, standing, lying down, falling

---

## 0. Fact-Checked Corrections to the Proposal

These are verified against current tooling/literature (July 2026). Fix these before writing any code.

### 0.1 Use `espressif/esp-csi`, NOT the original ESP32-CSI-Tool
Your proposal names **ESP32-CSI-Tool** (Hernandez & Bulut). That project targets **ESP-IDF v4.3 (2021-era)** and was written for the original ESP32. It can be made to build for S3 via forks (e.g., HKU's COMP3516 fork), but the **officially maintained path is Espressif's own [esp-csi](https://github.com/espressif/esp-csi) repository**, which:
- Explicitly supports ESP32-S3 (all series: ESP32/S2/C3/S3/C5/C6/C61)
- Ships a ready-made example for exactly your topology: **`examples/get-started/csi_recv_router`** — the ESP32 pings the router at a configurable rate and captures CSI from each reply
- Outputs CSV rows over USB-serial: metadata (seq, MAC, RSSI, MCS, bandwidth, timestamp) + a 128-value array of **int8 (real, imag) pairs = 64 complex subcarrier values** (20 MHz LLTF; ~52 carry useful information)
- Includes `console_test` (interactive capture + built-in visualization tool) and `wifi_sensing_demo`

**Action:** cite ESP32-CSI-Tool in the literature review as prior art; build on `esp-csi` in the implementation. Note for the report: Espressif itself ranks CSI quality **C5 > C6 > C3 ≈ S3 > ESP32** — the S3 is a fine mid-tier choice and better than the classic ESP32 used in most published ESP32-HAR papers.

### 0.2 Amplitude-first; treat phase as a stretch goal
The ESP32-S3 has a **single antenna** and a consumer-grade oscillator. Raw phase is corrupted by CFO/SFO/PLL offsets, and the standard multi-antenna phase-difference sanitization used with Intel 5300 / Atheros cards is **impossible on a single-antenna device**. Nearly all published ESP32 HAR work (including Moshiri et al., whose dataset your reference [7] describes) uses **amplitude only**. Plan the entire main pipeline on amplitude; only explore linear-detrended phase later if time allows. Don't promise phase features in the report.

### 0.3 Battery + USB contradiction
The proposal lists battery packs *and* USB streaming to a host. During **dataset collection you need USB-serial anyway** (115200 is too slow; use **921600 baud or higher** to sustain 100 pkt/s × 3 devices). So for data collection: 3 USB cables into one powered hub into the laptop — batteries are irrelevant there. Batteries matter only for the **final live demo**, where each ESP32 can stream CSI/features over **UDP to the host** instead of USB (acceptable at low rates; adds some channel traffic — keep the rate modest and constant so training and demo distributions match). Decide early which demo mode you want and collect at least some training data in that same mode.

### 0.4 Evaluation protocol — the single most common way these projects overclaim
Random shuffling of windows into train/test **leaks** temporally adjacent, nearly identical windows across the split and produces inflated numbers (95–99%) that collapse in the demo. Published ESP32 results (e.g., LSTM >93% with 5-fold CV) are mostly random-split numbers. Commit now to reporting **three splits**, all of them in the report:
1. **Random window split** (comparable to literature — will look great)
2. **Cross-session split** (train on sessions 1–3, test on session 4, same people)
3. **Cross-subject split** (leave-one-person-out — the honest number; expect a visible drop)

Your ≥85% target is realistic for splits 1–2; for split 3 treat 70–80% as a good result and frame it as the generalization analysis chapter. This framing turns a "failure" into scientific content.

### 0.5 Windows/laptop note
Three simultaneous 921600-baud serial streams are fine on Linux/macOS; on Windows use one process per port. Timestamp on the host at packet arrival and align the three streams by **window (e.g., 2–4 s), not per-packet** — the proposal already says this correctly. Keep it.

---

## 1. System Architecture (final)

```
WiFi Router (TX, fixed channel, e.g. ch 6, 20 MHz, 2.4 GHz)
     ▲ ping replies (ICMP), ~100 pkt/s per receiver
     │
 ┌───┴────────┬────────────┐
ESP32-S3 #1  ESP32-S3 #2  ESP32-S3 #3     (esp-csi csi_recv_router firmware)
 │ USB-serial 921600 baud (collection) / UDP (demo)
 └───────┬────┴────────────┘
      Powered USB hub
         │
      Host laptop
   ├─ collector.py  (3 async serial readers → timestamped Parquet/CSV per device)
   ├─ preprocess    (subcarrier selection → Hampel → lowpass/DWT → normalize → window)
   ├─ models        (PyTorch: CNN, CNN-LSTM; baselines from SenseFi)
   └─ realtime demo (sliding window → model → dashboard)
```

Key parameters (defaults; tune in Phase 2):
- **Packet rate:** 100 Hz per receiver (esp-csi default territory; enough for activity Doppler which is <60 Hz for human motion)
- **Window:** 3 s (300 packets) with 50% overlap → input tensor ≈ `3 × 52 × 300` (RX × subcarriers × time)
- **Router:** lock to one 2.4 GHz channel, 20 MHz bandwidth, disable band steering/smart connect; ideally a dedicated router with no other clients
- **Placement:** the three RX at different positions/heights around the sensing area (e.g., 3 corners of the room, ~1 m height) so their multipath views are complementary; tripods keep geometry reproducible across sessions — photograph and measure the layout, record it in metadata

---

## 2. Phase-by-Phase Plan (8 months)

### Phase 1 — Bring-up & tracer bullet (Weeks 1–4, Asar/Shrawan)
Goal: **one end-to-end sample flows from radio to a plotted spectrogram** before any dataset work.

1. Install ESP-IDF (v5.x, latest stable) + clone `esp-csi`; flash `csi_recv_router` on one S3.
2. Confirm CSI CSV rows arrive over serial; parse the 128-int8 array into 64 complex values; drop null/guard subcarriers → ~52 usable.
3. Write `collector.py`: async serial read, host timestamping, rotation into per-session Parquet files with a metadata sidecar (JSON: device ID, position, channel, rate, subject, activity, session).
4. Replicate to all 3 devices simultaneously via powered hub; verify sustained 100 Hz per device without drops (log sequence-number gaps; expect and record ~1–5% loss — resample windows to a fixed grid).
5. Sanity experiments: empty room vs. person walking — plot amplitude heatmap (subcarrier × time). You must *see* the disturbance by eye. If you can't, fix placement/rate before proceeding.
6. Baseline latency test of the pipeline end to end.

**Milestone M1:** live heatmap visibly reacting to a person walking, from all 3 receivers at once. *(This de-risks the entire project in month 1.)*

### Phase 2 — Preprocessing pipeline + pilot dataset (Weeks 5–8, Shrawan/Bhadra)
1. Implement preprocessing as a pure, testable Python module (unit-test each stage):
   - Subcarrier selection (drop nulls/pilots or keep pilots — decide empirically)
   - **Hampel filter** (outlier spikes) → **Butterworth low-pass** (~30–60 Hz cutoff is moot at 100 Hz sampling; instead cut <0.1 Hz drift with high-pass or subtract moving mean) and/or **DWT denoising**
   - Per-subcarrier standardization computed **on train split only**
   - Windowing (3 s, 50% overlap) + window-level alignment of the 3 receiver streams (tolerance ±100 ms; drop windows missing any receiver)
2. Build a **labeling harness**: operator presses a key / phone webapp to mark activity start/stop; labels recorded on the same host clock. Scripted collection sessions ("walk 30 s, sit 30 s, …") beat manual annotation.
3. **Pilot dataset:** 2 team members × all 5 activities × ~5 min each, single room, 2 sessions on different days.
4. Train a quick CNN on the pilot (even 80% on random split is fine). Purpose: expose pipeline bugs, class confusions (sitting vs lying are hard), and window-size sensitivity **before** the big collection effort.

**Milestone M2:** pilot model beats random by a wide margin; preprocessing choices frozen and documented.

### Phase 3 — Full dataset collection (Weeks 9–14, Bhadra/Ashwin)
This is the project's real asset — a locally collected multi-receiver ESP32-S3 dataset is itself a contribution (compare: Moshiri's [CSI-HAR dataset](https://github.com/parisafm/CSI-HAR-Dataset), the 2026 ESP-Fi HAR dataset).

Protocol (write it as a formal document first — the report needs it anyway):
- **Subjects:** all 4 team members + recruit 4–6 more (different heights/builds) → 8–10 subjects for a defensible cross-subject evaluation
- **Per subject per activity:** ≥ 5 min of clean data → ≥100 windows/class/subject → **≥25,000 windows total** across classes and subjects
- **Falls:** onto a mattress, multiple fall directions (forward/backward/sideways), spotter present; expect fewer fall samples — plan class weighting or oversampling
- **Transitions** (sit-down, stand-up, lie-down) — decide whether they're separate classes or excluded; excluding is fine, but be consistent
- **Sessions:** ≥3 per subject on different days/times (furniture micro-changes, clothing, router thermal drift all matter)
- **Environments:** primary room + **one second room** for a small transfer test set (even 30 min there gives you a cross-environment result for the report)
- Record **empty-room / no-activity** as a 6th "background" class — critical for a demo that doesn't hallucinate activities in an empty room
- Every session: fixed device layout (tripods + floor markings + photos), metadata JSON, immediate post-session validation script (packet loss %, per-device counts, quick heatmap render) so a bad session is discovered same-day, not at training time

**Milestone M3:** frozen dataset v1.0 with datasheet (counts per class/subject/session/environment), backed up in ≥2 places (Git LFS / Drive / external disk).

### Phase 4 — Models & experiments (Weeks 15–20, Kartik/Mangsir)
Use [SenseFi](https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark) (PyTorch benchmark library; Patterns 2023) as scaffolding for model implementations and training loops — adapt its models to your `3 × 52 × 300` tensor instead of reinventing.

Experiment ladder (in order; stop adding models when returns diminish):
1. **Classical baseline** (mandatory for the report): SVM / random forest on hand features (per-subcarrier mean, std, MAD, percentiles, spectral energy bands). If DL can't beat this, your DL pipeline is broken.
2. **CNN** (proposal model 1): 2D conv over (subcarrier × time) per receiver, receivers as channels — 4–6 conv blocks, global pooling, FC head.
3. **CNN-LSTM** (proposal model 2): CNN encoder per time-chunk → BiLSTM → attention/last-state → FC.
4. **Stretch:** lightweight transformer encoder or GRU-attention (cheap once SenseFi scaffolding exists) — gives the report a nice 4-way comparison.

Rules that keep results honest and the report strong:
- All three splits from §0.4 for every model; fixed seeds; log everything (Weights & Biases free tier or CSV + git-committed configs)
- **Ablations** (these are your "analysis" chapter): 1 vs 2 vs 3 receivers (directly tests the proposal's multi-RX claim), window length (1/2/3/5 s), packet rate (25/50/100 Hz by subsampling), amplitude-only vs amplitude+detrended-phase
- Metrics: macro-F1 + per-class precision/recall + confusion matrix; for **falling**, report recall separately (a missed fall is the costly error — frame it as the safety-critical class)
- Early-stop on validation from the *same* split regime being tested (no peeking across regimes)

**Milestone M4:** results table complete; best model chosen; ≥85% on splits 1–2 achieved or the gap understood and explained.

### Phase 5 — Real-time system integration (Weeks 21–26, Poush/Magh)
1. Streaming inference: ring buffer per receiver → aligned sliding window (1 s hop) → preprocessing → model → prediction
2. **Temporal smoothing:** majority vote / exponential smoothing over last 3–5 predictions + confidence threshold → "unknown" state (prevents flickering output — reviewers always poke at this)
3. Dashboard: simple web UI (Flask/FastAPI + websocket) showing live heatmap, predicted activity, confidence, fall alert banner
4. Decide demo transport: USB (safest) or UDP-over-WiFi (wireless demo; collect a small adaptation dataset in this mode and fine-tune if the distribution shifted)
5. **Stretch goal with high defense value:** quantize the CNN (int8, TFLite Micro / esp-tflite-micro) and run inference **on the ESP32-S3 itself** — the N16R8's 8 MB PSRAM makes a small CNN entirely feasible. Even a 3-class on-device demo is a headline result for a Nepali-hardware-constraints narrative.
6. Measure and report: end-to-end latency, sustained throughput, detection delay after activity onset.

**Milestone M5:** 10-minute unrehearsed live demo passes with a person the model has never seen (use a friend, not a team member).

### Phase 6 — Evaluation hardening, report, defense (Weeks 27–32, Magh/Falgun/Chaitra)
1. Final evaluation runs on frozen code + frozen dataset; regenerate every figure from scripts (`make figures` — no hand-made plots)
2. Failure analysis chapter: confusion pairs (sit vs lie), cross-subject drop, cross-environment drop, packet-loss sensitivity
3. Report writing (LaTeX, start the skeleton in Phase 4 — methodology and dataset chapters can be written early)
4. Release: GitHub repo (firmware config + collector + preprocessing + models + trained weights) and, if the department permits, publish the dataset — instant differentiation for the defense and a citable artifact
5. Defense prep: rehearse the live demo 3×, prepare a **recorded backup video** of the demo (WiFi at the defense venue WILL behave differently — channel congestion, different multipath; never rely on live-only)

**Milestone M6:** report submitted; demo + backup video ready.

---

## 3. Work Split for 4 People

Parallel tracks after Phase 1 (everyone does Phase 1 together to learn the stack):

| Track | Owner | Scope |
|---|---|---|
| Firmware & acquisition | Member A | esp-csi config, collector.py, session tooling, packet-loss monitoring |
| Signal processing & dataset | Member B | preprocessing module + tests, labeling harness, collection protocol, dataset QA |
| Models & experiments | Member C | SenseFi adaptation, training infra, ablations, results tracking |
| Integration, demo & report | Member D | realtime pipeline, dashboard, report skeleton, figures pipeline |

Everyone participates in data collection (you need bodies performing activities anyway). Rotate the "subject" role. Weekly 1-hour sync with a shared task board; the Gantt in the proposal maps cleanly onto Phases 1–6.

---

## 4. Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| Cross-subject accuracy well below 85% | High | Report all 3 splits (§0.4); more subjects; data augmentation (time-warp, subcarrier dropout, amplitude jitter); frame as generalization analysis |
| Channel interference (campus WiFi congestion on 2.4 GHz) | High | Survey channels first (`iwlist`/WiFi analyzer), pick the least congested, collect at consistent times, record RSSI/noise-floor metadata per session |
| Serial drops at 3×100 Hz | Medium | 921600+ baud, powered hub, sequence-gap logging, per-window loss threshold |
| Fall class scarcity/imbalance | Medium | Dedicated fall-collection sessions, class weighting, report per-class recall |
| A device dies mid-project | Medium | Order 1–2 spare S3 boards now (cheap); train an ablation model on 2 RX so the system degrades gracefully |
| Router firmware quirks (rate adaptation changes MCS mid-session) | Medium | Lock 20 MHz, log MCS per packet, filter windows with mixed rates |
| Demo-day environment shift | Certain | Backup video; collect a 10-min adaptation set at the venue morning-of if allowed; confidence-thresholded "unknown" output |
| Scope creep (multi-person, through-wall, pose) | High | Out of scope for v1 — list as future work only |

---

## 5. Repo Structure (create in week 1)

```
majorp/
├── firmware/            # esp-csi based config, sdkconfig per device, flash notes
├── collector/           # serial/UDP ingestion, session manager, live heatmap
├── preprocessing/       # pure functions + unit tests
├── datasets/            # (git-ignored data) + datasheet.md + validation scripts
├── models/              # SenseFi-derived model zoo, training configs
├── experiments/         # configs + results CSVs + figure scripts
├── realtime/            # streaming inference + dashboard
├── report/              # LaTeX
└── docs/                # collection protocol, device layout photos, decisions log
```

---

## 6. Key References (verified to exist, July 2026)

- **Tooling:** [espressif/esp-csi](https://github.com/espressif/esp-csi) (official, S3-supported); [ESP32-CSI-Tool](https://github.com/StevenMHernandez/ESP32-CSI-Tool) (prior art, IDF 4.3-era); [Espressif ESP-CSI solution notes](https://docs.espressif.com/projects/esp-techpedia/en/latest/esp-friends/solution-introduction/esp-csi/esp-csi-solution.html) (chip CSI ranking)
- **Benchmark/models:** Yang et al., *SenseFi: A library and benchmark on deep-learning-empowered WiFi human sensing*, Patterns 4(3), 2023 — [paper](https://www.cell.com/patterns/fulltext/S2666-3899(23)00040-5) / [code](https://github.com/xyanchen/WiFi-CSI-Sensing-Benchmark)
- **ESP32 HAR datasets:** Moshiri et al. 2021 [CSI-HAR dataset](https://github.com/parisafm/CSI-HAR-Dataset) (7 activities, your ref [7]); *ESP-Fi HAR* dataset, Ad Hoc Networks, 2026 ([ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S1570870526000582))
- **ESP32 HAR feasibility:** *WiFi CSI-Based Long-Range Through-Wall HAR with the ESP32*, ICVS 2023 ([Springer](https://link.springer.com/chapter/10.1007/978-3-031-44137-0_4)); *Real-Time Human Activity Detection Using Wi-Fi CSI and LSTM on Edge Devices*, 2025 (>93% 5-fold CV — note: random-split regime)
- **Surveys:** Miao et al., ACM Computing Surveys 57(5), 2025 (your ref [6]); [Awesome-WiFi-CSI-Sensing](https://github.com/NTUMARS/Awesome-WiFi-CSI-Sensing) curated paper list
- Your proposal's refs [1]–[9] all check out as real publications; keep them.

---

## 7. First Week — Concrete TODO

1. `git init` this folder, push to GitHub, add this plan + proposal
2. Order 1–2 spare ESP32-S3 boards + powered USB hub + mattress access for falls
3. Install ESP-IDF v5.x; clone esp-csi; flash `csi_recv_router` on device #1
4. WiFi channel survey of the collection room; lock router config
5. Get one CSI heatmap that visibly reacts to a person → screenshot → it goes in the report's methodology chapter
