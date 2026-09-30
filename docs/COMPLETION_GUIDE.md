# Completion guide — what's left to finish this project

Audited 2026-08-30 against IMPLEMENTATION_PLAN.md. Read AGENTS.md first
(project brief, hard-won facts, non-negotiable evaluation rules); this file
only says **what remains and who does it**. Update the checkboxes here as
work lands; don't duplicate protocol details that live in
`docs/collection_protocol.md`.

## Where the project stands

- **All host-side software is DONE and green** (219 tests pass): parser →
  collector/preflight/session_script → preprocessing → dataset/splits →
  baselines + CNN/CNN-LSTM → ablation runner → realtime engine → dashboard →
  figures → LaTeX skeleton. No software phase is blocking.
- **Hardware bring-up is DONE** (M1 passed): 3 receivers verified end-to-end
  2026-08-27 at 100 Hz / 0 loss; nulls confirmed; latency measured (p95
  ~23 ms, structural). Port/IP/MAC table in `docs/collection_protocol.md`.
- **Pilot session 1 is DONE** (2026-08-27/28, ~30 min, 1 subject, no falls).
  Read `docs/pilot_session_1.md` — its findings drive everything below.
  Key result: numbers from that session are NOT report-quality (contiguous
  blocks confound activity with time; background doesn't generalise across
  sessions; cross-session split was degenerate).
- **Collection progress: ~5% of M3** (1195 / 25,000 windows, 1 / 8 subjects,
  0 falling seconds, 1 environment, 1 day).
- Timeline check: plan puts Phase 3 (full collection) at weeks 9–14
  (Bhadra/Ashwin). It is now early Bhadra — **data collection is the
  critical path** and everything else waits on it.

## Blockers found in the 2026-08-30 audit (fix before anything else)

- [ ] **Raw pilot session data is missing from every location checked.**
  `datasets/raw/` on this machine is empty; Drive
  (`wifi-csi-har-data/`) holds only the assembled `pilot_v0.npz` (196 MB),
  checkpoints, results, figures — no session Parquet/metadata/labels
  sidecars. If the raw sessions exist on another machine or disk, copy them
  back into `datasets/raw/` and to a second backup (M3 requires ≥2 copies).
  If they are truly gone: the assembled npz keeps pilot analysis
  reproducible, but `csihar.datasheet` progress tracking and any
  re-assembly (different window/hop, subcarrier choices) lose session 1 —
  note it and move on; session 1 data was never going to be in the report
  anyway.
- [ ] **Uncommitted repo state.** `CLAUDE.md` was rewritten to generic
  guidelines with the project brief moved to `AGENTS.md` (+ a stray root
  copy `wifi csi har.md` — likely accidental, probably delete);
  `.agents/`, `docs/figures/cm_*.png`, `experiments/results/dl.csv` are
  untracked. Evaluation rule 4 requires results CSVs and configs committed.
  Decide the CLAUDE.md/AGENTS.md arrangement deliberately, commit the rest.
- [ ] `experiments/results/dl.csv` rows have `git_sha=unknown` (Colab runs
  without the repo). Future Colab runs must record the SHA — pass it in or
  clone the repo in the notebook. Those pilot rows also predate the
  degenerate-split detection; treat their cross-session numbers as invalid
  (accuracy 0.97 with macro-F1 0.16 = background-only test set).

## Remaining work — humans (hardware / bodies; agents cannot do these)

Ordered; each item names its skill.

- [ ] **Fill in device layout** in `docs/collection_protocol.md` (rx1–3 +
  router positions/heights), tape tripod feet, photograph. Session 2 is only
  comparable to future sessions if this is pinned. (5 minutes, do first.)
- [ ] **Decide the AP question**: dedicated spare router on channel 11 vs.
  staying on the ISP router's auto-select ch 6 (a -45 dBm neighbour shares
  it). A dedicated AP needs no internet, kills band steering, and needs no
  reflash (same SSID/passphrase → power-cycle boards, re-check DHCP IPs).
  Strongly recommended before mass collection; changing radios mid-dataset
  adds a confound.
- [ ] **Pilot session 2** (`/collect-session`), applying the four changes at
  the end of `docs/pilot_session_1.md`: interleave background between
  activities, reverse activity order, different day, `--lead-in 15`. Then
  `/validate-session`.
- [ ] **First falling data**: mattress + spotter, forward/backward/sideways,
  ~15 s spacing. Falling has 0 seconds collected and is the safety-critical
  class.
- [ ] **Recruit 4–6 outside subjects** (8–10 total incl. team). This has the
  longest lead time — start scheduling now.
- [ ] **Full collection campaign** (M3): ≥5 min/activity/subject, ≥3
  sessions/subject on different days, one half-session in a second room.
  Run `csihar.preflight` before and `/validate-session` after every session;
  track progress with `python -m csihar.datasheet datasets/raw`.
- [ ] **Back up every accepted session to ≥2 places same-day** (see blocker
  above for why this is now a checklist item, not advice).
- [ ] Run heavy training on Colab/Kaggle when the agent hands over configs
  (no GPU on this machine).
- [ ] Phase 6 humans-only: rehearse live demo 3× with an unseen person,
  record the backup video, verify department report format, defense prep.

## Remaining work — AI agent (do when the data exists)

Trigger for most of this: a new validated session landing in
`datasets/raw/`, or the user saying data is ready.

- [ ] **After pilot session 2** (`/train-model`): re-assemble, re-run
  baselines + CNN/CNN-LSTM on random and cross-session splits (now
  non-degenerate — background is interleaved). Verify the pilot-1 drift
  hypothesis: if reversing activity order collapses the within-session
  number, say so in a `docs/pilot_session_2.md` mirroring session 1's doc.
- [ ] **Cross-subject (LOSO) split** becomes mandatory in every results
  table the moment a second subject exists (evaluation rule 1 — until now
  only two splits were possible).
- [ ] **Falling recall** column must be populated once falling data exists
  (evaluation rule 3); check class weighting behaves with the real
  imbalance.
- [ ] **Ablations on the real dataset** (`csihar.experiments`, configs in
  `experiments/configs/`): receivers 1/2/3, window length, packet rate; add
  `--split cross-subject` once subjects allow. These are the report's
  analysis chapter.
- [ ] **Cross-environment result** once the second-room half-session exists
  (deferred gap 2.8 in `docs/research_review.md`).
- [ ] **Consider augmentation only if** cross-subject accuracy disappoints
  (deferred-gap triggers in `docs/research_review.md` §3 — check there
  before adding anything).
- [ ] **Dataset v1.0 freeze artifacts** (M3): final datasheet, backed-up
  npz + raw, tag the repo.
- [ ] **Figures**: extend `csihar.figures` as new result types appear; every
  report figure regenerates via `make figures`, no hand-made plots.
- [ ] **Report writing** (M6): chapters are ~200-line stubs. Methodology,
  system design, and dataset sections can be drafted NOW from
  IMPLEMENTATION_PLAN.md, AGENTS.md hard-won facts, collection_protocol,
  and pilot docs — don't wait for final numbers. Results/conclusion wait
  for frozen runs. Verify against the department's official format before
  submission.
- [ ] **Demo hardening** (M5): temporal smoothing thresholds tuned on real
  data, fall-alert fast path exercised, latency re-measured on the final
  model. Stretch: on-device int8 inference (esp-tflite-micro) only if M3–M4
  are comfortably done.

## Definition of done

- M3: frozen dataset v1.0 + datasheet meeting targets, 2+ backups.
- M4: results table with **all three splits × all models**, seeds fixed,
  configs committed, ≥85% on random/cross-session or the gap explained.
- M5: 10-min unrehearsed live demo with an unseen person.
- M6: report submitted; demo + backup video ready.

Every number that reaches the report obeys the four evaluation rules in
AGENTS.md. When in doubt, the pilot-1 story is the cautionary template: a
99% random-split accuracy was worth nothing.
