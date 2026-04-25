# GROUND TRUTH — SoccerNet SynLoc Ch1
**Sporządzono:** 2026-04-24 (koniec dnia)
**Autor:** forensic session
**Zasada:** FAKTY z plików > opis z rozmów. Domysły = NIEZWERYFIKOWANE.

---

## 1. STAN LEADERBOARDU

### TEST phase (9 309 images, 500/day)

| Submission | Score (mAP-LocSim) | Config | Threshold | Status |
|---|---|---|---|---|
| **v6** (baseline m960) | **63.91** ← BEST | 960×960 ✓ | 0.05 (broken) | #4 confirmed, CLAUDE.md |
| v4 | 49.32 | 640×640 ✗ | unknown | CLAUDE.md — regression |
| v5 (threshold sweep) | **NIEZWERYFIKOWANE** | 960×960 ✓ | 0.35 | local zip exists, nie wiadomo czy wrzucone |
| v7 TTA | **NIEZWERYFIKOWANE** | 960×960 ✓ | 0.05 | local zip exists, nie wiadomo czy wrzucone |

### CHALLENGE phase (11 352 images, 2/day)

| Submission | Score | Notes |
|---|---|---|
| v4 challenge | ~51.11 | Cytowany w `failure_analysis_m960.md` — ale score dotyczy challenge split i może być stare |
| v5/v6/v7 challenge | **NIEZWERYFIKOWANE** | Zips wygenerowane lokalnie, nie ma potwierdzenia wrzucenia |

### Referencje
- **#1 juliantziegler**: 87.39 — `CLAUDE.md`
- **#3 hakanardo** (Spiideo official): 76.17 — `CLAUDE.md` + `spiideo_baseline_audit.md`
- **#4 nas**: 63.91 — gap do #3 = **12.26 pts**

---

## 2. ATHENA — FAKTY

### Potwierdzone (z plików na kuba)
| Checkpoint | Ścieżka na kuba | Rozmiar | Data transferu |
|---|---|---|---|
| m960_ft_e20 | `models/finetuned/m960_ft_e20.pth` | 481 MB | 2026-04-24 17:12 |
| m960_ft_e30 | `models/finetuned/m960_ft_e30.pth` | 487 MB | 2026-04-24 17:11 |

**Wniosek:** Job 2545543 (m960 finetune) DONE — oba checkpointy na kuba.

### NIEZWERYFIKOWANE
- **Job 2545544 (m1280 finetune):** Brak checkpointu w `models/finetuned/`. Może nadal trwać, zakończyć się błędem lub nie być jeszcze skopiowany. **Wymaga SSH do Atheny.**
- Config finetune (load_from vs from-scratch): nieznany bez squeue/slurm log z Atheny. Wcześniejszy plan (`submission_plan_24h.md`) mówił że 2544643/2544644 były from-scratch. 2545543/2545544 to nowe joby (restarted jako finetune?) — NIEZWERYFIKOWANE.

---

## 3. MODELE I EWALUACJE

### Checkpointy lokalne
```
models/baselines/
  yoloxpose_m_4xb64-300e_960_epoch_300.pth  (555 MB)  ← używany w v6 (63.91)
  yoloxpose_m_4xb64-300e_640_epoch_300.pth  (494 MB)
  yoloxpose_s_4xb64-300e_{640,960}_epoch_300.pth
  yoloxpose_tiny_4xb64-300e_{640,960}_epoch_300.pth
models/finetuned/
  m960_ft_e20.pth  (481 MB, Apr 24 17:12)
  m960_ft_e30.pth  (487 MB, Apr 24 17:11)  ← najnowszy finetune
```

### Wyniki ewaluacji val (empiryczne)

| Model | mAP-LocSim val | Jak mierzono | Plik |
|---|---|---|---|
| Baseline m960 ep300 | **47.01** @th=0.05 | sskit bezpośrednio na val_predictions.pkl + annotations_fullhd/val.json | `reports/threshold_sweep.md` |
| Baseline m960 ep300 | ~0.000 | mmengine via evaluate.sh (BROKEN: annotations/val.json 4K + broken override) | `work_dirs/m960_fixed_val_eval*.log` |
| m960_ft_e30 | **58.39** (0.584) | mmengine evaluate, val, annotations_fullhd z configa | `reports/m960_ft_e30_val_eval_v3.log:L-final` |
| m960_ft_e30 (locsim_bbox) | 0.298 | jak wyżej (różny evaluator mode) | jak wyżej |
| m960_ft_e30 (F1-opt threshold) | **0.658** | F1@LocSim=0.5 na val | jak wyżej |
| m960_ft_e20 | INCOMPLETE | log urwany na 4550/6777 (brak ostatniej metryki) | `reports/m960_ft_e20_val_eval.log` |

**Porównanie baseline vs ft_e30 (oba na val):** +11.4 pts (47.01 → 58.39)

**TEST leaderboard (Codabench):**
- Baseline m960 ep300: 63.91 (v6, threshold=0.05) ← POTWIERDZONE

---

## 4. ZWERYFIKOWANE USTALENIA

| Fakt | Status | Dowód |
|---|---|---|
| Config 960 bug → fixed w v6 | ✅ FIXED | `work_dirs/submit_v6_960config.log`: `input_size=(960,960)` |
| `scripts/submit.sh:28` val_dataloader override → score_threshold=0.05 | ✅ BUG NADAL OBECNY | `scripts/submit.sh:28`; `work_dirs/submission_v6_*/submission.zip` metadata |
| F1-optimal threshold dla ft_e30 = 0.658 | ✅ EMPIRYCZNE | `reports/m960_ft_e30_val_eval_v3.log` (locsim/score_threshold) |
| F1-optimal threshold dla baseline ep300 ≈ 0.499–0.565 | ✅ Z LOGÓW | `work_dirs/m960_fixed_val_eval_v3.log` (score_threshold: 0.499/0.564) |
| `make_submission.py` dead code (optional_keypoint_index nie istnieje) | ✅ POTWIERDZONE | `scripts/make_submission.py:86`; `spiideo_baseline_audit.md` |
| `evaluate.sh` używa annotations/ (4K) → zero mAP | ✅ BUG AKTYWNY | `scripts/evaluate.sh:37`, `work_dirs/m960_fixed_val_eval*.log` |
| v7 TTA threshold=0.05 (nie F1-optimal) | ✅ POTWIERDZONE | `work_dirs/submission_v7_tta_test/submission.zip` metadata |
| v5 threshold=0.35 (ręcznie z val F1 analizy) | ✅ POTWIERDZONE | `work_dirs/submission_v5_test/submission.zip` metadata |
| Keypoints v6 w 4K coords (y≈1255) | ✅ POTWIERDZONE | `work_dirs/submission_v6_challenge/submission.zip` results.json[0] |
| mAP-LocSim monotonically spada ze wzrostem threshold | ✅ EMPIRYCZNE | `reports/threshold_sweep.md` Sweep 1 |

---

## 5. DECYZJE DO PODJĘCIA

### 🔴 P0 — Teraz, przed kolejnym submitem

**A. Fix val_dataloader override w submit.sh**
- Plik: `scripts/submit.sh:28`
- Usunąć linię: `--cfg-options "val_dataloader.dataset.ann_file=... val_dataloader.dataset.data_prefix.img=challenge"`
- Efekt: runner.val() dostaje prawdziwe GT → _val_stats.json → threshold 0.658 zamiast 0.05
- Oczekiwany gain: **+6–10 pts** (z 63.91 → ~70–74)

**B. Co submitować dziś na test?**
- Opcja 1: Napraw submit.sh + uruchom v8 z baseline m960 → oczekiwane ~70–74
- Opcja 2: Napraw submit.sh + uruchom v8 z ft_e30 (val +11 pts vs baseline) → oczekiwane ~73–78?
- Opcja 3: Zachowaj slot testowy, poczekaj na m1280

### 🟡 P1 — Sprawdź Athena

**C. Status m1280 finetune (2545544)**
```bash
ssh plgjkomosa@athena.cyfronet.pl 'squeue -u plgjkomosa; ls -lht $SCRATCH/synloc/work_dirs/ | head -10; tail -20 $(ls -1t $SCRATCH/synloc/slurm-*.out | head -1)'
```
- Jeśli running: ile epok, ETA, loss
- Jeśli done: scp checkpoint, uruchom val eval

**D. Czy 2545543/2545544 to finetune (load_from baseline) czy from-scratch?**
- Krytyczne dla interpretacji ft_e30 val 58.4%
- Sprawdź: `grep "load_from" $SCRATCH/synloc/configs/...` lub slurm log startup

### 🟡 P2 — Challenge submission

**E. Użyć 1 challenge slot dziś?**
- Mamy dostępne 2/dzień
- Najlepszy kandydat: naprawiony submit.sh + ft_e30 → przewidywane ~73–78 challenge?
- Ryzyko: traci slot jeśli coś pójdzie nie tak

---

## 6. CO NIE WIEMY (uncertainties)

| Pytanie | Co wiemy | Czego nie wiemy |
|---|---|---|
| Score v5 na test leaderboard | zip istnieje z th=0.35 | Czy był wrzucony na Codabench |
| Score v7 TTA na test | zip istnieje z th=0.05 | Czy był wrzucony |
| m1280 finetune status | Brak kptu na kuba | Running/done/failed na Athenie |
| ft_e30 config (from-scratch vs finetune) | Checkpunkt jest na kuba | Czy load_from baseline (kluczowe!) |
| Baseline test score bez val_dataloader override | v6=63.91 z broken override | Ile wyniosłoby poprawnie (~70+?) |
| ft_e30 test score | val=58.4% | Nie submittowano jeszcze |
| m1280 val score | - | Checkpoint niedostępny |
| Czy v6/v7 challenge były uploadowane | Zips istnieją | Potwierdzenie uploadu na Codabench |

---

*Źródła: CLAUDE.md, reports/*.md, work_dirs/*.log, models/, scripts/submit.sh*
