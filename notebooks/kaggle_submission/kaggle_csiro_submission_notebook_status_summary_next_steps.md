# Current status (Dec 22, 2025)

## What we confirmed
- You can load your head checkpoint successfully:
  - `/kaggle/input/dinov3-extraneck-4layerhead/pytorch/f0a/1/cv5_v1_f0a.pt`
  - It loads as a `dict` with keys like: `fold_scores`, `mean`, `std`, `states`.
- The DINOv3 pretrained backbone weights are available:
  - `/kaggle/input/dinov3-b-pretraind/pytorch/dinov3_b_pretraind/1/dinov3_vitb16_pretrain.pth`
- You cloned the DINOv3 repo inside the notebook:
  - Repo location: `/kaggle/working/dinov3`
  - This path should be used as `DINO_REPO` (not the `.pth` path).

## Competition test format & public leaderboard behavior
- `test.csv` in the notebook environment is **long format**:
  - Columns: `sample_id`, `image_path`, `target_name`
  - Example rows show **one image repeated 5 times** (one per target).
- In the notebook environment you currently see:
  - `len(df) = 5` (long rows)
  - `n_unique_images = 1`
  - This implies **public test is a tiny stub** (likely 1 image) and the public leaderboard is mainly a **pipeline sanity check**.

## Paths that should be used in the notebook
- `TEST_CSV = /kaggle/input/csiro-biomass/test.csv`
- Important path-join detail:
  - `image_path` values already look like `test/IDxxxx.jpg`, so set:
    - `IMAGE_ROOT = /kaggle/input/csiro-biomass`
  - Avoid setting `IMAGE_ROOT = .../test` or you’ll get `.../test/test/...`.

## Why your predictions repeat
- You observed predictions like a repeated 5-vector for each of the 5 rows:
  - e.g. a tensor of shape `[5, 5]` where all rows are identical.
- This is expected because:
  - All rows refer to the **same image**
  - The model output does **not** depend on `target_name` (it outputs all 5 targets at once)

## Current submission mapping outcome
- You produced a submission with 5 rows (one per target):
  - `Dry_Clover_g` → 0.0
  - `Dry_Dead_g` → 18.31
  - `Dry_Green_g` → 332.76
  - `Dry_Total_g` → 238.75
  - `GDM_g` → 176.62
- Mapping logic is correct **if and only if**:
  - `TARGETS` list order matches the model head output order
  - Any required post-processing (e.g., `expm1`) is applied correctly.

---

# Key risks to address next

## (R1) Internet / git clone during submission rerun
- Kaggle’s submission rerun may not have internet; cloning during submission can fail.
- Safer approach:
  - Package the DINOv3 repo into a Kaggle Dataset and attach it as input, or
  - Vendor minimal required code into the notebook.

## (R2) Target-space mismatch (log vs grams)
- If training used `log1p(y)` targets, inference must apply `expm1(pred)` before writing submission.
- If training used raw grams, do not apply `expm1`.
- Action: confirm from training code whether `y` was transformed with `log1p`.

## (R3) Target order mismatch
- If `TARGETS` order in the submission notebook differs from training, columns will be swapped.
- Action: copy `TARGETS = [...]` exactly (same order) from the training notebook.

## (R4) Inefficient inference due to long test format
- Current loader likely loops over long `df` rows, causing repeated inference per image.
- On the hidden test this can be 5× slower.
- Action: run inference on `df_img = df[[image_path]].drop_duplicates()` then map back using `target_name`.

---

# Recommended next steps (practical checklist)

## Step 1 — Hard-set config (no auto-detect)
- Use explicit constants:
  - `TEST_CSV`, `IMAGE_ROOT`, `WEIGHTS_PATH`, `DINO_REPO`, `DINO_WEIGHTS`
  - `IMAGE_PATH_COL = "image_path"`
  - `ID_COL = "sample_id"` (only if needed)

## Step 2 — Make inference robust to long-format CSV
- Build `df_img` (unique images) and infer once per image.
- Store `preds_by_path: {image_path -> vector(5)}`
- Build submission by iterating long `df` and selecting the correct target index via `target_name`.

## Step 3 — Verify target space
- Check whether `expm1` is needed:
  - Compare raw head outputs vs `expm1` outputs.
  - Confirm from training pipeline.

## Step 4 — Verify target order
- Print labeled outputs for a sample:
  - for each `i, t in enumerate(TARGETS): print(t, pred[i])`
- Sanity check: `Dry_Total_g` should typically be >= component dry masses (heuristic only).

## Step 5 — Submission safety for rerun
- Avoid relying on `git clone` in the final version.
- Ensure notebook writes `/kaggle/working/submission.csv` with columns exactly:
  - `sample_id`, `target`

---

# What to paste into the next chat to continue

## Useful debug outputs
- `TARGETS` list (exact order) from training notebook
- Whether training used `log1p` targets
- Model output shape check: `model(x).shape`
- A print of `df.shape`, `df['image_path'].nunique()`, and `df['target_name'].unique()`
- Confirmation whether DINOv3 repo is available without internet during rerun (dataset vs clone)

