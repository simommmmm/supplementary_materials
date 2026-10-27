# Supplementary Materials

This supplementary package accompanies the paper and provides the main resources underlying the study, with particular focus on the final annotation codebook, the derived datasets, and the reported baseline experiments.

## Contents

### 1. `codebook_final.pdf`

Final annotation codebook used in the study. It includes:

- the full set of annotation labels
- their operational definitions
- inclusion and exclusion criteria
- temporal thresholds where applicable
- the reduced set of final labels used for baseline modeling

### 2. `dataset_overview.pdf`

Overview of the released datasets and their construction pipeline. It summarizes:

- the distinction between CAM1, CAM2, and multi-view 3D datasets
- the temporal windowing strategy
- the final target labels
- the selection criteria used to build the ML-ready subsets

### 3. `data_dictionary.xlsx`

Column-level description of the released datasets, including:

- variable name
- type
- description
- dataset applicability

### 4. `thematic_analysis.pdf`

Document reporting the thematic analysis table derived from the educator interviews and used to support the co-design process.

### 5. `datasets/`

Derived tabular datasets used for baseline modeling.

#### `datasets/cam1/`

Person-level 2D feature dataset derived from CAM1.

#### `datasets/cam2/`

Person-level 2D feature dataset derived from CAM2.

#### `datasets/multiview_3d/`

Final person-level multi-view 3D datasets derived from real triangulation and controlled fusion.

#### `datasets/relational/`

Pair-level derived feature dataset used for the relational behavior benchmark reported in the paper.

The released relational benchmark contains 2,098 balanced 2-second pair-level windows:

- 1,049 `SocialOrientation`
- 1,049 `AffiliativeContact`

The benchmark file used for the reported grouped SVM experiment is:

`relational_tabular_features_2s_balanced_for_svm.csv`

### 6. `results/`

Summary files reporting the main baseline results for the released datasets.

#### `results/relational/grouped_svm/`

Outputs of the relational grouped SVM evaluation reported in the paper:

- `overall_metrics.json`
- `fold_metrics.csv`
- `classification_report.csv`
- `confusion_matrix.csv`
- `run_config.json`

The evaluation uses five outer folds grouped by `video_id`, with three-fold grouped cross-validation for model selection inside each training fold.

Reported aggregate performance:

- Accuracy: 0.3670
- Balanced accuracy: 0.3670
- Macro-F1: 0.3052
- Weighted-F1: 0.3052

#### `results/relational/3d_coverage/`

Coverage diagnostics for the relational multi-view 3D branch. These files document the limited availability of complete two-person 3D representations and support the decision not to report a binary 3D relational classifier.

### 7. `code/`

Code associated with the relational analyses reported in the paper.

#### `code/relational/train_svm_grouped.py`

Grouped SVM benchmark used for the reported relational classification experiment.

The model is implemented as a scikit-learn pipeline containing:

- `StandardScaler`
- `SVC(class_weight="balanced")`

Hyperparameters are selected using `GridSearchCV` with macro-F1 as the optimization metric.

#### `code/relational/diagnose_relational_3d_fused_coverage.py`

Diagnostic script used to quantify the availability of usable fused 3D pair-level representations for the relational task.

### 8. `checkpoints/`

Serialized fitted models for the relational grouped SVM benchmark.

#### `checkpoints/relational/grouped_svm/`

Contains one fitted pipeline for each outer cross-validation fold:

- `fold_01.joblib`
- `fold_02.joblib`
- `fold_03.joblib`
- `fold_04.joblib`
- `fold_05.joblib`

Each checkpoint contains the complete fitted pipeline, including the `StandardScaler` and the SVM selected for that fold.

Because the reported experiment is a five-fold grouped cross-validation benchmark, there is no single model checkpoint corresponding to all reported metrics. The five checkpoints correspond to the five fitted outer-fold models.

The checkpoints were regenerated using the released training configuration and validated against the original experiment: the aggregate metrics, fold metrics, classification report, and confusion matrix are identical to the original reported outputs.

## Reproducing the relational grouped SVM benchmark

From the repository root, the relational experiment can be reproduced with:

```bash
python code/relational/train_svm_grouped.py \
  --dataset_csv datasets/relational/relational_tabular_features_2s_balanced_for_svm.csv \
  --out_dir results/relational/grouped_svm_reproduction \
  --target_col label \
  --group_col video_id \
  --n_splits_outer 5 \
  --n_splits_inner 3 \
  --random_state 42 \
  --auto_numeric_features
```

The relational benchmark uses 69 numeric input features. Annotation-context features are not included in the reported baseline.

The script requires Python packages including:

- pandas
- scikit-learn
- joblib

## Notes on data format

- Rows in the CAM1, CAM2, and multi-view 3D individual datasets correspond to person-level temporal windows.
- Rows in the relational benchmark correspond to pair-level temporal windows.
- The datasets do not contain raw videos or directly identifying information.
- Annotated identities are pseudonymous and used only as internal linking variables across derived representations.
- The released files are derived feature representations designed for analysis and baseline modeling.
- Relational cross-validation is grouped by `video_id` to avoid assigning windows from the same video group to both training and test partitions within an outer fold.

## Ethical note

The supplementary materials do not include raw audiovisual recordings. The released datasets are limited to derived representations consistent with the privacy-preserving design of the project.

## Relationship between files

- The annotation codebook defines the labels.
- The dataset overview explains how labels and features were converted into trainable datasets.
- The data dictionary explains the meaning of the released variables.
- The relational dataset contains the derived pair-level features used by the grouped SVM benchmark.
- The relational training script reproduces the evaluation protocol.
- The result files contain the metrics reported for that evaluation.
- The checkpoint files contain the five fitted outer-fold pipelines.