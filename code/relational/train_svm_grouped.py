from pathlib import Path
import argparse
import json

import pandas as pd
from joblib import dump

from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


FEATURE_COLS = [
    "n_tracks_for_person",
    "n_pose_frames",
    "coverage_ratio_pose_frames",
    "quality_score_mean",
    "quality_score_min",
    "bbox_center_x_mean",
    "bbox_center_y_mean",
    "bbox_center_x_std",
    "bbox_center_y_std",
    "bbox_width_mean",
    "bbox_height_mean",
    "bbox_area_mean",
    "bbox_aspect_ratio_mean",
    "bbox_center_speed_mean",
    "bbox_center_speed_max",
    "bbox_center_dx_total",
    "bbox_center_dy_total",
    "shoulder_width_mean",
    "hip_width_mean",
    "torso_height_mean",
]


def undersample_majority_class(X, y, groups, class_name, keep_ratio, random_state):
    if keep_ratio is None or keep_ratio >= 1.0:
        return X, y, groups

    mask_major = (y == class_name)
    idx_major = y[mask_major].index
    idx_other = y[~mask_major].index

    n_major = len(idx_major)
    n_keep = max(1, int(round(n_major * keep_ratio)))

    idx_major_keep = idx_major.to_series().sample(
        n=n_keep,
        random_state=random_state,
        replace=False,
    ).tolist()

    keep_idx = sorted(idx_other.tolist() + idx_major_keep)

    return (
        X.loc[keep_idx].reset_index(drop=True),
        y.loc[keep_idx].reset_index(drop=True),
        groups.loc[keep_idx].reset_index(drop=True),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--target_col", default="label")
    parser.add_argument("--group_col", default="video_id")
    parser.add_argument("--n_splits_outer", type=int, default=5)
    parser.add_argument("--n_splits_inner", type=int, default=4)
    parser.add_argument("--random_state", type=int, default=42)

    parser.add_argument("--undersample_class", default=None)
    parser.add_argument("--undersample_keep_ratio", type=float, default=None)
    parser.add_argument(
        "--include_ctx_features",
        action="store_true",
        help="Include annotation-aware context features starting with ctx_.",
    )
    parser.add_argument(
        "--auto_numeric_features",
        action="store_true",
        help="Use all numeric columns as features after excluding metadata, target and group columns.",
    )
    args = parser.parse_args()

    dataset_csv = Path(args.dataset_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(dataset_csv)

    if args.auto_numeric_features:
        exclude_cols = {
            args.target_col,
            args.group_col,
            "label",
            "rel_label",
            "video_id",
            "entity_type",
            "pair_id",
            "entity_a",
            "entity_b",
            "annotated_id",
            "date",
            "session",
            "camera",
            "hour",
            "source_file",
            "pair_observation_status",
            "stitched_ids_a",
            "stitched_ids_b",
            "a_selected_stitched_id",
            "b_selected_stitched_id",
            "person_track_ids",
            "window_size_sec",
            "stride_sec",
            "t_start_ms",
            "t_end_ms",
            "t_start_sec",
            "t_end_sec",
            "n_expected_timesteps",
            "n_valid_timesteps",
            "n_missing_timesteps",
        }

        feature_cols = [
            c for c in df.columns
            if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])
        ]

        if not args.include_ctx_features:
            feature_cols = [c for c in feature_cols if not c.startswith("ctx_")]
    else:
        feature_cols = list(FEATURE_COLS)

        if args.include_ctx_features:
            ctx_cols = [
                c for c in df.columns
                if c.startswith("ctx_") and pd.api.types.is_numeric_dtype(df[c])
            ]
            feature_cols.extend(ctx_cols)

    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns in dataset: {missing_cols}")

    X = df[feature_cols].copy()
    y = df[args.target_col].copy()
    groups = df[args.group_col].copy()

    print("Rows:", len(df))
    print("Features used:", len(feature_cols))
    print("Context features used:", [c for c in feature_cols if c.startswith("ctx_")])
    print("Groups:", groups.nunique())

    outer_cv = GroupKFold(n_splits=args.n_splits_outer)

    all_true = []
    all_pred = []
    fold_rows = []

    checkpoints_dir = out_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups), start=1):
        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        X_test = X.iloc[test_idx].reset_index(drop=True)
        y_test = y.iloc[test_idx].reset_index(drop=True)

        train_groups = groups.iloc[train_idx].reset_index(drop=True)
        test_groups = groups.iloc[test_idx].reset_index(drop=True)

        X_train_used, y_train_used, train_groups_used = undersample_majority_class(
            X_train,
            y_train,
            train_groups,
            class_name=args.undersample_class,
            keep_ratio=args.undersample_keep_ratio,
            random_state=args.random_state + fold_idx,
        )

        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("svc", SVC(class_weight="balanced")),
        ])

        param_grid = {
            "svc__kernel": ["rbf"],
            "svc__C": [0.1, 1, 10],
            "svc__gamma": ["scale", 0.01, 0.001],
        }

        inner_cv = GroupKFold(n_splits=args.n_splits_inner)

        search = GridSearchCV(
            estimator=pipe,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=inner_cv,
            n_jobs=-1,
            refit=True,
        )

        search.fit(X_train_used, y_train_used, groups=train_groups_used)

        best_model = search.best_estimator_
        dump(best_model, checkpoints_dir / f"fold_{fold_idx:02d}.joblib")

        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        bacc = balanced_accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")

        fold_rows.append({
            "fold": fold_idx,
            "test_videos": sorted(test_groups.unique().tolist()),
            "n_train_original": int(len(train_idx)),
            "n_train_used": int(len(X_train_used)),
            "n_test": int(len(test_idx)),
            "accuracy": acc,
            "balanced_accuracy": bacc,
            "f1_macro": f1_macro,
            "f1_weighted": f1_weighted,
            "best_params": search.best_params_,
        })

        all_true.extend(y_test.tolist())
        all_pred.extend(y_pred.tolist())

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(out_dir / "fold_metrics.csv", index=False)

    overall = {
        "accuracy": accuracy_score(all_true, all_pred),
        "balanced_accuracy": balanced_accuracy_score(all_true, all_pred),
        "f1_macro": f1_score(all_true, all_pred, average="macro"),
        "f1_weighted": f1_score(all_true, all_pred, average="weighted"),
    }

    with open(out_dir / "overall_metrics.json", "w", encoding="utf-8") as f:
        json.dump(overall, f, indent=2, ensure_ascii=False)

    report_dict = classification_report(all_true, all_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df.to_csv(out_dir / "classification_report.csv")

    labels = sorted(pd.Series(y).unique().tolist())
    cm = confusion_matrix(all_true, all_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    cm_df.to_csv(out_dir / "confusion_matrix.csv")

    run_config = {
        "dataset_csv": str(dataset_csv),
        "out_dir": str(out_dir),
        "target_col": args.target_col,
        "group_col": args.group_col,
        "n_splits_outer": args.n_splits_outer,
        "n_splits_inner": args.n_splits_inner,
        "random_state": args.random_state,
        "undersample_class": args.undersample_class,
        "undersample_keep_ratio": args.undersample_keep_ratio,
        "feature_cols": feature_cols,
    }
    with open(out_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2, ensure_ascii=False)

    print("\n=== OVERALL METRICS ===")
    for k, v in overall.items():
        print(f"{k}: {v:.4f}")

    print("\n=== FOLD METRICS ===")
    print(
        fold_df[
            [
                "fold",
                "n_train_original",
                "n_train_used",
                "n_test",
                "accuracy",
                "balanced_accuracy",
                "f1_macro",
                "f1_weighted",
            ]
        ].to_string(index=False)
    )

    print("\nSaved to:")
    print(out_dir)


if __name__ == "__main__":
    main()
