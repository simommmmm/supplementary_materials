import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def frame_time_ms(frame_idx, fps):
    if pd.isna(frame_idx):
        return np.nan
    return float(frame_idx) * 1000.0 / float(fps)


def find_frame_master(project_root: Path, video_id: str):
    date, hour = video_id.split("_")
    p = (
        project_root
        / "data"
        / "processed"
        / date
        / "001"
        / "multiview"
        / "ml_prep_3d"
        / hour
        / "frame_master"
        / f"{hour}_frame_master_3d_identified.csv"
    )
    return p if p.exists() else None


def count_metric_xyz(row):
    count = 0
    for j in range(33):
        x = row.get(f"j{j:02d}_x_m", np.nan)
        y = row.get(f"j{j:02d}_y_m", np.nan)
        z = row.get(f"j{j:02d}_z_m", np.nan)
        if pd.notna(x) and pd.notna(y) and pd.notna(z):
            count += 1
    return count


def build_person_index(frame_master: pd.DataFrame, fps: float):
    df = frame_master.copy()

    df = df[df["annotated_id"].notna()].copy()

    if "role" in df.columns:
        df = df[df["role"].astype(str).str.lower().eq("participant")].copy()

    if "mapping_confidence" in df.columns:
        df = df[df["mapping_confidence"].astype(str).str.lower().eq("high")].copy()

    if "usable_for_ml" in df.columns:
        df = df[df["usable_for_ml"].astype(str).str.lower().eq("yes")].copy()

    df["t_ms"] = df["frame_cam1"].apply(lambda x: frame_time_ms(x, fps))

    for col in [
        "mv_joint_count",
        "sv_cam1_joint_count",
        "sv_cam2_joint_count",
        "sv_total_joint_count",
        "fused_joint_count",
        "mean_err",
        "identity_match_dist_px",
    ]:
        if col not in df.columns:
            df[col] = 0.0

    df["metric_xyz_joint_count"] = df.apply(count_metric_xyz, axis=1)

    index = {}
    for annotated_id, g in df.groupby("annotated_id"):
        index[str(annotated_id)] = g.sort_values("t_ms").reset_index(drop=True)

    return index


def summarize_person_window(person_df, t_start_ms, t_end_ms):
    if person_df is None or len(person_df) == 0:
        return {
            "has_node": 0,
            "has_fused": 0,
            "has_mv": 0,
            "has_metric_xyz": 0,
            "n_rows": 0,
            "n_fused_rows": 0,
            "n_mv_rows": 0,
            "n_metric_xyz_rows": 0,
            "fused_joint_count_mean": 0.0,
            "mv_joint_count_mean": 0.0,
            "sv_total_joint_count_mean": 0.0,
            "metric_xyz_joint_count_mean": 0.0,
            "mean_err_mean": 0.0,
            "identity_match_dist_px_mean": 0.0,
        }

    w = person_df[
        (person_df["t_ms"] >= float(t_start_ms))
        & (person_df["t_ms"] < float(t_end_ms))
    ].copy()

    if len(w) == 0:
        return summarize_person_window(None, t_start_ms, t_end_ms)

    fused_rows = w[w["fused_joint_count"].fillna(0) > 0]
    mv_rows = w[w["mv_joint_count"].fillna(0) > 0]
    metric_rows = w[w["metric_xyz_joint_count"].fillna(0) > 0]

    def safe_mean(col):
        vals = pd.to_numeric(w[col], errors="coerce").dropna()
        return float(vals.mean()) if len(vals) else 0.0

    return {
        "has_node": 1,
        "has_fused": int(len(fused_rows) > 0),
        "has_mv": int(len(mv_rows) > 0),
        "has_metric_xyz": int(len(metric_rows) > 0),
        "n_rows": int(len(w)),
        "n_fused_rows": int(len(fused_rows)),
        "n_mv_rows": int(len(mv_rows)),
        "n_metric_xyz_rows": int(len(metric_rows)),
        "fused_joint_count_mean": safe_mean("fused_joint_count"),
        "mv_joint_count_mean": safe_mean("mv_joint_count"),
        "sv_total_joint_count_mean": safe_mean("sv_total_joint_count"),
        "metric_xyz_joint_count_mean": safe_mean("metric_xyz_joint_count"),
        "mean_err_mean": safe_mean("mean_err"),
        "identity_match_dist_px_mean": safe_mean("identity_match_dist_px"),
    }


def prefix(d, name):
    return {f"{name}_{k}": v for k, v in d.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_root", required=True)
    parser.add_argument(
        "--input_csv",
        default="data/processed/relational_graph_dataset/relational_pairs_mapped_2s_balanced.csv",
    )
    parser.add_argument(
        "--output_csv",
        default="data/processed/relational_graph_dataset/relational_3d_fused_coverage_2s_balanced.csv",
    )
    parser.add_argument(
        "--summary_txt",
        default="data/processed/relational_graph_dataset/summary_3d_fused_coverage_2s_balanced.txt",
    )
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()

    project_root = Path(args.project_root)
    input_csv = project_root / args.input_csv
    output_csv = project_root / args.output_csv
    summary_txt = project_root / args.summary_txt

    rel = pd.read_csv(
        input_csv,
        dtype={"rel_label": "string", "label": "string"},
        low_memory=False,
    )

    cache = {}
    rows = []

    for _, r in rel.iterrows():
        video_id = str(r["video_id"])

        if video_id not in cache:
            fm_path = find_frame_master(project_root, video_id)
            if fm_path is None:
                cache[video_id] = None
            else:
                fm = pd.read_csv(fm_path, low_memory=False)
                cache[video_id] = build_person_index(fm, args.fps)

        idx = cache[video_id]

        out = r.to_dict()
        out["frame_master_found"] = int(idx is not None)

        if idx is None:
            a = summarize_person_window(None, r["t_start_ms"], r["t_end_ms"])
            b = summarize_person_window(None, r["t_start_ms"], r["t_end_ms"])
        else:
            a = summarize_person_window(
                idx.get(str(r["entity_a"])),
                r["t_start_ms"],
                r["t_end_ms"],
            )
            b = summarize_person_window(
                idx.get(str(r["entity_b"])),
                r["t_start_ms"],
                r["t_end_ms"],
            )

        out.update(prefix(a, "a"))
        out.update(prefix(b, "b"))

        out["has_any_3d_node"] = int(out["a_has_node"] == 1 or out["b_has_node"] == 1)
        out["has_both_3d_nodes"] = int(out["a_has_node"] == 1 and out["b_has_node"] == 1)

        out["has_any_fused_node"] = int(out["a_has_fused"] == 1 or out["b_has_fused"] == 1)
        out["has_both_fused_nodes"] = int(out["a_has_fused"] == 1 and out["b_has_fused"] == 1)

        out["has_any_mv_node"] = int(out["a_has_mv"] == 1 or out["b_has_mv"] == 1)
        out["has_both_mv_nodes"] = int(out["a_has_mv"] == 1 and out["b_has_mv"] == 1)

        out["has_any_metric_xyz_node"] = int(
            out["a_has_metric_xyz"] == 1 or out["b_has_metric_xyz"] == 1
        )
        out["has_both_metric_xyz_nodes"] = int(
            out["a_has_metric_xyz"] == 1 and out["b_has_metric_xyz"] == 1
        )

        rows.append(out)

    out_df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_csv, index=False)

    summary = []
    summary.append("RELATIONAL 3D/FUSED COVERAGE DIAGNOSTIC")
    summary.append("=" * 80)
    summary.append(f"Input: {input_csv}")
    summary.append(f"Output: {output_csv}")
    summary.append(f"Rows: {len(out_df)}")
    summary.append(f"FPS used for frame-to-time conversion: {args.fps}")
    summary.append("")
    summary.append("Label counts:")
    summary.append(str(out_df["rel_label"].value_counts(dropna=False)))
    summary.append("")
    summary.append("frame_master_found:")
    summary.append(str(out_df["frame_master_found"].value_counts(dropna=False)))
    summary.append("")
    for col in [
        "has_any_3d_node",
        "has_both_3d_nodes",
        "has_any_fused_node",
        "has_both_fused_nodes",
        "has_any_mv_node",
        "has_both_mv_nodes",
        "has_any_metric_xyz_node",
        "has_both_metric_xyz_nodes",
    ]:
        summary.append("")
        summary.append(col + ":")
        summary.append(str(out_df[col].value_counts(dropna=False)))
        summary.append("")
        summary.append(f"Label x {col}:")
        summary.append(str(pd.crosstab(out_df["rel_label"], out_df[col], dropna=False)))

    summary.append("")
    summary.append("Rows by video/label/has_both_fused_nodes:")
    summary.append(str(
        out_df.groupby(["video_id", "rel_label", "has_both_fused_nodes"])
        .size()
        .reset_index(name="count")
        .sort_values(["video_id", "rel_label", "has_both_fused_nodes"])
        .to_string(index=False)
    ))

    summary.append("")
    summary.append("Rows by video/label/has_both_metric_xyz_nodes:")
    summary.append(str(
        out_df.groupby(["video_id", "rel_label", "has_both_metric_xyz_nodes"])
        .size()
        .reset_index(name="count")
        .sort_values(["video_id", "rel_label", "has_both_metric_xyz_nodes"])
        .to_string(index=False)
    ))

    summary_txt.write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))


if __name__ == "__main__":
    main()