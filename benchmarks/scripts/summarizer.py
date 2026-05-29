import argparse
import pandas as pd
from pathlib import Path
import glob
from datetime import datetime
from tqdm import tqdm
import traceback
import yaml


def read_metrics_file(metrics_file):
    def process_metric(metric_dict, key_prefix=""):
        result = {}
        for key, value in metric_dict.items():
            if isinstance(value, list):
                result[f"{key_prefix}/{key}"] = value[-1]
            else:
                result[f"{key_prefix}/{key}"] = value
        return result

    with open(metrics_file, "r") as f:
        metrics = yaml.safe_load(f)

    final_metrics = {}
    if "classification" in metrics:
        r = process_metric(metrics["classification"], "classification")
        final_metrics.update(r)
    if "regression" in metrics:
        r = process_metric(metrics["regression"], "regression")
        final_metrics.update(r)

    return final_metrics


def read_timing_csv(timing_file_path):
    """Read timing CSV file and extract duration information."""
    try:
        if timing_file_path.exists():
            timing_df = pd.read_csv(timing_file_path)

            if timing_df.empty:
                return None

            # Check for elapsed_seconds column (that's what your files have!)
            if "elapsed_seconds" in timing_df.columns:
                total_duration = timing_df["elapsed_seconds"].sum()
                print(f"✅ Found elapsed_seconds column: {total_duration:.2f} seconds")
                return {
                    "total_duration_seconds": total_duration,
                    "num_entries": len(timing_df),
                    "tasks": (
                        timing_df["task"].tolist()
                        if "task" in timing_df.columns
                        else []
                    ),
                    "columns": list(timing_df.columns),
                }
            else:
                # Fallback: try other column names
                for col in timing_df.columns:
                    if any(
                        keyword in col.lower()
                        for keyword in ["duration", "time", "elapsed"]
                    ):
                        if pd.api.types.is_numeric_dtype(timing_df[col]):
                            total_duration = timing_df[col].sum()
                            print(
                                f"✅ Found duration column '{col}': {total_duration:.2f} seconds"
                            )
                            return {
                                "total_duration_seconds": total_duration,
                                "num_entries": len(timing_df),
                                "tasks": (
                                    timing_df["task"].tolist()
                                    if "task" in timing_df.columns
                                    else []
                                ),
                                "columns": list(timing_df.columns),
                            }

                print(f"⚠️  Could not find duration column in {timing_file_path}")
                print(f"   Available columns: {list(timing_df.columns)}")

    except Exception as e:
        print(f"❌ Error reading timing file {timing_file_path}: {e}")

    return None


def get_pretrain_timing(pretrain_id, root_dir):
    """Find pretraining timing for a given pretrain ID."""
    # Search for pretrain timing in the pretrain directory
    pretrain_root_dir = root_dir.parent.parent / pretrain_id / "final"

    timing_patterns = [f"timings_fit.csv", f"timing_metrics.csv", f"timings_*.csv"]

    for pattern in timing_patterns:
        timing_files = list(pretrain_root_dir.glob(pattern))
        if timing_files:
            timing_file = timing_files[0]
            timing_info = read_timing_csv(timing_file)
            if timing_info:
                return timing_info["total_duration_seconds"]

    return None


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(
        description="Check execution directories for required files."
    )
    parser.add_argument(
        "executions_csv_path", type=str, help="Path to the CSV file to process"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        help="Path to save the updated CSV file. If None, print on screen",
        default=None,
        required=False,
    )
    parser.add_argument(
        "--include_timing",
        action="store_true",
        help="Include timing information from timings_*.csv files",
    )
    args = parser.parse_args()

    # Load the CSV file
    df = pd.read_csv(args.executions_csv_path)

    # Ensure new columns exist in the DataFrame
    df["execution/run_id"] = ""
    df["execution/metric_file"] = ""
    df["execution/run_file"] = ""

    # Add timing columns if requested
    if args.include_timing:
        df["execution/pretrain_duration_seconds"] = ""
        df["execution/task_duration_seconds"] = ""
        df["execution/timing_file"] = ""
        df["execution/pretrain_id"] = ""
        df["execution/total_duration_seconds"] = ""

    # Iterate over each row with a progress bar
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows"):
        try:
            root_dir = Path(row["execution/root_dir"])

            # 1. Look for metric files
            metric_files = root_dir.glob("metrics_*.yaml")
            metric_files = sorted(
                metric_files, key=lambda x: x.stat().st_mtime, reverse=True
            )

            if metric_files:
                latest_metric_file = metric_files[0]
                metrics = read_metrics_file(latest_metric_file)

                # Add metrics to the DataFrame
                for key, value in metrics.items():
                    df.at[index, f"metric/{key}"] = value

                # Extracting run_id from the file name
                run_id = latest_metric_file.stem.split("-")[-1]
                df.at[index, "execution/run_id"] = run_id
                df.at[index, "execution/metric_file"] = str(latest_metric_file)

                run_file = list(latest_metric_file.parent.glob(f"run_*{run_id}.yaml"))
                if run_file:
                    run_file = run_file[0]
                    df.at[index, "execution/run_file"] = str(run_file)

            else:
                # Populate columns with empty strings if no metric file is found
                df.at[index, "execution/run_id"] = ""
                df.at[index, "execution/metric_file"] = ""
                df.at[index, "execution/run_file"] = ""

            # 2. Look for timing files if requested
            if args.include_timing:
                # Find timing files in the current directory
                timing_files = list(root_dir.glob("timings_*.csv"))

                if timing_files:
                    # Sort by modification time, get the most recent
                    timing_files = sorted(
                        timing_files, key=lambda x: x.stat().st_mtime, reverse=True
                    )
                    timing_file = timing_files[0]

                    # Read timing information
                    timing_info = read_timing_csv(timing_file)

                    if (
                        timing_info
                        and timing_info["total_duration_seconds"] is not None
                    ):
                        df.at[index, "execution/task_duration_seconds"] = timing_info[
                            "total_duration_seconds"
                        ]
                        df.at[index, "execution/timing_file"] = str(timing_file)

                        # Log what tasks were found
                        if timing_info["tasks"]:
                            print(
                                f"✅ Found timing tasks: {timing_info['tasks']} in {timing_file}"
                            )

                # Look for pretrain timing if this is a finetune or evaluate task
                task = row.get("pipeline/task", "")

                # Extract pretrain ID from different possible columns
                pretrain_id = None

                # Try to get pretrain ID from backbone/load_from_uid
                if pd.notna(row.get("backbone/load_from_uid")):
                    pretrain_id = row["backbone/load_from_uid"]

                # If not found, try to extract from ckpt_path in bash_command
                elif pd.notna(row.get("execution/bash_command")):
                    import re

                    bash_cmd = row["execution/bash_command"]
                    # Look for patterns like logs/id_xxxxxx/final/checkpoints/
                    pattern = r"logs/(id_[a-f0-9]+)/final/checkpoints/"
                    match = re.search(pattern, bash_cmd)
                    if match:
                        pretrain_id = match.group(1)

                # If we found a pretrain ID, look for its timing
                if pretrain_id:
                    df.at[index, "execution/pretrain_id"] = pretrain_id

                    # Only get pretrain timing for finetune/evaluate tasks
                    if task in ["train", "finetune", "evaluate"]:
                        pretrain_duration = get_pretrain_timing(pretrain_id, root_dir)
                        if pretrain_duration:
                            df.at[index, "execution/pretrain_duration_seconds"] = (
                                pretrain_duration
                            )
                            print(
                                f"✅ Found pretrain timing for {pretrain_id}: {pretrain_duration} seconds"
                            )

        except Exception as e:
            # Handle any errors, populating with empty strings
            traceback.print_exc()
            print(f"❌ Error processing row {index}: {e}")
            df.at[index, "execution/run_id"] = ""
            df.at[index, "execution/metric_file"] = ""
            df.at[index, "execution/run_file"] = ""

            if args.include_timing:
                df.at[index, "execution/pretrain_duration_seconds"] = ""
                df.at[index, "execution/task_duration_seconds"] = ""
                df.at[index, "execution/timing_file"] = ""
                df.at[index, "execution/pretrain_id"] = ""
                df.at[index, "execution/total_duration_seconds"] = ""

    # Reorganize columns
    metric_columns = [col for col in df.columns if "metric/" in col]

    # Base columns
    base_columns = [
        "execution/id",
        "execution/uid",
        "backbone/load_from_uid",
        "execution/status",
        "execution/num_deps",
        "ckpt/resume",
        "model/uid",
        "model/config",
        "model/name",
        "model/override_id",
        "data/uid",
        "data/data_module",
        "data/view",
        "data/dataset",
        "data/partition",
        "data/name",
        "data/override_id",
        "pipeline/uid",
        "pipeline/task",
        "pipeline/name",
        "pipeline/override_id",
        "execution/root_dir",
        "execution/run_id",
        "execution/metric_file",
        "execution/run_file",
    ]

    # Add timing columns if included
    if args.include_timing:
        timing_columns = [
            "execution/pretrain_id",
            "execution/pretrain_duration_seconds",
            "execution/task_duration_seconds",
            "execution/timing_file",
            "execution/total_duration_seconds",
        ]
        base_columns.extend(timing_columns)

    # Combine all columns
    all_columns = base_columns + metric_columns
    # Only include columns that actually exist in the dataframe
    existing_columns = [col for col in all_columns if col in df.columns]
    df = df[existing_columns]

    # Calculate total time for finetune/evaluate tasks
    if args.include_timing:
        for index, row in df.iterrows():
            pretrain_time = row.get("execution/pretrain_duration_seconds")
            task_time = row.get("execution/task_duration_seconds")

            total_time = None
            if (
                pd.notna(pretrain_time)
                and pretrain_time != ""
                and pd.notna(task_time)
                and task_time != ""
            ):
                try:
                    total_time = float(pretrain_time) + float(task_time)
                except:
                    pass
            elif pd.notna(task_time) and task_time != "":
                try:
                    total_time = float(task_time)
                except:
                    pass
            elif pd.notna(pretrain_time) and pretrain_time != "":
                try:
                    total_time = float(pretrain_time)
                except:
                    pass

            if total_time is not None:
                df.at[index, "execution/total_duration_seconds"] = total_time

    if args.output_csv:
        df.to_csv(args.output_csv, index=False)
        print(f"✅ Updated CSV saved to {args.output_csv}")

        # Print timing summary
        if args.include_timing:
            print("\n" + "=" * 80)
            print("TIMING SUMMARY")
            print("=" * 80)

            # Filter rows with timing data
            timing_rows = df[
                (df["execution/task_duration_seconds"] != "")
                | (df["execution/pretrain_duration_seconds"] != "")
            ]

            if not timing_rows.empty:
                print(f"\nFound timing data for {len(timing_rows)} executions")

                # Group by task type
                for task_type in timing_rows["pipeline/task"].unique():
                    if pd.isna(task_type):
                        continue

                    task_df = timing_rows[timing_rows["pipeline/task"] == task_type]
                    print(f"\n{'='*40}")
                    print(f"{task_type.upper()} TASKS ({len(task_df)} executions):")
                    print("=" * 40)

                    total_task_time = 0
                    total_pretrain_time = 0
                    total_combined_time = 0

                    for idx, row in task_df.iterrows():
                        execution_id = row.get("execution/id", "")
                        pretrain_time = row.get(
                            "execution/pretrain_duration_seconds", ""
                        )
                        task_time = row.get("execution/task_duration_seconds", "")
                        total_time = row.get("execution/total_duration_seconds", "")
                        pretrain_id = row.get("execution/pretrain_id", "")

                        # Update totals
                        if task_time and task_time != "":
                            try:
                                total_task_time += float(task_time)
                            except:
                                pass
                        if pretrain_time and pretrain_time != "":
                            try:
                                total_pretrain_time += float(pretrain_time)
                            except:
                                pass
                        if total_time and total_time != "":
                            try:
                                total_combined_time += float(total_time)
                            except:
                                pass

                        timing_info = []
                        if pretrain_id and pretrain_id != "":
                            timing_info.append(f"Pretrain: {pretrain_id}")
                        if pretrain_time and pretrain_time != "":
                            try:
                                timing_info.append(
                                    f"Pretrain: {float(pretrain_time):.1f}s"
                                )
                            except:
                                pass
                        if task_time and task_time != "":
                            try:
                                timing_info.append(f"Task: {float(task_time):.1f}s")
                            except:
                                pass
                        if total_time and total_time != "":
                            try:
                                timing_info.append(f"Total: {float(total_time):.1f}s")
                            except:
                                pass

                        if timing_info:
                            print(f"  {execution_id}: {', '.join(timing_info)}")

                    # Print totals for this task type
                    print(f"\n📊 TOTALS for {task_type}:")
                    if total_pretrain_time > 0:
                        print(f"  Total pretrain time: {total_pretrain_time:.1f}s")
                    if total_task_time > 0:
                        print(f"  Total task time: {total_task_time:.1f}s")
                    if total_combined_time > 0:
                        print(f"  Total combined time: {total_combined_time:.1f}s")

                    # Calculate averages
                    count = len(task_df)
                    if count > 0:
                        if total_task_time > 0:
                            print(f"  Average task time: {total_task_time/count:.1f}s")
                        if total_combined_time > 0:
                            print(
                                f"  Average total time: {total_combined_time/count:.1f}s"
                            )
            else:
                print("❌ No timing data found in any execution!")

    else:
        print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
