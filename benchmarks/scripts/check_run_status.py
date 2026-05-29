import pandas as pd
from pathlib import Path
import yaml
import argparse

def check_run_status(result_path, output_path=None):
    df = pd.read_csv(result_path).fillna("")
    for row_id, row in df.iterrows():
        # Skip completed executions
        if row["execution/status"] == "completed":
            continue

        # Iterate over non completed and check if the last run was successful
        try:
            # Get the last run file (any files with run_*.yaml pattern inside root_dir folder)
            exec_dir = Path(row["execution/root_dir"])
            if not exec_dir.exists():
                continue

            run_files = sorted(
                exec_dir.glob("run_*.yaml"),
                key=lambda f: f.stat().st_mtime,
                reverse=False,
            )
            if not run_files:
                continue
            # Load status from the last found run file
            result = yaml.load(run_files[-1].read_text(), Loader=yaml.Loader)
            if len(result["runs"]) == 0:
                continue
            last_run = result["runs"][-1]
            last_run_status = last_run["status"]

            # If the last run was not successful, skip, else set status to completed
            if last_run_status != "SUCCESS":
                continue
            else:
                print(f"Setting status to completed for {exec_dir}")
                df.at[row_id, "execution/status"] = "completed"
        except yaml.error.YAMLError as e:
            print(f"Error parsing file: {e}")
            continue

    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Done. Check the file {output_path} for the updated status.")
    else:
        print(df.to_markdown())

def main():
    parser = argparse.ArgumentParser(description="Check and update run status.")
    parser.add_argument(
        "result_path",
        type=str,
        help="Path to the CSV file containing execution statuses."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path to save the updated CSV file. If None, print on screen",
        default=None,
        required=False,
    )
    args = parser.parse_args()
    check_run_status(Path(args.result_path), args.output_path)

if __name__ == "__main__":
    main()
