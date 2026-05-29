import argparse
import fcntl
import subprocess
from pathlib import Path
from typing import Dict,Iterable
import pandas as pd
import ray
from dataclasses import dataclass
from minerva.utils.typing import PathLike
import shutil
from tempfile import NamedTemporaryFile
import pandasql as ps
import sqlitedict
# stdlib only; no sqlite / sqlitedict
import base64
import json
import os
import uuid
import io
import errno

# ========= Internal helpers (NFS-safe primitives) =========

def _as_path(p: PathLike) -> Path:
    return Path(p)

def _table_dir(db_file: PathLike, table_name: str) -> Path:
    """
    Represent the "database" as a directory named exactly `db_file`,
    with one subdirectory per table. This keeps the function signature
    unchanged while switching the on-disk format.
    """
    root = _as_path(db_file)
    tbl = root / table_name
    tbl.mkdir(parents=True, exist_ok=True)
    return tbl

def _fsync_dir(path: Path) -> None:
    """
    Fsync the directory to persist metadata (file creation/rename) across crashes.
    This is important on NFS: rename is atomic, but persisting it benefits from a dir fsync.
    """
    try:
        fd = os.open(str(path), os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except (PermissionError, NotADirectoryError, FileNotFoundError):
        # Best effort; some environments may not allow directory fsync.
        pass

def _encode_key(key: str) -> str:
    """
    File-name-safe, reversible encoding (URL-safe base64 without padding).
    Guarantees no '/' or special chars, so it's safe on NFS.
    """
    b = base64.urlsafe_b64encode(key.encode("utf-8")).decode("ascii")
    return b.rstrip("=")

def _decode_key(encoded: str) -> str:
    pad = "=" * (-len(encoded) % 4)
    return base64.urlsafe_b64decode((encoded + pad).encode("ascii")).decode("utf-8")

def _record_path(table_dir: Path, index: str) -> Path:
    return table_dir / f"{_encode_key(index)}.json"

def _atomic_write_json(target_path: Path, payload: dict) -> None:
    """
    NFSv3-safe write:
      1) write to a temp file in the same directory
      2) fsync the temp file
      3) os.replace(temp, target) (atomic rename)
      4) fsync the directory
    Multiple writers to different keys never contend. Writers to the same key
    will "last writer wins" atomically.
    """
    table_dir = target_path.parent
    tmp = table_dir / f".tmp-{uuid.uuid4().hex}.json"

    # Write JSON bytes in one go to avoid partials
    data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")

    # Use low-level os.open to control flags on NFS
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with io.FileIO(fd, "wb", closefd=False) as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
    finally:
        os.close(fd)

    os.replace(str(tmp), str(target_path))  # atomic
    _fsync_dir(table_dir)

def _read_json(path: Path) -> dict:
    with open(path, "rb") as f:
        return json.loads(f.read().decode("utf-8"))

def _list_records(table_dir: Path) -> Iterable[Path]:
    # Only .json records; avoids temp files and foreign files
    return (p for p in table_dir.iterdir() if p.is_file() and p.suffix == ".json")


# ========= Public API (same signatures, same return types) =========

def df_to_sqlite(
    df: pd.DataFrame,
    db_file: PathLike,
    table_name: str = "experiments",
    index_col="execution/uid",
):
    """
    Store each row of the given DataFrame in a key-value "table", where the key
    is taken from `row[index_col]` and the value is the entire row as a dict.

    Concurrency/process safety:
      - No sqlite/sqlitedict is used.
      - Data is stored as per-key JSON files under: <db_file>/<table_name>/
      - Each write is an atomic write+rename with directory fsync, which is
        safe on NFSv3 with 'local_lock=none'. Multiple writers to different keys
        never contend; writers to the same key are last-writer-wins atomically.

    Behavior preserved:
      - Same parameters and return type (None).
      - Keys and values equivalent to what sqlitedict would have stored:
        { index -> dict(row) }.

    On-disk format note:
      - The path given by `db_file` is treated as a directory (created if needed).
        This keeps the call signature identical while avoiding sqlite entirely.
    """
    table_dir = _table_dir(db_file, table_name)

    # Mirror original behavior: iterate rows as dicts and use entry[index_col]
    items = df.to_dict(orient="index")
    for _, entry in items.items():
        key = entry[index_col]
        _atomic_write_json(_record_path(table_dir, str(key)), entry)


def sqlite_to_df(
    db_file: PathLike,
    table_name: str = "experiments",
) -> pd.DataFrame:
    """
    Read all values from the "table" into a pandas DataFrame.

    Concurrency/process safety:
      - Values are read from immutable points-in-time (files written via atomic
        rename). Readers see either the old or the new version of a key; never
        a partial write.

    Behavior preserved:
      - Same parameters and return type (pd.DataFrame).
      - Rows correspond to the previously written dict values.

    Returns:
      - Empty DataFrame if the table directory is missing or has no records.
    """
    table_dir = _table_dir(db_file, table_name)
    rows = []
    for p in _list_records(table_dir):
        try:
            rows.append(_read_json(p))
        except json.JSONDecodeError:
            # Extremely rare: if a client crashes between write and rename,
            # a temp file could linger, but we only read *.json targets that
            # were created by atomic rename. If corruption happens, skip.
            continue
        except FileNotFoundError:
            # Raced with a writer replacing the file; safely skip.
            continue

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def write_on_db(
    db_file: PathLike, index: str, data: dict, table_name: str = "experiments"
):
    """
    Write a single key/value to the "table" (last-writer-wins).

    Concurrency/process safety:
      - Atomic write+rename with directory fsync on NFSv3; safe for multiple
        processes/threads without relying on fcntl locks.

    Behavior preserved:
      - Same parameters and return type (None).
      - Equivalent to: db[index] = data in a sqlitedict-backed table.
    """
    table_dir = _table_dir(db_file, table_name)
    _atomic_write_json(_record_path(table_dir, str(index)), data)


def read_from_db(
    db_file: PathLike, index: str, table_name: str = "experiments"
) -> dict:
    """
    Read a single key/value from the "table".

    Concurrency/process safety:
      - Reads are of fully-written files only (thanks to atomic rename). If a
        concurrent writer updates the same key, readers see either the old or
        the new version, never a partial file.

    Behavior preserved:
      - Same parameters and return type (dict).
      - Raises KeyError if the key does not exist, mirroring sqlitedict.

    Raises:
      - KeyError: if `index` is missing.
    """
    table_dir = _table_dir(db_file, table_name)
    path = _record_path(table_dir, str(index))
    try:
        return _read_json(path)
    except FileNotFoundError:
        raise KeyError(index)


def _exp_print(exp, *args, **kwargs):
    print(f"[{exp}] ", *args, **kwargs)


# @ray.remote
def ray_func(
    db_file: PathLike,
    execution_id: str,
    tablename: str = "experiments",
    force: bool = False,
) -> dict:
    def exp_print(*args, **kwargs):
        print(f"[{execution_id}] ", *args, **kwargs)
  
    # Read the row from our JSON-based DB
    row = read_from_db(db_file, execution_id, tablename)
    # Open the database and read the row
    # with sqlitedict.SqliteDict(
    #     filename=db_file, tablename=tablename, autocommit=True
    # ) as db:
    #     row = db[execution_id]

    # execution/status can be (completed, running, failed, unknown)
    if row["execution/status"] == "completed":
        if not force:
            exp_print(f"Skipping completed execution: {execution_id}")
            return row
        exp_print(f"Re-running {execution_id} due to --force flag.")
    elif row["execution/status"] == "running":
        exp_print(f"Skipping running execution: {execution_id}")
        return row

    elif row["execution/status"] == "failed":
        exp_print(f"Retrying failed execution: {execution_id}")

    else:  # unknown and any other state
        exp_print(f"Running new execution: {execution_id}")

    # Set the status to running
    # row["execution/status"] = "running"
    # db[execution_id] = row
    # Set the status to running
    row["execution/status"] = "running"
    write_on_db(db_file, execution_id, row, tablename)


    try:
        # Create the root directory for the experiment
        root_dir = Path(row["execution/root_dir"])
        root_dir.mkdir(parents=True, exist_ok=True)

        # Create the checkpoint directory (remove any existing ckpt)
        ckpt_dir = root_dir / "checkpoints"
        if ckpt_dir.exists():
            exp_print(f"Removing existing checkpoint directory: {ckpt_dir}")
            shutil.rmtree(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Create the log file
        log_file = root_dir / "log.txt"
        
        # Run the experiment
        cmd_to_run = row["execution/bash_command"]
        exp_print(f"Running command: '{cmd_to_run}'")
        exp_print(f"Check log file at: {log_file}")

        # Run the experiment
        with log_file.open("w") as log:
            # Run the command
            process = subprocess.Popen(
                cmd_to_run,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                shell=True,
                bufsize=1,
            )

            # Keep track of the output
            for i, line in enumerate(process.stdout):  # type: ignore
                # print(line, end="")
                log.write(line)
                if i % 100 == 0:
                    log.flush()
                    print(line, end="", flush=True)
                else:
                    print(line, end="", flush=False)

            # Wait for the process to finish
            process.wait()

        # Check the return code
        if process.returncode != 0:
            exp_print(f"Command failed with return code {process.returncode}")
            row["execution/status"] = "failed"
            raise subprocess.CalledProcessError(process.returncode, cmd_to_run)
        else:
            exp_print("Command completed successfully.")
            row["execution/status"] = "completed"
    except Exception as e:
        print(f"Error running experiment {execution_id}: {e}")
        row["execution/status"] = "failed"
        raise e

    finally:
        # write_on_db(db_file, execution_id, row)
        write_on_db(db_file, execution_id, row, tablename)

    return row

def update_csv_from_db(db_file: PathLike, csv_file: PathLike):
    df = sqlite_to_df(db_file)
    df.to_csv(csv_file, index=False)
    print(f"Updated CSV file: {csv_file}")


def main() -> None:
    """
    Main function to orchestrate experiment processing using either multiprocessing or Ray.

    Command line arguments:
    - Experiment directory: the directory containing experiment YAML files.
    - Parallelism control: controls local multiprocessing or Ray GPU fractions.
    - Ray execution options: flag to use Ray and provide Ray cluster address.
    - Checkpoint removal: option to remove the checkpoint directory after job completion.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Process experiment files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "experiments_csv",
        type=str,
        help="CSV file containing experiment configurations",
    )

    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=1,
        help="Number of workers to use for parallel execution (default: 1)",
    )
    parser.add_argument(
        "--use-ray",
        action="store_true",
        help="Use ray cluster for parallel execution.",
    )
    parser.add_argument(
        "--ray-address",
        type=str,
        help="Address of Ray cluster (e.g., <ip>:<port>)",
        default="auto",
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Query to filter experiments to run. Use SQL syntax.",
        default=None,
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force re-running of experiments independent of status",
    )
    
    # Parse the command line arguments
    args = parser.parse_args()
    df = pd.read_csv(args.experiments_csv)
    sqlite_file = Path(args.experiments_csv).with_suffix(".sqlite")

    if sqlite_file.exists():
        print(f"Removing existing DB directory: {sqlite_file}")
        if sqlite_file.is_dir():
            shutil.rmtree(sqlite_file)   # remove directory + contents
        else:
            sqlite_file.unlink()         # safety: remove file if it's not a dir


    print(f"Converting CSV to SQLite: {sqlite_file}...")
    df_to_sqlite(df, sqlite_file)

    futures = {}  # Keep track of futures to handle interrupt
    finished_futures = []

    try:
        # First, filter the experiments using the query
        if args.query:
            sql_query = f"SELECT * FROM experiments WHERE {args.query}"
            df = ps.sqldf(sql_query, {"experiments": df})
            if df is None:
                raise ValueError(f"Query '{args.query}' returned None.")
            print(f"Filtered experiments using query: {sql_query}")
            print(f"Matched {len(df)} rows!")

        if args.use_ray:
            num_gpus = 1 / args.workers
            print(
                f"Connecting to Ray cluster at '{args.ray_address}' with {num_gpus} GPUs per worker."
            )
            if args.ray_address == "auto":
                print("Initializing Ray with default settings.")
                ray.init()
            else:
                ray.init(address=args.ray_address)
            submit_func = ray.remote(num_gpus=num_gpus)(ray_func)
        else:
            # submit_func = multiprocessing_func
            raise NotImplementedError(
                "Local multiprocessing is not implemented yet."
            )

        for priority, group in df.groupby("execution/num_deps"):
            print(f" ---- Processing executions with num/deps={priority} ---- ")
            for _, row in group.iterrows():
                execution_id = row["execution/uid"]
                # Submit the task to Ray
                future = submit_func.options(  # type: ignore
                    name=f"{row['execution/id']}/{execution_id}"
                ).remote(sqlite_file, execution_id, "experiments", args.force)
                futures[execution_id] = future

            # Wait for results and handle any errors
            for execution_id, future in futures.items():
                try:
                    result_row = ray.get(future)
                    status = result_row["execution/status"]  # type: ignore
                    print(f"Experiment {execution_id}: {status.upper()}")
                except Exception as e:
                    print(f"Experiment {execution_id}: FAILED")
                finally:
                    update_csv_from_db(sqlite_file, args.experiments_csv)
                    finished_futures.append(future)
                
    except KeyboardInterrupt:
        print("\nKeyboard Interrupt detected! Cancelling running tasks...")

        # Load all current rows
        df_running = sqlite_to_df(sqlite_file, "experiments")
        for _, row in df_running.iterrows():
            if row["execution/status"] == "running":
                execution_id = row["execution/uid"]
                print(f"Marking execution {execution_id} as failed due to interrupt.")
                row["execution/status"] = "failed"
                # Write back the updated row
                write_on_db(sqlite_file, execution_id, row.to_dict(), "experiments")
        raise

    finally:
        update_csv_from_db(sqlite_file, args.experiments_csv)

        print(f"Removing DB directory: {sqlite_file}")
        if sqlite_file.exists():
            if sqlite_file.is_dir():
                shutil.rmtree(sqlite_file)
            else:
                sqlite_file.unlink()

        if args.use_ray:
            print("Shutting down Ray...")
            ray.shutdown()



if __name__ == "__main__":
    main()
