import hashlib
import logging
from abc import abstractmethod
from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, ClassVar, Dict, List, Optional, Protocol

import networkx as nx
import pandas as pd
import pandasql as ps
import graphviz
import tqdm

from minerva.pipelines.base import Pipeline
from minerva.utils.typing import PathLike

# Configure logging
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


class DataclassType(Protocol):
    # as already noted in comments, checking for this attribute is currently
    # the most reliable way to ascertain that something is a dataclass
    __dataclass_fields__: ClassVar[Dict[str, Any]]


# Configure logging
class MissingColumnError(Exception):
    """Raised when a column is missing in a DataFrame."""

    pass


def hash_str(s: str, preffix: str = 'id_') -> str:
    """Computes the MD5 hash of a string and returns the first 8 characters.

    Parameters
    ----------
    s : str
        The input string to hash.

    Returns
    -------
    str
        The first 8 characters of the MD5 hash.
    """
    return preffix + hashlib.md5(s.encode()).hexdigest()[:12]


def hash_file(file_path: PathLike) -> str:
    """Computes the MD5 hash of a file's contents.

    Parameters
    ----------
    file_path : PathLike
        Path to the file to hash.

    Returns
    -------
    str
        The first 8 characters of the MD5 hash of the file's contents.
    """
    file_path = Path(file_path)
    with file_path.open("rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:12]


class ConfigDatabase:
    """Base class to manage configurations, supporting overrides and conversion
    between DataFrame and schema.

    Attributes
    ----------
    base_path : Path
        The base directory path where configuration files are stored.
    overrides_path : Optional[Path]
        The path to the CSV file containing overrides, if any.
    schema_cls : type
        The dataclass type used to represent the schema of the configurations.
    """

    schema_cls: type = object
    overrides_csv_columns: List[str] = []

    def __init__(
        self, base_path: PathLike, overrides_path: Optional[PathLike] = None
    ) -> None:
        """
        Parameters
        ----------
        base_path : PathLike
            The base directory path where configuration files are stored.
        overrides_path : Optional[PathLike], optional
            The path to the CSV file containing overrides, if any, by default
            None.
        """
        self.base_path = Path(base_path)
        self.overrides_path = Path(overrides_path) if overrides_path else None

    @abstractmethod
    def get_df(self, **kwargs) -> pd.DataFrame:
        """Returns configurations as a pandas DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame containing all the configurations.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in a subclass.
        """
        raise NotImplementedError

    def _load_overrides(self) -> pd.DataFrame:
        """Loads the overrides from the CSV file or returns an empty DataFrame
        with expected overrides columns.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the overrides. Returns an empty DataFrame
            if the file is not found or is empty.

        Raises
        ------
        MissingColumnError
            If the overrides file is missing columns.
        """
        if self.overrides_path:
            try:
                overrides = pd.read_csv(self.overrides_path)
                missing_columns = set(self.overrides_csv_columns) - set(
                    overrides.columns
                )
                if missing_columns:
                    raise MissingColumnError(
                        f"Overrides file at '{self.overrides_path}' is missing columns: {missing_columns}"
                    )
                return overrides
            except pd.errors.EmptyDataError:
                logging.warning(
                    f"Overrides file {self.overrides_path} is empty. Ignoring..."
                )
        return pd.DataFrame(columns=self.overrides_csv_columns)

    def _find_applicable_overrides(
        self, overrides: pd.DataFrame, config_info: Dict[str, str]
    ) -> pd.DataFrame:
        """Finds overrides that apply to the given configuration using wildcard
        support.

        Parameters
        ----------
        overrides : pd.DataFrame
            DataFrame containing all available overrides, based on the
            overrides CSV file. If empty, a default override is created named
            'no_override' with an empty string as the override value.
        config_info : Dict[str, str]
            Dictionary containing configuration details.

        Returns
        -------
        pd.DataFrame
            DataFrame of applicable overrides.
        """
        mask = pd.Series([True] * len(overrides))
        for key, value in config_info.items():
            if key in overrides.columns:
                mask &= overrides[key].isin([value, "*"])
        applicable_overrides = overrides[mask]
        if applicable_overrides.empty:
            applicable_overrides = pd.DataFrame(
                [{"overrides": "", "override_id": "no_override"}]
            )
        return applicable_overrides

    def _extract_config_info(
        self, file_path: Path, hierarchy: List[str]
    ) -> Dict[str, str]:
        """Extracts configuration information from the directory structure
        based on a specified hierarchy.

        Parameters
        ----------
        file_path : Path
            Path to the configuration file.
        hierarchy : List[str]
            List of directory levels to extract information from (in reverse
            order, that is, from the file to the base directory).

        Returns
        -------
        Dict[str, str]
            A dictionary containing configuration information.
        """
        info = {}
        current = file_path
        for key in hierarchy:
            current = current.parent
            info[key] = current.name
        info["name"] = file_path.stem
        return info


class DataModuleDatabase(ConfigDatabase):
    """Manages data modules and configurations, applying overrides based on the
    data modules directory structure and an optional overrides CSV file.

    Directory structure:
    data_modules/
    └── multimodal_df
        └── daghar_standardized_balanced
            ├── kuhar
            │   ├── test
            │   │   └── config_0.yaml
            │   └── train
            │       └── config_0.yaml
            └── motionsense
                ├── test
                │   └── config_0.yaml
                └── train
                    └── config_0.yaml

    An optional overrides CSV file extends configurations:
    | override_id | data_module   | view                         | dataset | partition | name     | overrides          |
    |-------------|---------------|------------------------------|---------|-----------|----------|--------------------|
    | perc_100    | multimodal_df | daghar_standardized_balanced | kuhar   | test      | config_0 | "percentage 100"   |
    | perc_1      | multimodal_df | daghar_standardized_balanced | *       | train     | config_0 | "percentage 1"     |

    Wildcards can be used in the CSV columns (e.g., `*` for `dataset`) to apply
    overrides to multiple configurations. Each configuration has a unique `uid`
    generated based on its parameters and file hash.


    Attributes
    ----------
    schema_cls : type
        The dataclass type used to represent the schema of the data module configurations.
    overrides_csv_columns : List[str]
        List of expected columns in the overrides CSV file.
    """

    overrides_csv_columns = [
        "override_id",
        "data_module",
        "view",
        "dataset",
        "partition",
        "name",
        "overrides",
    ]

    def get_df(self, **kwargs) -> pd.DataFrame:
        """Generates a DataFrame containing data module configurations.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the configurations for all data modules.
        """
        dms = []
        overrides = self._load_overrides()

        for ds_config in self.base_path.rglob("*.yaml"):
            file_hash = hash_file(ds_config)
            config_info = self._extract_config_info(
                ds_config, ["partition", "dataset", "view", "data_module"]
            )

            # Base config ID is path to file (without extension)
            base_config_id = f"{config_info['data_module']}/{config_info['view']}/{config_info['dataset']}/{config_info['partition']}/{config_info['name']}"

            # Find applicable overrides. An "no_override" override is returned
            # if no overrides are found
            applicable_overrides = self._find_applicable_overrides(
                overrides, config_info
            )

            for _, override in applicable_overrides.iterrows():
                override_id = override["override_id"]
                override_value = override.get("overrides", "")
                override_value_hash = hash_str(override_value)
                # ID: data_module/view/dataset/partition/name/override_id
                config_id = f"{base_config_id}/{override_id}"
                # UID: HASH(data_module/view/dataset/partition/name/override_id/file_hash/override_str_hash)
                config_uid = hash_str(
                    f"{config_id}/{file_hash}/{override_value_hash}"
                )

                entry = {
                    "id": config_id,
                    "uid": config_uid,
                    "data_module": config_info["data_module"],
                    "view": config_info["view"],
                    "dataset": config_info["dataset"],
                    "partition": config_info["partition"],
                    "name": config_info["name"],
                    "override_id": override_id,
                    "overrides": override_value,
                    "file_hash": file_hash,
                    "override_str_hash": override_value_hash,
                    "file_path": str(ds_config),
                }

                dms.append(entry)

        df = pd.DataFrame(dms).sort_values("id").reset_index(drop=True)
        return df


class PipelineDatabase(ConfigDatabase):
    """Manages pipeline configurations, supporting overrides based on a
    directory structure and optional overrides CSV file.

    Directory structure:
    pipelines/
    ├── <task>/
    │   ├── <pipeline_name>.yaml
    │   ├── ...
    └── ...

    An optional overrides CSV file can extend configurations:
    | override_id | task       | name               | overrides       |
    |-------------|------------|--------------------|-----------------|
    | override_1  | train      | train_pipeline_1   | "--option 1"    |
    | global_opt  | *          | *                  | "--global_opt"  |

    The class will generate unique configuration entries for each pipeline file
    based on specified overrides.
    """

    overrides_csv_columns = [
        "override_id",
        "task",
        "name",
        "overrides",
    ]

    def get_df(self, **kwargs) -> pd.DataFrame:
        """Generates a DataFrame containing pipeline configurations.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the configurations for all pipelines.
        """
        pipelines = []
        overrides = self._load_overrides()

        for pipeline_config in self.base_path.rglob("*.yaml"):
            file_hash = hash_file(pipeline_config)
            config_info = self._extract_config_info(pipeline_config, ["task"])

            # Base config ID is path to file (without extension)
            base_config_id = f"{config_info['task']}/{config_info['name']}"

            # Find applicable overrides. An "no_override" override is returned
            # if no overrides are found
            applicable_overrides = self._find_applicable_overrides(
                overrides, config_info
            )

            for _, override in applicable_overrides.iterrows():
                override_id = override["override_id"]
                override_value = override.get("overrides", "")
                override_value_hash = hash_str(override_value)
                # ID: task/name/override_id
                config_id = f"{base_config_id}/{override_id}"
                # UID: HASH(task/name/override_id/file_hash/override_str_hash)
                uid = hash_str(
                    f"{base_config_id}/{override_id}/{file_hash}/{override_value_hash}"
                )

                entry = {
                    "id": config_id,
                    "uid": uid,
                    "task": config_info["task"],
                    "name": config_info["name"],
                    "override_id": override_id,
                    "overrides": override_value,
                    "file_hash": file_hash,
                    "override_str_hash": override_value_hash,
                    "file_path": str(pipeline_config),
                }
                pipelines.append(entry)

        df = pd.DataFrame(pipelines).sort_values("id").reset_index(drop=True)
        return df


class ModelDatabase(ConfigDatabase):
    """Manages model configurations, supporting overrides based on a
    directory structure and optional overrides CSV file.

    Directory structure:
    models/
    ├── <config>/
    │   ├── <model_name>.yaml
    │   ├── ...
    └── ...

    An optional overrides CSV file can extend configurations:
    | override_id | config    | name           | overrides       |
    |-------------|-----------|----------------|-----------------|
    | override_1  | config_1  | model_1        | "option 1"      |
    | global_opt  | *         | *              | "global_opt"    |

    The class will generate unique configuration entries for each model file
    based on specified overrides.
    """

    overrides_csv_columns = [
        "override_id",
        "config",
        "name",
        "overrides",
    ]

    def get_df(self, **kwargs) -> pd.DataFrame:
        """Generates a DataFrame containing model configurations.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the configurations for all models.
        """
        models = []
        overrides = self._load_overrides()

        for model_config in self.base_path.rglob("*.yaml"):
            file_hash = hash_file(model_config)
            config_info = self._extract_config_info(model_config, ["config"])

            # Base config ID is path to file (without extension)
            base_config_id = f"{config_info['config']}/{config_info['name']}"

            # Find applicable overrides. An "no_override" override is returned
            # if no overrides are found
            applicable_overrides = self._find_applicable_overrides(
                overrides, config_info
            )

            for _, override in applicable_overrides.iterrows():
                override_id = override["override_id"]
                override_value = override.get("overrides", "")
                override_value_hash = hash_str(override_value)
                # ID: config/name/override_id
                config_id = f"{base_config_id}/{override_id}"
                # UID: HASH(config/name/override_id/file_hash/override_str_hash)
                uid = hash_str(
                    f"{base_config_id}/{override_id}/{file_hash}/{override_value_hash}"
                )

                entry = {
                    "id": config_id,
                    "uid": uid,
                    "config": config_info["config"],
                    "name": config_info["name"],
                    "override_id": override_id,
                    "overrides": override_value,
                    "file_hash": file_hash,
                    "override_str_hash": override_value_hash,
                    "file_path": str(model_config),
                }
                models.append(entry)

        df = pd.DataFrame(models).sort_values("id").reset_index(drop=True)
        return df


class ExecutionDatabase(ConfigDatabase):
    """Class to manage executions and generate execution configurations based on a CSV file."""

    execution_csv_columns = [
        "execution/id",
        "execution/run_name",
        "model/config",
        "model/name",
        "model/override_id",
        "data/data_module",
        "data/view",
        "data/dataset",
        "data/partition",
        "data/name",
        "data/override_id",
        "pipeline/task",
        "pipeline/name",
        "pipeline/override_id",
        "backbone/load_from_id",
        "ckpt/resume",
    ]

    column_order = [
        "execution/id",
        "execution/uid",
        "execution/run_name",
        "backbone/load_from_uid",
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
    ]

    def __init__(
        self,
        data_module_database: DataModuleDatabase,
        pipeline_database: PipelineDatabase,
        model_database: ModelDatabase,
        executions_path: PathLike,
        run_name: str = ""
    ) -> None:
        self.data_module_db = data_module_database
        self.pipeline_db = pipeline_database
        self.model_db = model_database
        self.executions_path = Path(executions_path)
        self.run_name = run_name

    # Function to safely wrap column names with special characters
    @staticmethod
    def escape_column(col_name):
        """Escapes column names with special characters by wrapping them in backticks."""
        if "/" in col_name or " " in col_name:  # Add more checks if necessary
            return f"`{col_name}`"
        return col_name

    # Function to replace variables with $xxxx format using kwargs
    @staticmethod
    def replace_variables(query, query_variables):
        """
        Replaces variables in the query (prefixed by $) with their values from kwargs.

        Parameters:
        query (str): The SQL query to process.
        query_variables (dict): Dictionary of variables to replace in the query.

        Returns:
        str: Query with all $xxxx variables replaced by their values from kwargs.
        """
        # Find all variables in the query (that start with $)
        variables_in_query = re.findall(r"\$\w+", query)
        # print(f"Variables in query '{query}': {variables_in_query}")

        # Replace each variable with the value from query_variables (removing the $)
        for var in variables_in_query:
            var_name = var[1:]  # Remove the $ prefix to get the variable name
            if var_name in query_variables:
                value = query_variables[var_name]
                if isinstance(value, str):
                    value = f"'{value}'"  # Ensure string values are wrapped in quotes for SQL
                if isinstance(value, list):
                    value = tuple(value)
                query = query.replace(var, str(value))
            else:
                raise ValueError(
                    f"Variable '{var_name}' not found in query_variables."
                )

        return query

    def expand_df_wildcards(
        self,
        experiment_df: pd.DataFrame,
        data_modules_df: pd.DataFrame,
        models_df: pd.DataFrame,
        pipelines_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Expands the DataFrame based on wildcards in the given columns."""

        executions = []
        tables = {
            "data": data_modules_df,
            "model": models_df,
            "pipeline": pipelines_df,
        }

        for row_idx, row in experiment_df.iterrows():
            where_clauses = []
            for column in [
                "data/data_module",
                "data/view",
                "data/dataset",
                "data/partition",
                "data/name",
                "data/override_id",
                "model/config",
                "model/name",
                "model/override_id",
                "pipeline/task",
                "pipeline/name",
                "pipeline/override_id",
            ]:
                value = row[column]
                table, column = column.split("/")
                if value == "*" or value == "":
                    continue
                clauses = []
                for val in value.split("|"):
                    clauses.append(f"{table}.{column} == '{val.strip()}'")
                where_clauses.append("(" + " OR ".join(clauses) + ")")
                
                # where_clauses.append(f'{table}.{column} == "{value}"')

            where_clause = " AND ".join(where_clauses)
            sql_query = f"SELECT data.uid AS 'data.uid', model.uid AS 'model.uid', pipeline.uid AS 'pipeline.uid' FROM data, model, pipeline WHERE {where_clause}"

            # print(f"Row {row_idx}: {sql_query}")
            matching_rows = ps.sqldf(sql_query, tables)
            if matching_rows.empty:  # type: ignore
                # print(" " * 4, "No matching rows")
                continue

            for _, matched_row in matching_rows.iterrows():  # type: ignore
                # TODO do this directly in the SQL query
                matched_model = models_df[
                    models_df["uid"] == matched_row["model.uid"]
                ].iloc[0]
                matched_data = data_modules_df[
                    data_modules_df["uid"] == matched_row["data.uid"]
                ].iloc[0]
                matched_pipeline = pipelines_df[
                    pipelines_df["uid"] == matched_row["pipeline.uid"]
                ].iloc[0]

                row = {
                    "execution/id": row["execution/id"],
                    "execution/run_name": self.run_name,
                    "model/config": matched_model["config"],
                    "model/name": matched_model["name"],
                    "model/override_id": matched_model["override_id"],
                    "data/data_module": matched_data["data_module"],
                    "data/view": matched_data["view"],
                    "data/dataset": matched_data["dataset"],
                    "data/partition": matched_data["partition"],
                    "data/name": matched_data["name"],
                    "data/override_id": matched_data["override_id"],
                    "pipeline/task": matched_pipeline["task"],
                    "pipeline/name": matched_pipeline["name"],
                    "pipeline/override_id": matched_pipeline["override_id"],
                    "backbone/load_from_id": row["backbone/load_from_id"],
                    "ckpt/resume": (str(row["ckpt/resume"]) or "").lower()
                    == "true",
                    "model/uid": matched_row["model.uid"],
                    "data/uid": matched_row["data.uid"],
                    "pipeline/uid": matched_row["pipeline.uid"],
                }

                uid = hash_str(
                    f"{self.run_name}/{row['execution/id']}/{matched_row['model.uid']}/{matched_row['data.uid']}/{matched_row['pipeline.uid']}"
                )
                row["execution/uid"] = uid
                # print(" " * 4, f"Matched row: {row}")
                executions.append(row)

        df = pd.DataFrame(executions)
        df = df.drop_duplicates(subset=["execution/uid"]).reset_index(drop=True)
        # df = df.drop(columns=["execution/uid"])
        # print(f"----- Returned {len(df)} executions -----")
        # print(df.to_markdown())
        return df

    def expand_df_backbones(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        df["execution/num_deps"] = 0

        executions = []
        incremental_df = pd.DataFrame(columns=df.columns)
        not_seen_df = df.copy()

        print("Expanding DataFrame based on dependency conditions...")
        for idx, row in df.iterrows():
            if not not_seen_df.empty:
                not_seen_df = not_seen_df.iloc[1:]

            # print(f"Row {idx}: {row.to_dict()}")
            load_from = row["backbone/load_from_id"].strip()

            if not load_from:
                execution = row.to_dict()
                execution["backbone/load_from_id"] = ""
                execution["backbone/load_from_uid"] = ""
                execution["execution/num_deps"] = 0
                execution["execution/run_name"] = self.run_name
                # Execution UID is based on model, data, pipeline, and backbone UID
                execution["execution/uid"] = hash_str(
                    f"{self.run_name}/{row['execution/id']}/{row['model/uid']}/{row['data/uid']}/{row['pipeline/uid']}/{execution['backbone/load_from_uid']}"
                )
                executions.append(execution)
                # print("+   NO SELECT")
                # print(f"--      {execution}")
                # print()
            else:
                # "this" refers to the current row as a DataFrame (single row)
                this_df = pd.DataFrame([row])
                # "other" refers to all other rows except the current one
                other_df = pd.concat([incremental_df, not_seen_df])

                # Create a dictionary with "this" variables for SQL query
                this_row_vars = {
                    f"{self.escape_column(col)}": row[col] for col in df.columns
                }
                query_variables = {
                    **this_row_vars,
                    **kwargs,
                }

                # Replace variables (like $id) in the 'depends' condition using the query_variables
                query = self.replace_variables(
                    row["backbone/load_from_id"], query_variables
                )

                # Construct the SQL query where 'this' refers to the current row and 'other' to all other rows
                sql_query = f"""SELECT other.* FROM other, this WHERE {query}"""
                combined_env = {"other": other_df, "this": this_df}
                # print(f"+   {sql_query}")
                matching_rows = ps.sqldf(sql_query, combined_env)
                if matching_rows.empty:  # type: ignore
                    # print("-       No matching rows")
                    continue

                for i, (_, matched_row) in enumerate(matching_rows.iterrows()):  # type: ignore
                    execution = row.to_dict()
                    backbone_execution_uid = matched_row["execution/uid"]
                    execution["backbone/load_from_id"] = matched_row[
                        "execution/id"
                    ]
                    execution["execution/run_name"] = self.run_name
                    execution["backbone/load_from_uid"] = backbone_execution_uid
                    execution["execution/uid"] = hash_str(
                        f"{self.run_name}/{row['execution/id']}/{row['model/uid']}/{row['data/uid']}/{row['pipeline/uid']}/{execution['backbone/load_from_uid']}"
                    )
                    execution["execution/num_deps"] = (
                        matched_row["execution/num_deps"] + 1
                    )
                    executions.append(execution)
                    # print(f"--      {execution}")

                # print()

            incremental_df = pd.DataFrame(executions)

        df = pd.DataFrame(executions)
        df = df.drop_duplicates(subset=["execution/uid"])
        df = df.reset_index(drop=True)
        # print(f"----- Returned {len(df)} executions -----")
        return df

    def get_df(self, **kwargs) -> pd.DataFrame:
        """Generates the execution DataFrame based on the executions CSV file."""
        # Read the executions CSV file
        executions_df = pd.read_csv(self.executions_path, dtype=str).fillna("")
        executions_df = self.expand_df_wildcards(
            executions_df,
            self.data_module_db.get_df(),
            self.model_db.get_df(),
            self.pipeline_db.get_df(),
        )
        executions_df = self.expand_df_backbones(executions_df, **kwargs)
        # print("----- Final DataFrame -----")
        # print(executions_df.to_markdown())
        # print("-" * 80)

        executions_df = executions_df[self.column_order]
        executions_df = executions_df.sort_values(
            ["execution/num_deps", "execution/uid"]
        )
        executions_df = executions_df.reset_index(drop=True)
        return executions_df


class LightningPipelineExecutionGenerator(Pipeline):
    command_template = (
        "python -m minerva.pipelines.lightning_pipeline "
        "{seed} "
        "--config '{pipeline_file}' "
        "{pipeline_overrides} "
        "--trainer.logger.save_dir '{output_log_dir}' "
        "--trainer.logger.name '{execution_uid}' "
        "--trainer.logger.version '{version_name}' "
        "--model '{model_file}' "
        "{model_overrides} "
        "{ckpt_backbone} "
        "run "
        "--data '{data_file}' "
        "{data_overrides} "
        "{resume} "
    )

    @dataclass
    class Execution:
        uid: str
        pipeline_file: Path
        pipeline_overrides: Dict[str, str]
        data_file: Path
        data_overrides: Dict[str, str]
        model_file: Path
        model_overrides: Dict[str, str]
        save_dir: Path
        save_name: str
        version_name: str
        backbone_uid: Optional[str] = None
        resume: bool = False
        seed: Optional[int] = None

    def __init__(
        self,
        executions_db: ExecutionDatabase,
        executions_vars: Optional[Dict[str, str]] = None,
        save_dir: PathLike = "experiments/logs",
        seed: Optional[int] = None,
        version_name: str = "final",
        run_name: str = "",
    ):
        super().__init__(save_run_status=False, log_dir=None, seed=seed)
        # Expanded executions DataFrame
        self.executions_db = executions_db
        self.executions_vars = executions_vars or dict()
        self.save_dir = Path(save_dir)
        self.seed = seed
        self.version_name = version_name

    def get_expanded_df(self):
        executions_df = self.executions_db.get_df(**self.executions_vars)
        models_df = self.executions_db.model_db.get_df()
        data_df = self.executions_db.data_module_db.get_df()
        pipelines_df = self.executions_db.pipeline_db.get_df()

        models_df.columns = [f"{col}|model" for col in models_df.columns]
        data_df.columns = [f"{col}|data" for col in data_df.columns]
        pipelines_df.columns = [
            f"{col}|pipeline" for col in pipelines_df.columns
        ]

        expanded_df = executions_df.merge(
            models_df,
            left_on="model/uid",
            right_on="uid|model",
        )

        expanded_df = expanded_df.merge(
            data_df,
            left_on="data/uid",
            right_on="uid|data",
        )

        expanded_df = expanded_df.merge(
            pipelines_df,
            left_on="pipeline/uid",
            right_on="uid|pipeline",
        )

        return expanded_df

    @staticmethod
    def _parse_overrides(overrides: str) -> Dict[str, str]:
        if not overrides:
            return {}

        values = dict()
        for var in overrides.split(" "):
            if "=" not in var:
                raise ValueError(f"Invalid variable format '{var}'")
            if var.count("=") > 1:
                raise ValueError(
                    f"Invalid variable format (multiple '=') '{var}'"
                )

            k, v = var.split("=")
            values[k] = v

        return values

    def _create_execution_cmd(self, execution: Execution):
        seed = f"--seed {execution.seed}" if execution.seed else ""
        pipeline_file = str(execution.pipeline_file.resolve())
        pipeline_overrides_str = " ".join(
            f"--trainer.{k}='{v}'"
            for k, v in execution.pipeline_overrides.items()
        )
        models_file = str(execution.model_file.resolve())
        model_overrides_str = " ".join(
            f"--model.{k}='{v}'" for k, v in execution.model_overrides.items()
        )
        data_file = str(execution.data_file.resolve())
        data_overrides_str = " ".join(
            f"--data.{k}='{v}'" for k, v in execution.data_overrides.items()
        )
        save_dir = str(execution.save_dir.resolve())
        execution_uid = execution.uid
        version_name = execution.version_name
        ckpt_backbone = ""
        resume = ""
        if execution.backbone_uid and not execution.resume:
            ckpt_backbone = f"--model.backbone.ckpt_path '{save_dir}/{execution.backbone_uid}/{execution.version_name}/checkpoints/best.ckpt'"
        if execution.resume:
            resume = f"--ckpt_path '{save_dir}/{execution.backbone_uid}/{execution.version_name}/checkpoints/best.ckpt'"

        cmd = self.command_template.format(
            seed=seed,
            pipeline_file=pipeline_file,
            pipeline_overrides=pipeline_overrides_str,
            model_file=models_file,
            model_overrides=model_overrides_str,
            data_file=data_file,
            data_overrides=data_overrides_str,
            output_log_dir=save_dir,
            execution_uid=execution_uid,
            version_name=version_name,
            ckpt_backbone=ckpt_backbone,
            resume=resume,
        )
        return cmd

    def run(self):
        executions = self.get_expanded_df()
        executions["execution/bash_command"] = ""
        executions["execution/root_dir"] = ""
        executions["execution/status"] = "unknown"
        for row_idx, row in tqdm.tqdm(
            executions.iterrows(),
            desc="Generating commands",
            total=len(executions),
        ):
            e = self.Execution(
                uid=row["execution/uid"],
                pipeline_file=Path(row["file_path|pipeline"]),
                pipeline_overrides=self._parse_overrides(
                    row["overrides|pipeline"]
                ),
                data_file=Path(row["file_path|data"]),
                data_overrides=self._parse_overrides(row["overrides|data"]),
                model_file=Path(row["file_path|model"]),
                model_overrides=self._parse_overrides(row["overrides|model"]),
                save_dir=self.save_dir,
                save_name=row["execution/uid"],
                version_name=self.version_name,
                backbone_uid=row["backbone/load_from_uid"] or None,
                resume=row["ckpt/resume"],
                seed=self.seed,
            )

            if e.resume and not e.backbone_uid:
                raise ValueError(
                    f"Error at execution {e.uid}: Backbone checkpoint is required for resuming training."
                )

            cmd = self._create_execution_cmd(e)
            executions.at[row_idx, "execution/bash_command"] = cmd
            executions.at[row_idx, "execution/root_dir"] = (
                e.save_dir / e.uid / e.version_name
            ).resolve()

        executions = executions[
            [
                "execution/id",
                "execution/uid",
                "execution/run_name",
                "backbone/load_from_uid",
                "execution/status",
                "execution/num_deps",
                "ckpt/resume",
                "model/uid",
                "model/config",
                "model/name",
                "model/override_id",
                "file_path|model",
                "file_hash|model",
                "data/uid",
                "data/data_module",
                "data/view",
                "data/dataset",
                "data/partition",
                "data/name",
                "data/override_id",
                "file_path|data",
                "file_hash|data",
                "pipeline/uid",
                "pipeline/task",
                "pipeline/name",
                "pipeline/override_id",
                "file_path|pipeline",
                "file_hash|pipeline",
                "execution/root_dir",
                "execution/bash_command",
            ]
        ]

        executions = executions.rename(
            columns={
                "file_path|model": "model/file_path",
                "file_hash|model": "model/file_hash",
                "file_path|data": "data/file_path",
                "file_hash|data": "data/file_hash",
                "file_path|pipeline": "pipeline/file_path",
                "file_hash|pipeline": "pipeline/file_hash",
            }
        )

        executions = executions.sort_values(
            ["execution/num_deps", "execution/uid"]
        ).reset_index(drop=True)

        return executions


def render_graph(
    executions_df: pd.DataFrame, output_path: PathLike, format: str = "pdf"
):
    executions_df = executions_df.sort_values(
        by=[
            "model/file_path",
            "model/override_id",
            "data/file_path",
            "data/override_id",
            "pipeline/file_path",
            "pipeline/override_id",
        ]
    )
    dot = graphviz.Digraph(comment="Execution Dependency Graph")
    output_path = Path(output_path)

    for row_idx, row in executions_df.iterrows():
        # label = f"""<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
        #             <TR><TD><B>execution/uid:</B> {row['execution/uid']}</TD></TR>
        #             <TR><TD><B>data/uid:</B> {row['data/uid']}</TD></TR>
        #             <TR><TD><B>model/uid:</B> {row['model/uid']}</TD></TR>
        #             <TR><TD><B>pipeline/uid:</B> {row['pipeline/uid']}</TD></TR>
        #             </TABLE>>"""
        label = f"""<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
                    <TR><TD><B>execution/uid:</B> {row['execution/uid']}</TD></TR>
                    <TR><TD>{row['model/config']}/{row['model/name']}/{row['model/override_id']}</TD></TR>
                    <TR><TD>{row['data/data_module']}/{row['data/view']}/{row['data/dataset']}/{row['data/partition']}/{row['data/name']}/{row['data/override_id']}</TD></TR>
                    <TR><TD>{row['pipeline/task']}/{row['pipeline/name']}/{row['pipeline/override_id']}</TD></TR>
                    </TABLE>>"""
        dot.node(
            row["execution/uid"],
            label=label,
            shape="plaintext",
        )

        if row["backbone/load_from_uid"]:
            edge_label = "resume" if row["ckpt/resume"] else "backbone"
            dot.edge(
                row["execution/uid"],
                row["backbone/load_from_uid"],
                label=edge_label,
            )

    dot.render(str(output_path), format=format, view=False, cleanup=True)
    return dot


def main(
    executions_path: PathLike,
    base_configs_path: PathLike,
    overrides_path: PathLike,
    log_dir: PathLike,
    output_db_file: Optional[PathLike] = None,
    seed: Optional[int] = None,
    version_name: str = "final",
    graph_output_path: Optional[PathLike] = None,
    run_name: str = "",
    append: bool = False,
):
    base_configs_path = Path(base_configs_path)
    executions_path = Path(executions_path)
    output_db_file = Path(output_db_file) if output_db_file else None
    log_dir = Path(log_dir)

    # Create the databases paths
    dm_path = base_configs_path / "data_modules"
    pl_path = base_configs_path / "pipelines"
    model_path = base_configs_path / "models"

    # Check overrides path
    if overrides_path:
        overrides_path = Path(overrides_path)
        dm_overrides_path = overrides_path / "data_modules.csv"
        pl_overrides_path = overrides_path / "pipelines.csv"
        model_overrides_path = overrides_path / "models.csv"
    else:
        dm_overrides_path = None
        pl_overrides_path = None
        model_overrides_path = None

    # Create the databases
    dm_db = DataModuleDatabase(
        base_path=dm_path, overrides_path=dm_overrides_path
    )
    pl_db = PipelineDatabase(
        base_path=pl_path, overrides_path=pl_overrides_path
    )
    model_db = ModelDatabase(
        base_path=model_path, overrides_path=model_overrides_path
    )
    exp_db = ExecutionDatabase(
        data_module_database=dm_db,
        pipeline_database=pl_db,
        model_database=model_db,
        executions_path=executions_path,
        run_name=run_name,
    )
    # Create the execution generator
    exec_gen = LightningPipelineExecutionGenerator(
        executions_db=exp_db,
        executions_vars={},
        save_dir=log_dir,
        seed=seed,
        version_name=version_name,
        run_name=run_name,
    )

    df = exec_gen.run()

    if graph_output_path:
        graph_output_path = Path(graph_output_path)
        output_format = graph_output_path.suffix or "pdf"
        if output_format.startswith("."):
            output_format = output_format[1:]
        render_graph(
            df, graph_output_path.with_suffix(""), format=output_format
        )
        print(
            f"Graph saved to: {graph_output_path.with_suffix('')}.{output_format}"
        )

    if output_db_file:
        if append and output_db_file.exists():
            print(f"Appending to existing database: {output_db_file}")
            existing_df = pd.read_csv(output_db_file)
            
            # Remove existing rows from the new dataframe
            existing_uids = set(existing_df["execution/uid"])
            df = df[~df["execution/uid"].isin(existing_uids)]
            
            print(f"Existing executions: {len(existing_df)}. It will be untouched!")
            if len(df) == 0:
                print(f"No new executions to append! CSV at '{output_db_file}' remain unchanged.")
                return df
            else:
                # Remove columns from existing_df that are not in df
                # existing_df = existing_df[df.columns]
                
                # Concatenate the existing and new dataframes
                df = pd.concat([existing_df, df], ignore_index=True)
                df = df.sort_values([
                    "execution/num_deps",
                    "execution/uid",
                    "execution/run_name"
                ])
                df = df.reset_index(drop=True)

        if output_db_file.suffix == ".csv":
            df.to_csv(output_db_file, index=False)
        elif output_db_file.suffix == ".xlsx":
            df.to_excel(output_db_file, index=False)
        elif output_db_file.suffix == ".md":
            df.to_markdown(output_db_file, index=False)
        else:
            raise ValueError(
                f"Unsupported output file format: {output_db_file.suffix}"
            )

        print(f"Number of executions: {len(df)}. Saved to: {output_db_file}")
    else:
        print(df.to_markdown())

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate executions and optionally plot graphs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--executions_path",
        type=str,
        required=True,
        help="Path to the executions CSV file.",
    )
    parser.add_argument(
        "--base_configs_path",
        type=str,
        required=True,
        help="Path to the base configurations.",
    )
    parser.add_argument(
        "--overrides_path",
        type=str,
        help="Path to the overrides CSV files.",
        required=False,
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="experiments/logs",
        help="Directory to save logs.",
    )
    parser.add_argument(
        "--db_file",
        type=str,
        default=None,
        help="Path to save the generated executions.",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed for generation."
    )
    parser.add_argument(
        "--version_name",
        type=str,
        default="final",
        help="Version name for execution logs.",
    )
    parser.add_argument(
        "--graph_output_path",
        type=str,
        help="Path to save the generated graph (if provided, will generate a graph).",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        help="name of the run. Useful for running multiple times",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to the existing database file.",
    )

    args = parser.parse_args()

    main(
        executions_path=args.executions_path,
        base_configs_path=args.base_configs_path,
        overrides_path=args.overrides_path,
        log_dir=args.log_dir,
        output_db_file=args.db_file,
        seed=args.seed,
        version_name=args.version_name,
        graph_output_path=args.graph_output_path,
        run_name=args.run_name,
        append=args.append,
    )

# Example command
"""
export BASE_CONFIGS_PATH="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/base_configs/"     # Path to the base configurations
export MY_EXPERIMENT_DIR="/workspaces/HIAAC-KR-Dev-Container/Minerva-Exps/benchmarks/experiments/example/"        # Path to the experiment directory
export EXECUTIONS_PATH="${MY_EXPERIMENT_DIR}/configs/generated_executions.csv"                          # Path to save the generated executions

python execution_planner.py  \
    --executions_path ${MY_EXPERIMENT_DIR}/configs/experiments.csv \
    --base_configs_path ${BASE_CONFIGS_PATH}   \
    --overrides_path ${MY_EXPERIMENT_DIR}/configs/overrides   \
    --db_file ${EXECUTIONS_PATH}  \
    --graph_output_path ${MY_EXPERIMENT_DIR}/configs/generated_executions_graph \
    --log_dir ${MY_EXPERIMENT_DIR}/logs  \
    --seed 42  
"""