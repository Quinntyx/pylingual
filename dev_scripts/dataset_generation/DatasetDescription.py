import json
import logging
import pathlib
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union


def get_dataset_description_from_arg_json(json_path: str, logger: Union[logging.Logger, None] = None) -> "DatasetDescription":
    json_file_path = pathlib.Path(json_path)

    if not json_file_path.exists():
        raise FileNotFoundError(f"{json_file_path} does not exist")

    if logger:
        logger.info(f"Loading dataset description from {json_file_path}...")

    with json_file_path.open() as json_file:
        dataset_description_dict = json.load(json_file)

    dataset_description_dict["data_requests"] = [DataRequest(**d) for d in dataset_description_dict["data_requests"]]
    if "collect" in dataset_description_dict:
        dataset_description_dict["collect"] = CollectDescription(**dataset_description_dict["collect"])
    return DatasetDescription(**dataset_description_dict)


@dataclass
class DataRequest:
    name: str
    source_path: pathlib.Path
    num_train: int
    num_test: int
    num_valid: int
    corpus: Optional[str] = None  # NAS corpus (pypi or csn) the collect stage fills source_path from

    @property
    def total_files(self):
        return self.num_train + self.num_test + self.num_valid

    def __post_init__(self):
        self.source_path = pathlib.Path(self.source_path)
        if not self.source_path.exists() and self.corpus is None:
            raise FileNotFoundError(f"{self.source_path} for DataRequest {self.name} does not exist")

        if self.num_train < 0:
            raise ValueError(f"Training sample count for DataRequest {self.name} must be non-negative")
        if self.num_test < 0:
            raise ValueError(f"Testing sample count for DataRequest {self.name} must be non-negative")
        if self.num_valid < 0:
            raise ValueError(f"Validation sample count for DataRequest {self.name} must be non-negative")


@dataclass
class CollectDescription:
    """Where the collect stage samples the sources of the data requests with a corpus (see collect_sources.py)."""

    nas_root: pathlib.Path
    index: pathlib.Path  # cache of the NAS index, built on first use
    exclude: List[dict]  # eval dataset releases to keep out: {"repo": git URL, "ref": tag}
    seed: int
    min_bytes: int = 100
    max_bytes: int = 100_000

    def __post_init__(self):
        self.nas_root = pathlib.Path(self.nas_root)
        self.index = pathlib.Path(self.index)


@dataclass
class DatasetDescription:
    name: str
    version: Tuple[int, int]
    save_to_dir: pathlib.Path
    huggingface_user: str
    data_requests: List[DataRequest]
    collect: Optional[CollectDescription] = None

    @property
    def code_dir(self):
        return self.save_to_dir / self.name / "code"

    @property
    def csv_dir(self):
        return self.save_to_dir / self.name / "csv"

    @property
    def cflow_dir(self):
        return self.save_to_dir / self.name / "cflow"

    @property
    def collect_dir(self):
        return self.save_to_dir / self.name / "collect"

    def __post_init__(self):
        self.save_to_dir = pathlib.Path(self.save_to_dir)
        self.version = tuple(self.version)
