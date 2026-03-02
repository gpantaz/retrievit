import csv
import json
from pathlib import Path
from typing import Any


def json_serializer(object_to_serialize: Any) -> str:
    """Serialize an object."""
    return str(object_to_serialize)


def read_csv(input_file_path: str | Path, delimiter: str = ",") -> list[Any]:
    """Read the groundtruth csv file."""
    with open(input_file_path) as f:
        spamreader = csv.reader(f, delimiter=delimiter)
        lines = list(spamreader)
    return lines


def write_json(data: Any, output_file_path: str | Path, indent: int | None = None) -> None:
    """Write the data to a json file."""
    with open(output_file_path, "w") as fp:
        json.dump(data, fp, indent=indent)


def read_json(input_file_path: str | Path) -> Any:
    """Read the json file."""
    with open(input_file_path) as fp:
        data = json.load(fp)
    return data


def read_jsonl(input_file_path: str | Path) -> list[Any]:
    """Read the jsonl file."""
    with open(input_file_path) as f:
        return [json.loads(line) for line in f]


def write_jsonl(data: list[Any], output_file_path: str | Path) -> None:
    """Write the data to a jsonl file."""
    with open(output_file_path, "w") as fp:
        for item in data:
            fp.write(f"{json.dumps(item)}\n")
