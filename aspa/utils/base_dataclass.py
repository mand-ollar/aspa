from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BaseDataClass:
    """A base ModelParams class to filter out extra attributes during initialization."""

    def update_from_dict(self, **kwargs: Any) -> None:
        """Update existing attributes dynamically."""

        valid_fields = {f.name for f in fields(self)}
        invalid_fields: list[str] = []
        for key, value in kwargs.items():
            if key in valid_fields:
                setattr(self, key, value)
            else:
                invalid_fields.append(key)

        if invalid_fields:
            print(f"Warning: The following fields are not valid and will be ignored:\n{', '.join(invalid_fields)}")

    def update_from_yaml(self, path: str | Path) -> None:
        """Update existing attributes dynamically from a YAML file."""

        with open(file=path, mode="r") as file:
            data: dict[str, dict] = yaml.safe_load(file)
            for k, v in data.items():
                self.update_from_dict(**v)

    def update_from_object(self, obj: object) -> None:
        """Update existing attributes dynamically."""
        attributes: list[str] = list(obj.__annotations__.keys())
        update_dict: dict[str, Any] = {k: v for k, v in obj.__dict__.items() if k in attributes}

        self.update_from_dict(**update_dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the dataclass to a dictionary."""
        return asdict(self)
