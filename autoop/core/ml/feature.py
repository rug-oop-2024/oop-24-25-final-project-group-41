
from pydantic import BaseModel, Field
from typing import Literal
# import numpy as np

# from autoop.core.ml.dataset import Dataset


class Feature(BaseModel):
    """A class to represent a feature in a dataset.

    Attributes:
        name (str): The name of the feature (column name in dataset)
        type (str): The type of the feature ('categorical' or 'numerical')
    """
    name: str = Field(..., description="Name of the feature")
    type: Literal["categorical", "numerical"] = Field(
        ..., description="Type of the feature")

    def __str__(self):
        """String representation of the feature."""
        return f"Feature(name='{self.name}', type='{self.type}')"
