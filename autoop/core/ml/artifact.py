from pydantic import BaseModel, Field
from typing import Any
import pickle


class Artifact(BaseModel):
    """
    A class to represent stored data (datasets, models, etc.).

    Attributes:
        name: Name of the artifact
        data: Binary data of the artifact
    """
    name: str = Field(..., description="Name of the artifact")
    data: bytes = Field(..., description="Binary data of the artifact")

    def read(self) -> bytes:
        """Read the raw binary data of the artifact."""
        return self.data

    def encode(self, obj: Any) -> bytes:
        """Encode an object into bytes for storage."""
        return pickle.dumps(obj)

    def decode(self, data: bytes) -> Any:
        """Decode bytes back into an object."""
        return pickle.loads(data)

    class Config:
        arbitrary_types_allowed = True
