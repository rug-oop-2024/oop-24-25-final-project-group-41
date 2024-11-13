from typing import List
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature


def detect_feature_types(dataset: Dataset) -> List[Feature]:
    """
    Detect whether each column in the dataset is categorical or numerical.

    Args:
        dataset: Dataset object containing the data

    Returns:
        List[Feature]: List of features with their detected types
    """
    detected_features = []

    # Get the data from the dataset using read()
    data = dataset.read()
    if data.empty or not data.columns.tolist():
        return []

    for column in data.columns:
        values = data[column].tolist()
        if not values:
            continue

        # Try converting to numeric
        try:
            numeric_values = [float(val) for val in values]
            is_numeric = True
        except (ValueError, TypeError):
            is_numeric = False

        if not is_numeric:
            # If we can't convert to numbers, it's definitely categorical
            feature = Feature(
                name=column,
                type="categorical"
            )
        else:
            unique_values = set(numeric_values)
            unique_ratio = len(unique_values) / len(values)
            are_all_integer = all(
                float(val).is_integer() for val in numeric_values)

            # Modified heuristic: consider as categorical only if:
            # 1. Values are all integers
            # 2. Small number of unique values (<=3)
            # 3. Unique ratio is small (suggesting repeated values)
            if are_all_integer and (
              len(unique_values) <= 3 and unique_ratio < 0.05):
                feature = Feature(
                    name=column,
                    type="categorical"
                )
            else:
                feature = Feature(
                    name=column,
                    type="numerical"
                )

        detected_features.append(feature)

    return detected_features
