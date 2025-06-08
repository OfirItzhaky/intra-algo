import pandas as pd
from typing import List, Optional, Dict, Any

def validate_csv_against_indicators(
    df: pd.DataFrame,
    required_indicators: List[str],
    required_bar_count: Optional[int] = None
) -> Dict[str, Any]:
    """
    Validate the uploaded CSV DataFrame against a list of required indicator columns.

    Args:
        df (pd.DataFrame): The uploaded data as a pandas DataFrame.
        required_indicators (List[str]): List of indicator column names required for analysis.
        required_bar_count (Optional[int]): Minimum number of rows required in the DataFrame (if specified).

    Returns:
        Dict[str, Any]: A dictionary with:
            - is_valid (bool): True if all required indicators are present and row count is sufficient.
            - missing (List[str]): List of missing indicator columns.
            - validated_df (Optional[pd.DataFrame]): DataFrame trimmed to required_bar_count rows if valid, else None.
    """
    missing = [col for col in required_indicators if col not in df.columns]
    errors = []
    if required_bar_count is not None:
        try:
            required_bar_count = int(required_bar_count)
        except (TypeError, ValueError):
            required_bar_count = None
    if required_bar_count is not None and len(df) < required_bar_count:
        errors.append(f"Not enough rows: required {required_bar_count}, found {len(df)}")
    validated_df = None
    if not missing and not errors:
        if required_bar_count is not None:
            validated_df = df.tail(required_bar_count)
        else:
            validated_df = df.copy()
    return {
        'is_valid': len(missing) == 0 and len(errors) == 0,
        'missing': missing,
        'validated_df': validated_df
    }
