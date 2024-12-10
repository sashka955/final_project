from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Union
import statistics
import numpy as np


@dataclass
class Descriptor:
    """
    Class for describing real estate data.
    """
    data: List[Dict[str, Any]]

    def _validate_columns(self, columns: Union[List[str], str]) -> List[str]:
        """Validate column names and return a list of valid column names."""
        available_columns = set(self.data[0].keys()) if self.data else set()
        
        if columns == "all":
            return list(available_columns)
        
        if not isinstance(columns, list):
            raise ValueError("`columns` must be a list of strings or 'all'.")

        invalid_columns = [col for col in columns if col not in available_columns]
        if invalid_columns:
            raise ValueError(f"Invalid column names: {invalid_columns}")
        
        return columns

    def none_ratio(self, columns: Union[List[str], str] = "all") -> Dict[str, float]:
        """
        Compute the ratio of None values per column.
        """
        valid_columns = self._validate_columns(columns)
        none_ratios = {}

        for col in valid_columns:
            total = len(self.data)
            none_count = sum(1 for row in self.data if row[col] is None)
            none_ratios[col] = none_count / total if total > 0 else 0
        
        return none_ratios

    def average(self, columns: Union[List[str], str] = "all") -> Dict[str, float]:
        """
        Compute the average value for numeric variables. Omit None values.
        """
        valid_columns = self._validate_columns(columns)
        averages = {}

        for col in valid_columns:
            try:
                numeric_values = [row[col] for row in self.data if row[col] is not None and isinstance(row[col], (int, float))]
                averages[col] = sum(numeric_values) / len(numeric_values) if numeric_values else 0
            except TypeError:
                raise ValueError(f"Column '{col}' contains non-numeric data.")

        return averages

    def median(self, columns: Union[List[str], str] = "all") -> Dict[str, float]:
        """
        Compute the median value for numeric variables. Omit None values.
        """
        valid_columns = self._validate_columns(columns)
        medians = {}

        for col in valid_columns:
            try:
                numeric_values = sorted(row[col] for row in self.data if row[col] is not None and isinstance(row[col], (int, float)))
                medians[col] = statistics.median(numeric_values) if numeric_values else 0
            except TypeError:
                raise ValueError(f"Column '{col}' contains non-numeric data.")

        return medians

    def percentile(self, columns: Union[List[str], str] = "all", percentile: int = 50) -> Dict[str, float]:
        """
        Compute a specific percentile for numeric variables. Omit None values.
        """
        valid_columns = self._validate_columns(columns)
        percentiles = {}

        for col in valid_columns:
            try:
                numeric_values = sorted(row[col] for row in self.data if row[col] is not None and isinstance(row[col], (int, float)))
                if numeric_values:
                    k = (len(numeric_values) - 1) * (percentile / 100)
                    f = int(k)
                    c = f + 1
                    if f == c or c >= len(numeric_values):
                        percentiles[col] = numeric_values[f]
                    else:
                        percentiles[col] = numeric_values[f] + (numeric_values[c] - numeric_values[f]) * (k - f)
                else:
                    percentiles[col] = 0
            except TypeError:
                raise ValueError(f"Column '{col}' contains non-numeric data.")

        return percentiles

    def type_and_mode(self, columns: Union[List[str], str] = "all") -> Dict[str, Union[Tuple[str, float], Tuple[str, str]]]:
        """
        Compute the mode for variables. Omit None values.
        """
        valid_columns = self._validate_columns(columns)
        type_modes = {}

        for col in valid_columns:
            non_none_values = [row[col] for row in self.data if row[col] is not None]
            
            if not non_none_values:
                type_modes[col] = ("unknown", None)
                continue

            var_type = type(non_none_values[0]).__name__
            try:
                mode = statistics.mode(non_none_values)
            except statistics.StatisticsError:
                mode = None  # No unique mode if all values are equally frequent

            type_modes[col] = (var_type, mode)

        return type_modes


@dataclass
class DescriptorNumpy:
    """
    Class for describing real estate data using NumPy.
    """
    data: np.ndarray
    column_names: List[str]

    def _validate_columns(self, columns: Union[List[str], str]) -> List[int]:
        """Validate column names and return a list of valid column indices."""
        if columns == "all":
            return list(range(len(self.column_names)))
        
        if not isinstance(columns, list):
            raise ValueError("`columns` must be a list of strings or 'all'.")

        invalid_columns = [col for col in columns if col not in self.column_names]
        if invalid_columns:
            raise ValueError(f"Invalid column names: {invalid_columns}")
        
        return [self.column_names.index(col) for col in columns]

    def none_ratio(self, columns: Union[List[str], str] = "all") -> Dict[str, float]:
        """
        Compute the ratio of None values (represented as np.nan) per column.
        """
        valid_indices = self._validate_columns(columns)
        none_ratios = {}

        for idx in valid_indices:
            col_data = self.data[:, idx]
            none_count = np.sum(np.isnan(col_data))
            none_ratios[self.column_names[idx]] = none_count / len(col_data) if len(col_data) > 0 else 0
        
        return none_ratios

    def average(self, columns: Union[List[str], str] = "all") -> Dict[str, float]:
        """
        Compute the average value for numeric variables. Omit None (np.nan) values.
        """
        valid_indices = self._validate_columns(columns)
        averages = {}

        for idx in valid_indices:
            col_data = self.data[:, idx].astype(float)
            if np.issubdtype(col_data.dtype, np.number):
                averages[self.column_names[idx]] = np.nanmean(col_data)
            else:
                raise ValueError(f"Column '{self.column_names[idx]}' contains non-numeric data.")

        return averages

    def median(self, columns: Union[List[str], str] = "all") -> Dict[str, float]:
        """
        Compute the median value for numeric variables. Omit None (np.nan) values.
        """
        valid_indices = self._validate_columns(columns)
        medians = {}

        for idx in valid_indices:
            col_data = self.data[:, idx].astype(float)
            if np.issubdtype(col_data.dtype, np.number):
                medians[self.column_names[idx]] = np.nanmedian(col_data)
            else:
                raise ValueError(f"Column '{self.column_names[idx]}' contains non-numeric data.")

        return medians

    def percentile(self, columns: Union[List[str], str] = "all", percentile: int = 50) -> Dict[str, float]:
        """
        Compute a specific percentile for numeric variables. Omit None (np.nan) values.
        """
        valid_indices = self._validate_columns(columns)
        percentiles = {}

        for idx in valid_indices:
            col_data = self.data[:, idx].astype(float)
            if np.issubdtype(col_data.dtype, np.number):
                percentiles[self.column_names[idx]] = np.nanpercentile(col_data, percentile)
            else:
                raise ValueError(f"Column '{self.column_names[idx]}' contains non-numeric data.")

        return percentiles

    def type_and_mode(self, columns: Union[List[str], str] = "all") -> Dict[str, Tuple[str, Union[int, float, str]]]:
        """
        Compute the mode for variables. Omit None (np.nan) values.
        """
        valid_indices = self._validate_columns(columns)
        type_modes = {}

        for idx in valid_indices:
            col_data = self.data[:, idx]
            non_none_values = col_data[~np.isnan(col_data)] if col_data.dtype.kind in 'fc' else col_data[col_data != None]  # noqa: E711

            if not len(non_none_values):
                type_modes[self.column_names[idx]] = ("unknown", None)
                continue

            var_type = "float" if col_data.dtype.kind in 'fc' else "str"
            unique, counts = np.unique(non_none_values, return_counts=True)
            mode_index = np.argmax(counts)
            mode = unique[mode_index]

            type_modes[self.column_names[idx]] = (var_type, mode)

        return type_modes
