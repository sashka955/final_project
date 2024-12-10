from dataclasses import dataclass
from typing import Dict, List, Any
import re


@dataclass
class Cleaner:
    """
    Class for cleaning real estate data.
    """
    data: List[Dict[str, Any]]

    def rename_with_best_practices(self) -> None:
        """
        Rename the columns with best practices (e.g., snake_case and descriptive names).
        This method modifies the column names directly in the data list.
        """
        if not self.data:
            return  # No data to rename
        
        # Extract the current column names from the first dictionary
        original_columns = list(self.data[0].keys())

        # Create a mapping of old column names to new ones
        column_mapping = {
            col: self._to_snake_case(col) for col in original_columns
        }

        # Rename the columns in every row of the data
        for row in self.data:
            for old_col, new_col in column_mapping.items():
                if old_col in row:
                    row[new_col] = row.pop(old_col)

    def na_to_none(self) -> List[Dict[str, Any]]:
        """
        Replace 'NA' values with None in all entries.

        Returns:
            List[Dict[str, Any]]: The modified data with 'NA' replaced by None.
        """
        for row in self.data:
            for key, value in row.items():
                if value == "NA":
                    row[key] = None
        return self.data

    @staticmethod
    def _to_snake_case(column_name: str) -> str:
        """
        Convert a string to snake_case.

        Args:
            column_name (str): Original column name.

        Returns:
            str: Transformed column name in snake_case.
        """
        # Replace spaces or hyphens with underscores, convert to lowercase, and remove special characters
        column_name = re.sub(r'[^a-zA-Z0-9\s]', '', column_name)  # Remove non-alphanumeric chars
        column_name = re.sub(r'[\s\-]+', '_', column_name)  # Replace spaces/hyphens with underscores
        return column_name.lower()
