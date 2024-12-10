from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any
import csv


@dataclass
class DataLoader:
    """
    Class for loading and basic processing of real estate data.
    """
    data_path: Path

    def load_data_from_csv(self) -> List[Dict[str, Any]]:
        """
        Load data from CSV file into a list of dictionaries.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary represents a row.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"The file at {self.data_path} does not exist.")
        
        with open(self.data_path, mode='r', encoding='utf-8') as csv_file:
            reader = csv.DictReader(csv_file)
            data = [row for row in reader]  # Convert reader to a list of dictionaries
        return data

    def validate_columns(self, required_columns: List[str]) -> bool:
        """
        Validate that all required columns are present in the dataset.

        Args:
            required_columns (List[str]): List of column names that are required.

        Returns:
            bool: True if all required columns are present, False otherwise.
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"The file at {self.data_path} does not exist.")
        
        with open(self.data_path, mode='r', encoding='utf-8') as csv_file:
            reader = csv.reader(csv_file)
            headers = next(reader)  # Get the first row as headers
        
        missing_columns = [col for col in required_columns if col not in headers]
        if missing_columns:
            print(f"Missing columns: {missing_columns}")
            return False
        return True
