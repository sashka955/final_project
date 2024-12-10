from enum import Enum
from dataclasses import dataclass
from typing import Optional


class QualityScore(Enum):
    """
    Enum to represent the quality of a house.
    """
    EXCELLENT = 5
    GOOD = 4
    AVERAGE = 3
    FAIR = 2
    POOR = 1


@dataclass
class House:
    """
    Class representing a house in the real estate market.
    """
    id: int
    price: float
    area: float  # Square footage
    bedrooms: int
    year_built: int
    quality_score: Optional[QualityScore] = None
    available: bool = True

    def calculate_price_per_square_foot(self) -> float:
        """
        Calculate and return the price per square foot.

        Returns:
            float: Price per square foot, rounded to 2 decimal places.
        """
        if self.area <= 0:
            raise ValueError("Area must be greater than zero to calculate price per square foot.")
        return round(self.price / self.area, 2)

    def is_new_construction(self, current_year: int = 2024) -> bool:
        """
        Determine if the house is considered new construction (< 5 years old).

        Args:
            current_year (int): The current year for comparison. Default is 2024.

        Returns:
            bool: True if the house is new construction, False otherwise.
        """
        age = current_year - self.year_built
        return age < 5

    def get_quality_score(self) -> QualityScore:
        """
        Generate a quality score based on house attributes if it is missing.

        Returns:
            QualityScore: Quality score of the house.
        """
        if self.quality_score is not None:
            return self.quality_score

        # Compute a default quality score based on attributes
        if self.year_built >= 2020 and self.bedrooms >= 4 and self.area >= 2000:
            self.quality_score = QualityScore.EXCELLENT
        elif self.year_built >= 2000 and self.bedrooms >= 3 and self.area >= 1500:
            self.quality_score = QualityScore.GOOD
        elif self.year_built >= 1980 and self.bedrooms >= 2 and self.area >= 1000:
            self.quality_score = QualityScore.AVERAGE
        elif self.year_built >= 1960 and self.area >= 800:
            self.quality_score = QualityScore.FAIR
        else:
            self.quality_score = QualityScore.POOR

        return self.quality_score

    def sell_house(self) -> None:
        """
        Mark the house as sold by setting its availability to False.
        """
        if not self.available:
            raise ValueError("House is already sold.")
        self.available = False
