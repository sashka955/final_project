from typing import List, Optional
from statistics import mean
from real_estate_toolkit.agent_based_model.houses import House


class HousingMarket:
    """
    Class to manage a collection of houses and perform market-wide operations.
    """
    def __init__(self, houses: List[House]):
        self.houses: List[House] = houses

    def get_house_by_id(self, house_id: int) -> Optional[House]:
        """
        Retrieve specific house by ID.

        Args:
            house_id (int): The unique ID of the house.

        Returns:
            House: The house with the specified ID, or None if not found.
        """
        for house in self.houses:
            if house.id == house_id:
                return house
        return None  # House not found

    def calculate_average_price(self, bedrooms: Optional[int] = None) -> float:
        """
        Calculate the average price of houses.

        Args:
            bedrooms (Optional[int]): Filter houses by a specific number of bedrooms. Defaults to None.

        Returns:
            float: The average price of houses meeting the criteria.
        """
        filtered_houses = (
            [house.price for house in self.houses if house.bedrooms == bedrooms]
            if bedrooms is not None
            else [house.price for house in self.houses]
        )

        if not filtered_houses:
            return 0.0  # No houses meet the criteria

        return round(mean(filtered_houses), 2)

    def get_houses_that_meet_requirements(self, max_price: int, segment: str) -> List[House]:
        """
        Filter houses based on buyer requirements.

        Args:
            max_price (int): Maximum price the buyer can afford.
            segment (str): Buyer segment, which can influence the filtering logic.

        Returns:
            List[House]: A list of houses that meet the buyer's requirements.
        """
        segment = segment.lower()
        filtered_houses = [
            house
            for house in self.houses
            if house.price <= max_price and house.available
        ]

        # Segment-specific filtering logic
        if segment == "fancy":
            return [house for house in filtered_houses if house.quality_score and house.quality_score.value >= 4]
        elif segment == "optimizer":
            return [house for house in filtered_houses if house.calculate_price_per_square_foot() <= 200]
        elif segment == "average":
            return filtered_houses
        else:
            raise ValueError(f"Unknown segment: {segment}")
