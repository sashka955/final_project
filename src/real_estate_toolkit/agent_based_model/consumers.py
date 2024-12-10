from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional
from real_estate_toolkit.agent_based_model.houses import House
from real_estate_toolkit.agent_based_model.house_market import HousingMarket


class Segment(Enum):
    """
    Enumeration of consumer segments.
    """
    FANCY = auto()  # Prefers new construction and high quality scores
    OPTIMIZER = auto()  # Focuses on price per square foot value
    AVERAGE = auto()  # Considers average market prices


@dataclass
class Consumer:
    """
    Class representing a consumer in the real estate market.
    """
    id: int
    annual_income: float
    children_number: int
    segment: Segment
    house: Optional[House] = None
    savings: float = 0.0
    saving_rate: float = 0.3
    interest_rate: float = 0.05

    def compute_savings(self, years: int) -> None:
        """
        Calculate accumulated savings over time using compound interest.

        Args:
            years (int): Number of years to compute savings for.
        """
        yearly_savings = self.annual_income * self.saving_rate
        self.savings = yearly_savings * (((1 + self.interest_rate) ** years - 1) / self.interest_rate)

    def buy_a_house(self, housing_market: HousingMarket) -> None:
        """
        Attempt to purchase a suitable house.

        Args:
            housing_market (HousingMarket): The housing market to search for houses.

        Implementation:
        - Check savings against house prices
        - Consider down payment requirements
        - Match house to family size needs
        - Apply segment-specific preferences
        """
        if self.savings <= 0:
            print(f"Consumer {self.id} does not have sufficient savings.")
            return

        # Filter houses based on segment
        if self.segment == Segment.FANCY:
            max_price = self.savings * 5  # FANCY buyers may qualify for higher mortgages
            filtered_houses = housing_market.get_houses_that_meet_requirements(max_price=max_price, segment="fancy")
        elif self.segment == Segment.OPTIMIZER:
            max_price = self.savings * 4
            filtered_houses = housing_market.get_houses_that_meet_requirements(max_price=max_price, segment="optimizer")
        else:  # Segment.AVERAGE
            max_price = self.savings * 3
            filtered_houses = housing_market.get_houses_that_meet_requirements(max_price=max_price, segment="average")

        # If no houses meet the criteria
        if not filtered_houses:
            print(f"Consumer {self.id} could not find a suitable house.")
            return

        # Attempt to buy the first house in the filtered list
        for house in filtered_houses:
            down_payment = house.price * 0.2  # Assume 20% down payment
            if self.savings >= down_payment:
                # Buy the house
                self.house = house
                self.savings -= down_payment
                house.sell_house()
                print(f"Consumer {self.id} bought house {house.id} for ${house.price}.")
                return

        print(f"Consumer {self.id} could not afford any suitable house.")
