import sys
print("Python Path:", sys.path)
from enum import Enum, auto
from dataclasses import dataclass
from random import gauss, randint, shuffle
from typing import List, Dict, Any
from real_estate_toolkit.agent_based_model.houses import House
from real_estate_toolkit.agent_based_model.house_market import HousingMarket
from real_estate_toolkit.agent_based_model.consumers import Segment, Consumer


class CleaningMarketMechanism(Enum):
    """
    Enum for different market cleaning mechanisms.
    """
    INCOME_ORDER_DESCENDANT = auto()
    INCOME_ORDER_ASCENDANT = auto()
    RANDOM = auto()


@dataclass
class AnnualIncomeStatistics:
    """
    Class representing statistics for generating annual income.
    """
    minimum: float
    average: float
    standard_deviation: float
    maximum: float


@dataclass
class ChildrenRange:
    """
    Class representing range of children per household.
    """
    minimum: int = 0
    maximum: int = 5


@dataclass
class Simulation:
    """
    Class for orchestrating a real estate market simulation.
    """
    housing_market_data: List[Dict[str, Any]]
    consumers_number: int
    years: int
    annual_income: AnnualIncomeStatistics
    children_range: ChildrenRange
    cleaning_market_mechanism: CleaningMarketMechanism
    down_payment_percentage: float = 0.2
    saving_rate: float = 0.3
    interest_rate: float = 0.05

    def create_housing_market(self):
        """
        Initialize market with houses.

        Converts raw data to House objects and assigns to self.housing_market.
        """
        self.housing_market = HousingMarket(
            houses=[
                House(
                    id=entry["id"],
                    price=entry["price"],
                    area=entry["area"],
                    bedrooms=entry["bedrooms"],
                    year_built=entry["year_built"],
                    quality_score=entry["quality_score"],
                )
                for entry in self.housing_market_data
            ]
        )

    def create_consumers(self) -> None:
        """
        Generate consumer population.

        Generates a list of consumers based on statistical distributions and assigns to self.consumers.
        """
        self.consumers = []
        for i in range(self.consumers_number):
            # Generate random annual income within specified range
            annual_income = gauss(self.annual_income.average, self.annual_income.standard_deviation)
            while annual_income < self.annual_income.minimum or annual_income > self.annual_income.maximum:
                annual_income = gauss(self.annual_income.average, self.annual_income.standard_deviation)

            # Generate random number of children
            children_number = randint(self.children_range.minimum, self.children_range.maximum)

            # Assign a random consumer segment
            segment = Segment(randint(1, len(Segment)))

            # Create the consumer
            consumer = Consumer(
                id=i,
                annual_income=annual_income,
                children_number=children_number,
                segment=segment,
                saving_rate=self.saving_rate,
                interest_rate=self.interest_rate,
            )
            self.consumers.append(consumer)

    def compute_consumers_savings(self) -> None:
        """
        Calculate savings for all consumers.
        """
        for consumer in self.consumers:
            consumer.compute_savings(self.years)

    def clean_the_market(self) -> None:
        """
        Execute market transactions based on the cleaning mechanism.

        Consumers attempt to buy houses in a specified order.
        """
        # Sort consumers based on the cleaning mechanism
        if self.cleaning_market_mechanism == CleaningMarketMechanism.INCOME_ORDER_DESCENDANT:
            self.consumers.sort(key=lambda c: c.annual_income, reverse=True)
        elif self.cleaning_market_mechanism == CleaningMarketMechanism.INCOME_ORDER_ASCENDANT:
            self.consumers.sort(key=lambda c: c.annual_income)
        elif self.cleaning_market_mechanism == CleaningMarketMechanism.RANDOM:
            shuffle(self.consumers)

        # Consumers attempt to buy houses
        for consumer in self.consumers:
            consumer.buy_a_house(self.housing_market)

    def compute_owners_population_rate(self) -> float:
        """
        Compute the proportion of consumers who own a house.

        Returns:
            float: Ownership rate as a percentage.
        """
        owners = sum(1 for consumer in self.consumers if consumer.house is not None)
        return (owners / len(self.consumers)) * 100 if self.consumers else 0.0

    def compute_houses_availability_rate(self) -> float:
        """
        Compute the proportion of houses still available in the market.

        Returns:
            float: Availability rate as a percentage.
        """
        available_houses = sum(1 for house in self.housing_market.houses if house.available)
        return (available_houses / len(self.housing_market.houses)) * 100 if self.housing_market.houses else 0.0
