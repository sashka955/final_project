from typing import List, Dict
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import os


class MarketAnalyzer:
    def __init__(self, data_path: str):
        """
        Initialize the analyzer with data from a CSV file.
        
        Args:
            data_path (str): Path to the Ames Housing dataset
        """
        self.data_path = data_path
        self.real_estate_data = pl.read_csv(data_path)
        self.real_estate_clean_data = None

        # Ensure outputs folder exists
        self.output_folder = "src/real_estate_toolkit/analytics/outputs/"
        os.makedirs(self.output_folder, exist_ok=True)

    def clean_data(self) -> None:
        """
        Perform comprehensive data cleaning.
        """
        # Handle missing values
        self.real_estate_clean_data = self.real_estate_data.fill_null_strategy("mean")

        # Convert columns to appropriate types
        for column in self.real_estate_clean_data.columns:
            if "int" in str(self.real_estate_clean_data[column].dtype):
                self.real_estate_clean_data = self.real_estate_clean_data.with_column(
                    pl.col(column).cast(pl.Int32)
                )
            elif "float" in str(self.real_estate_clean_data[column].dtype):
                self.real_estate_clean_data = self.real_estate_clean_data.with_column(
                    pl.col(column).cast(pl.Float32)
                )
            elif "str" in str(self.real_estate_clean_data[column].dtype):
                self.real_estate_clean_data = self.real_estate_clean_data.with_column(
                    pl.col(column).cast(pl.Categorical)
                )

    def generate_price_distribution_analysis(self) -> pl.DataFrame:
        """
        Analyze sale price distribution using clean data.
        """
        if self.real_estate_clean_data is None:
            raise ValueError("Data must be cleaned before analysis.")

        # Compute statistics
        price_statistics = self.real_estate_clean_data.select([
            pl.col("SalePrice").mean().alias("mean"),
            pl.col("SalePrice").median().alias("median"),
            pl.col("SalePrice").std().alias("std_dev"),
            pl.col("SalePrice").min().alias("min"),
            pl.col("SalePrice").max().alias("max"),
        ])

        # Create histogram
        fig = px.histogram(
            self.real_estate_clean_data.to_pandas(),
            x="SalePrice",
            nbins=50,
            title="Sale Price Distribution",
        )
        fig.write_html(os.path.join(self.output_folder, "price_distribution.html"))
        fig.write_image(os.path.join(self.output_folder, "price_distribution.png"))

        return price_statistics

    def neighborhood_price_comparison(self) -> pl.DataFrame:
        """
        Create a boxplot comparing house prices across different neighborhoods.
        """
        if self.real_estate_clean_data is None:
            raise ValueError("Data must be cleaned before analysis.")

        # Group data by neighborhood and compute price statistics
        neighborhood_stats = self.real_estate_clean_data.groupby("Neighborhood").agg([
            pl.col("SalePrice").mean().alias("mean_price"),
            pl.col("SalePrice").median().alias("median_price"),
            pl.col("SalePrice").min().alias("min_price"),
            pl.col("SalePrice").max().alias("max_price"),
            pl.col("SalePrice").std().alias("std_dev"),
        ])

        # Create boxplot
        fig = px.box(
            self.real_estate_clean_data.to_pandas(),
            x="Neighborhood",
            y="SalePrice",
            title="Neighborhood Price Comparison",
        )
        fig.update_layout(xaxis={'categoryorder': 'total descending'})
        fig.write_html(os.path.join(self.output_folder, "neighborhood_price_comparison.html"))
        fig.write_image(os.path.join(self.output_folder, "neighborhood_price_comparison.png"))

        return neighborhood_stats

    def feature_correlation_heatmap(self, variables: List[str]) -> None:
        """
        Generate a correlation heatmap for specified numerical variables.
        """
        if self.real_estate_clean_data is None:
            raise ValueError("Data must be cleaned before analysis.")

        # Compute correlation matrix
        correlation_matrix = self.real_estate_clean_data.select(variables).to_pandas().corr()

        # Create heatmap
        fig = px.imshow(
            correlation_matrix,
            title="Feature Correlation Heatmap",
            labels={"color": "Correlation"},
            text_auto=True,
        )
        fig.write_html(os.path.join(self.output_folder, "correlation_heatmap.html"))
        fig.write_image(os.path.join(self.output_folder, "correlation_heatmap.png"))

    def create_scatter_plots(self) -> Dict[str, go.Figure]:
        """
        Create scatter plots exploring relationships between key features.
        """
        if self.real_estate_clean_data is None:
            raise ValueError("Data must be cleaned before analysis.")

        scatter_plots = {}

        # SalePrice vs Total square footage
        scatter_plots["price_vs_sqft"] = px.scatter(
            self.real_estate_clean_data.to_pandas(),
            x="GrLivArea",
            y="SalePrice",
            title="Sale Price vs Total Square Footage",
            trendline="ols",
            color="Neighborhood",
        )
        scatter_plots["price_vs_sqft"].write_html(os.path.join(self.output_folder, "price_vs_sqft.html"))
        scatter_plots["price_vs_sqft"].write_image(os.path.join(self.output_folder, "price_vs_sqft.png"))

        # SalePrice vs Year Built
        scatter_plots["price_vs_year"] = px.scatter(
            self.real_estate_clean_data.to_pandas(),
            x="YearBuilt",
            y="SalePrice",
            title="Sale Price vs Year Built",
            trendline="ols",
            color="Neighborhood",
        )
        scatter_plots["price_vs_year"].write_html(os.path.join(self.output_folder, "price_vs_year.html"))
        scatter_plots["price_vs_year"].write_image(os.path.join(self.output_folder, "price_vs_year.png"))

        # Overall Quality vs Sale Price
        scatter_plots["quality_vs_price"] = px.scatter(
            self.real_estate_clean_data.to_pandas(),
            x="OverallQual",
            y="SalePrice",
            title="Overall Quality vs Sale Price",
            trendline="ols",
            color="Neighborhood",
        )
        scatter_plots["quality_vs_price"].write_html(os.path.join(self.output_folder, "quality_vs_price.html"))
        scatter_plots["quality_vs_price"].write_image(os.path.join(self.output_folder, "quality_vs_price.png"))

        return scatter_plots
