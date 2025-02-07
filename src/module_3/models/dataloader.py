import pandas as pd
import logging
from typing_extensions import Annotated
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class DataSchema(BaseModel):
    """Schema validation for the dataset"""
    variant_id: Annotated[int, Field(ge=0)]  # Non-negative integer
    product_type: Annotated[str, Field(min_length=1)]  # Non-empty string
    order_id: Annotated[int, Field(ge=0)]  # Must be a non-negative integer
    user_id: Annotated[int, Field(ge=0)]  # Must be a non-negative integer
    created_at: Annotated[datetime, Field()]  # Must be a valid datetime
    order_date: Annotated[datetime, Field()]  # Must be a valid datetime
    user_order_seq: Annotated[int, Field(ge=0)]  # Non-negative integer
    outcome: Annotated[int, Field(ge=0, le=1)]  # Binary column (0 or 1)
    ordered_before: Annotated[int, Field(ge=0, le=1)]  # Binary column (0 or 1)
    abandoned_before: Annotated[int, Field(ge=0, le=1)]  # Binary column (0 or 1)
    active_snoozed: Annotated[int, Field(ge=0, le=1)]  # Binary column (0 or 1)
    set_as_regular: Annotated[int, Field(ge=0, le=1)]  # Binary column (0 or 1)
    normalised_price: Annotated[float, Field(ge=0)]  # Non-negative float
    discount_pct: Annotated[float, Field(ge=-100, le=100)]  # Percentage (-100 to 100)
    vendor: Optional[Annotated[str, Field(min_length=1)]]  # Non-empty string (optional)
    global_popularity: Annotated[float, Field(ge=0, le=1)]  # Probability range (0 to 1)
    count_adults: Annotated[int, Field(ge=0)]  # Non-negative integer
    count_children: Annotated[int, Field(ge=0)]  # Non-negative integer
    count_babies: Annotated[int, Field(ge=0)]  # Non-negative integer
    count_pets: Annotated[int, Field(ge=0)]  # Non-negative integer
    people_ex_baby: Annotated[int, Field(ge=0)]  # Non-negative integer
    days_since_purchase_variant_id: Optional[Annotated[int, Field(ge=0)]]  # Non-negative integer (optional)
    avg_days_to_buy_variant_id: Optional[Annotated[float, Field(ge=0)]]  # Non-negative float (optional)
    std_days_to_buy_variant_id: Optional[Annotated[float, Field(ge=0)]]  # Non-negative float (optional)
    days_since_purchase_product_type: Optional[Annotated[int, Field(ge=0)]]  # Non-negative integer (optional)
    avg_days_to_buy_product_type: Optional[Annotated[float, Field(ge=0)]]  # Non-negative float (optional)
    std_days_to_buy_product_type: Optional[Annotated[float, Field(ge=0)]]  # Non-negative float (optional)

    @field_validator("created_at", "order_date", mode='before')
    @classmethod
    def parse_dates(cls, value: str) -> str:
        """Ensure datetime fields are properly parsed"""
        if isinstance(value, str):
            return datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
        return value

class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load_data(self) -> pd.DataFrame:
        logger.info(f"Loading data from {self.filepath}")
        df = pd.read_csv(self.filepath)
        self.validate_data(df)
        logger.info("Data successfully loaded and validated")
        return df

    def validate_data(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            try:
                DataSchema(**row.to_dict())  # Validate each row
            except ValidationError as e:
                logger.error(f"Data validation error: {e}")
                raise ValueError(f"Data validation error: {e}")
