import logging
from datetime import datetime
from typing import Annotated

import pandas as pd
from pydantic import BaseModel, Field, ValidationError, field_validator

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataSchema(BaseModel):
    variant_id: Annotated[int, Field(ge=0)]
    product_type: Annotated[str, Field(min_length=1)]
    order_id: Annotated[int, Field(ge=0)]
    user_id: Annotated[int, Field(ge=0)]
    created_at: Annotated[datetime, Field()]
    order_date: Annotated[datetime, Field()]
    user_order_seq: Annotated[int, Field(ge=0)]
    outcome: Annotated[int, Field(ge=0, le=1)]
    ordered_before: Annotated[int, Field(ge=0, le=1)]
    abandoned_before: Annotated[int, Field(ge=0, le=1)]
    active_snoozed: Annotated[int, Field(ge=0, le=1)]
    set_as_regular: Annotated[int, Field(ge=0, le=1)]
    normalised_price: Annotated[float, Field(ge=0)]
    discount_pct: Annotated[float, Field(ge=-100, le=100)]
    vendor: Annotated[str, Field(min_length=1)] | None
    global_popularity: Annotated[float, Field(ge=0, le=1)]
    count_adults: Annotated[int, Field(ge=0)]
    count_children: Annotated[int, Field(ge=0)]
    count_babies: Annotated[int, Field(ge=0)]
    count_pets: Annotated[int, Field(ge=0)]
    people_ex_baby: Annotated[int, Field(ge=0)]
    days_since_purchase_variant_id: Annotated[int, Field(ge=0)] | None
    avg_days_to_buy_variant_id: Annotated[float, Field(ge=0)] | None
    std_days_to_buy_variant_id: Annotated[float, Field(ge=0)] | None
    days_since_purchase_product_type: Annotated[int, Field(ge=0)] | None
    avg_days_to_buy_product_type: Annotated[float, Field(ge=0)] | None
    std_days_to_buy_product_type: Annotated[float, Field(ge=0)] | None

    @field_validator('created_at', 'order_date', mode='before')
    @classmethod
    def parse_dates(cls, value: str) -> str:
        if isinstance(value, str):
            return datetime.strptime(value, '%Y-%m-%d %H:%M:%S')
        return value


class DataLoader:
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load_data(self) -> pd.DataFrame:
        logger.info(f'Loading data from {self.filepath}')
        df = pd.read_csv(self.filepath)
        self.validate_data(df)
        logger.info('Data successfully loaded and validated')
        return df

    def validate_data(self, df: pd.DataFrame):
        for _, row in df.iterrows():
            try:
                DataSchema(**row.to_dict())  # Validate each row
            except ValidationError as e:
                logger.error(f'Data validation error: {e}')
                raise ValueError(f'Data validation error: {e}')
