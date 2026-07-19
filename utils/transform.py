"""Module for transforming Fashion Studio product data."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


USD_TO_IDR_RATE = 16_000

REQUIRED_COLUMNS = [
    "Title",
    "Price",
    "Rating",
    "Colors",
    "Size",
    "Gender",
    "Timestamp",
]

INVALID_TITLES = {
    "unknown product",
}

INVALID_PRICES = {
    "price unavailable",
}

INVALID_RATINGS = {
    "invalid rating",
    "not rated",
}


def validate_input_data(data: Any) -> pd.DataFrame:
    """Validate and convert raw input data into a DataFrame.

    Args:
        data: Raw product data as a list of dictionaries or DataFrame.

    Returns:
        A copy of the validated pandas DataFrame.

    Raises:
        ValueError: If input data is empty or missing required columns.
        TypeError: If input type is unsupported.
        RuntimeError: If DataFrame conversion fails unexpectedly.
    """
    if data is None:
        raise ValueError("Input data must not be None.")

    try:
        if isinstance(data, pd.DataFrame):
            dataframe = data.copy()

        elif isinstance(data, list):
            if not data:
                raise ValueError("Input data must not be empty.")

            dataframe = pd.DataFrame(data)

        else:
            raise TypeError(
                "Input data must be a list of dictionaries "
                "or a pandas DataFrame."
            )

        if dataframe.empty:
            raise ValueError("Input data must not be empty.")

        missing_columns = [
            column
            for column in REQUIRED_COLUMNS
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing required columns: "
                f"{', '.join(missing_columns)}"
            )

        return dataframe[REQUIRED_COLUMNS].copy()

    except (ValueError, TypeError):
        raise

    except Exception as error:
        raise RuntimeError(
            f"Failed to validate input data: {error}"
        ) from error


def clean_title(value: Any) -> str | None:
    """Clean and validate a product title.

    Args:
        value: Raw title value.

    Returns:
        Cleaned title or None when invalid.
    """
    try:
        if value is None or pd.isna(value):
            return None

        title = str(value).strip()

        if not title:
            return None

        if title.lower() in INVALID_TITLES:
            return None

        return title

    except Exception:
        return None


def clean_price(
    value: Any,
    exchange_rate: int = USD_TO_IDR_RATE,
) -> float | None:
    """Convert a raw USD price into Indonesian rupiah.

    Examples:
        "$102.15" becomes 1634400.0.
        "Price Unavailable" becomes None.

    Args:
        value: Raw price value.
        exchange_rate: USD to IDR exchange rate.

    Returns:
        Converted rupiah price as float or None when invalid.
    """
    try:
        if value is None or pd.isna(value):
            return None

        if exchange_rate <= 0:
            raise ValueError(
                "Exchange rate must be greater than zero."
            )

        raw_price = str(value).strip()

        if not raw_price:
            return None

        if raw_price.lower() in INVALID_PRICES:
            return None

        normalized_price = (
            raw_price
            .replace("$", "")
            .replace(",", "")
            .strip()
        )

        price_usd = float(normalized_price)

        if price_usd < 0:
            return None

        return float(price_usd * exchange_rate)

    except ValueError:
        return None

    except (TypeError, AttributeError):
        return None


def clean_rating(value: Any) -> float | None:
    """Extract a numeric rating from raw rating text.

    Examples:
        "Rating: ⭐ 3.9 / 5" becomes 3.9.
        "Rating: Not Rated" becomes None.

    Args:
        value: Raw rating value.

    Returns:
        Rating as float or None when invalid.
    """
    try:
        if value is None or pd.isna(value):
            return None

        raw_rating = str(value).strip()

        if not raw_rating:
            return None

        lowered_rating = raw_rating.lower()

        if any(
            invalid_value in lowered_rating
            for invalid_value in INVALID_RATINGS
        ):
            return None

        match = re.search(
            r"(\d+(?:\.\d+)?)",
            raw_rating,
        )

        if match is None:
            return None

        rating = float(match.group(1))

        if not 0 <= rating <= 5:
            return None

        return rating

    except (ValueError, TypeError, AttributeError):
        return None


def clean_colors(value: Any) -> int | None:
    """Extract the number of available colors.

    Examples:
        "3 Colors" becomes 3.

    Args:
        value: Raw colors value.

    Returns:
        Number of colors as int or None when invalid.
    """
    try:
        if value is None or pd.isna(value):
            return None

        raw_colors = str(value).strip()

        if not raw_colors:
            return None

        match = re.search(r"(\d+)", raw_colors)

        if match is None:
            return None

        colors = int(match.group(1))

        if colors < 1:
            return None

        return colors

    except (ValueError, TypeError, AttributeError):
        return None


def clean_size(value: Any) -> str | None:
    """Remove the 'Size:' prefix from a raw size value.

    Examples:
        "Size: M" becomes "M".

    Args:
        value: Raw size value.

    Returns:
        Cleaned size or None when invalid.
    """
    try:
        if value is None or pd.isna(value):
            return None

        raw_size = str(value).strip()

        if not raw_size:
            return None

        cleaned_size = re.sub(
            r"^size\s*:\s*",
            "",
            raw_size,
            flags=re.IGNORECASE,
        ).strip()

        return cleaned_size if cleaned_size else None

    except (TypeError, AttributeError):
        return None


def clean_gender(value: Any) -> str | None:
    """Remove the 'Gender:' prefix from a raw gender value.

    Examples:
        "Gender: Women" becomes "Women".

    Args:
        value: Raw gender value.

    Returns:
        Cleaned gender or None when invalid.
    """
    try:
        if value is None or pd.isna(value):
            return None

        raw_gender = str(value).strip()

        if not raw_gender:
            return None

        cleaned_gender = re.sub(
            r"^gender\s*:\s*",
            "",
            raw_gender,
            flags=re.IGNORECASE,
        ).strip()

        return cleaned_gender if cleaned_gender else None

    except (TypeError, AttributeError):
        return None


def clean_timestamp(value: Any) -> str | None:
    """Validate and normalize an extraction timestamp.

    Args:
        value: Raw timestamp value.

    Returns:
        ISO-formatted timestamp string or None when invalid.
    """
    try:
        if value is None or pd.isna(value):
            return None

        parsed_timestamp = pd.to_datetime(
            value,
            errors="coerce",
        )

        if pd.isna(parsed_timestamp):
            return None

        return parsed_timestamp.isoformat()

    except (ValueError, TypeError, OverflowError):
        return None


def remove_invalid_rows(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Remove rows containing null transformed values.

    Args:
        dataframe: Transformed product DataFrame.

    Returns:
        DataFrame without null values.

    Raises:
        ValueError: If dataframe is empty.
        RuntimeError: If filtering fails.
    """
    if dataframe.empty:
        raise ValueError(
            "DataFrame must not be empty before filtering."
        )

    try:
        cleaned_dataframe = dataframe.dropna(
            subset=REQUIRED_COLUMNS
        ).copy()

        return cleaned_dataframe

    except Exception as error:
        raise RuntimeError(
            f"Failed to remove invalid rows: {error}"
        ) from error


def remove_duplicate_rows(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Remove duplicate product records.

    Timestamp is excluded from duplicate comparison because products
    scraped at different times should still be considered duplicates
    when their business attributes are identical.

    Args:
        dataframe: Product DataFrame.

    Returns:
        DataFrame without duplicate product records.

    Raises:
        ValueError: If dataframe is empty.
        RuntimeError: If duplicate removal fails.
    """
    if dataframe.empty:
        raise ValueError(
            "DataFrame must not be empty before deduplication."
        )

    duplicate_subset = [
        "Title",
        "Price",
        "Rating",
        "Colors",
        "Size",
        "Gender",
    ]

    try:
        deduplicated_dataframe = dataframe.drop_duplicates(
            subset=duplicate_subset,
            keep="first",
        ).copy()

        return deduplicated_dataframe

    except Exception as error:
        raise RuntimeError(
            f"Failed to remove duplicate rows: {error}"
        ) from error


def enforce_data_types(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the final data types required by the submission.

    Args:
        dataframe: Clean product DataFrame.

    Returns:
        DataFrame with normalized data types.

    Raises:
        ValueError: If dataframe is empty.
        RuntimeError: If type conversion fails.
    """
    if dataframe.empty:
        raise ValueError(
            "DataFrame must not be empty before type conversion."
        )

    try:
        typed_dataframe = dataframe.copy()

        typed_dataframe["Title"] = (
            typed_dataframe["Title"].astype("string")
        )
        typed_dataframe["Price"] = (
            typed_dataframe["Price"].astype("float64")
        )
        typed_dataframe["Rating"] = (
            typed_dataframe["Rating"].astype("float64")
        )
        typed_dataframe["Colors"] = (
            typed_dataframe["Colors"].astype("int64")
        )
        typed_dataframe["Size"] = (
            typed_dataframe["Size"].astype("string")
        )
        typed_dataframe["Gender"] = (
            typed_dataframe["Gender"].astype("string")
        )
        typed_dataframe["Timestamp"] = (
            typed_dataframe["Timestamp"].astype("string")
        )

        return typed_dataframe

    except Exception as error:
        raise RuntimeError(
            f"Failed to enforce data types: {error}"
        ) from error


def validate_transformed_data(
    dataframe: pd.DataFrame,
) -> None:
    """Validate the final transformed DataFrame.

    Args:
        dataframe: Final transformed DataFrame.

    Raises:
        ValueError: If final data quality requirements are not met.
    """
    if dataframe.empty:
        raise ValueError(
            "Transformed data must not be empty."
        )

    if dataframe.isnull().any().any():
        raise ValueError(
            "Transformed data still contains null values."
        )

    duplicate_subset = [
        "Title",
        "Price",
        "Rating",
        "Colors",
        "Size",
        "Gender",
    ]

    if dataframe.duplicated(
        subset=duplicate_subset
    ).any():
        raise ValueError(
            "Transformed data still contains duplicate values."
        )

    if (
        dataframe["Title"]
        .str.strip()
        .str.lower()
        .isin(INVALID_TITLES)
        .any()
    ):
        raise ValueError(
            "Transformed data still contains invalid titles."
        )

    if not dataframe["Rating"].between(0, 5).all():
        raise ValueError(
            "Rating values must be between 0 and 5."
        )

    if not dataframe["Price"].ge(0).all():
        raise ValueError(
            "Price values must not be negative."
        )

    if not dataframe["Colors"].ge(1).all():
        raise ValueError(
            "Colors values must be greater than zero."
        )


def transform_products(
    data: list[dict[str, Any]] | pd.DataFrame,
    exchange_rate: int = USD_TO_IDR_RATE,
) -> pd.DataFrame:
    """Transform raw Fashion Studio product data.

    Args:
        data: Raw product data from the extraction stage.
        exchange_rate: USD to IDR conversion rate.

    Returns:
        Clean pandas DataFrame ready for loading.

    Raises:
        ValueError: If data is empty or transformation produces no rows.
        TypeError: If input type is unsupported.
        RuntimeError: If unexpected transformation failure occurs.
    """
    try:
        dataframe = validate_input_data(data)

        dataframe["Title"] = (
            dataframe["Title"].apply(clean_title)
        )

        dataframe["Price"] = dataframe["Price"].apply(
            lambda value: clean_price(
                value,
                exchange_rate=exchange_rate,
            )
        )

        dataframe["Rating"] = (
            dataframe["Rating"].apply(clean_rating)
        )

        dataframe["Colors"] = (
            dataframe["Colors"].apply(clean_colors)
        )

        dataframe["Size"] = (
            dataframe["Size"].apply(clean_size)
        )

        dataframe["Gender"] = (
            dataframe["Gender"].apply(clean_gender)
        )

        dataframe["Timestamp"] = (
            dataframe["Timestamp"].apply(clean_timestamp)
        )

        dataframe = remove_invalid_rows(dataframe)

        if dataframe.empty:
            raise ValueError(
                "No valid product data remains after cleaning."
            )

        dataframe = remove_duplicate_rows(dataframe)

        if dataframe.empty:
            raise ValueError(
                "No product data remains after deduplication."
            )

        dataframe = enforce_data_types(dataframe)

        dataframe = dataframe.reset_index(drop=True)

        validate_transformed_data(dataframe)

        return dataframe

    except (ValueError, TypeError):
        raise

    except Exception as error:
        raise RuntimeError(
            f"Failed to transform product data: {error}"
        ) from error