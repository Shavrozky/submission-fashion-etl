"""Unit tests for the transformation module."""

import pandas as pd
import pytest

from utils.transform import (
    REQUIRED_COLUMNS,
    USD_TO_IDR_RATE,
    clean_colors,
    clean_gender,
    clean_price,
    clean_rating,
    clean_size,
    clean_timestamp,
    clean_title,
    enforce_data_types,
    remove_duplicate_rows,
    remove_invalid_rows,
    transform_products,
    validate_input_data,
    validate_transformed_data,
)


VALID_RAW_PRODUCT = {
    "Title": "T-shirt 2",
    "Price": "$102.15",
    "Rating": "Rating: ⭐ 3.9 / 5",
    "Colors": "3 Colors",
    "Size": "Size: M",
    "Gender": "Gender: Women",
    "Timestamp": "2026-07-19T10:00:00",
}


def test_transform_constants() -> None:
    """Required transformation configuration should be available."""
    assert USD_TO_IDR_RATE == 16_000

    assert REQUIRED_COLUMNS == [
        "Title",
        "Price",
        "Rating",
        "Colors",
        "Size",
        "Gender",
        "Timestamp",
    ]


def test_validate_input_data_from_list() -> None:
    """A list of dictionaries should become a DataFrame."""
    dataframe = validate_input_data(
        [VALID_RAW_PRODUCT]
    )

    assert isinstance(dataframe, pd.DataFrame)
    assert list(dataframe.columns) == REQUIRED_COLUMNS
    assert len(dataframe) == 1


def test_validate_input_data_from_dataframe() -> None:
    """An existing DataFrame should be copied and validated."""
    source_dataframe = pd.DataFrame(
        [VALID_RAW_PRODUCT]
    )

    result = validate_input_data(source_dataframe)

    assert isinstance(result, pd.DataFrame)
    assert result.equals(source_dataframe)
    assert result is not source_dataframe


@pytest.mark.parametrize(
    "invalid_data",
    [
        None,
        [],
    ],
)
def test_validate_input_data_rejects_empty_data(
    invalid_data,
) -> None:
    """Empty input should be rejected."""
    with pytest.raises(
        ValueError,
        match="Input data must not",
    ):
        validate_input_data(invalid_data)


def test_validate_input_data_rejects_invalid_type() -> None:
    """Unsupported input types should be rejected."""
    with pytest.raises(
        TypeError,
        match="Input data must be a list",
    ):
        validate_input_data("invalid input")


def test_validate_input_data_rejects_missing_columns() -> None:
    """Missing required columns should be rejected."""
    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        validate_input_data(
            [
                {
                    "Title": "T-shirt",
                    "Price": "$100.00",
                }
            ]
        )


@pytest.mark.parametrize(
    ("raw_value", "expected_value"),
    [
        ("T-shirt 2", "T-shirt 2"),
        ("  Jacket 3  ", "Jacket 3"),
        ("Unknown Product", None),
        ("unknown product", None),
        ("", None),
        (None, None),
    ],
)
def test_clean_title(
    raw_value,
    expected_value,
) -> None:
    """Raw titles should be cleaned correctly."""
    assert clean_title(raw_value) == expected_value


@pytest.mark.parametrize(
    ("raw_value", "expected_value"),
    [
        ("$102.15", 1_634_400.0),
        ("$100", 1_600_000.0),
        ("$1,000.50", 16_008_000.0),
        ("Price Unavailable", None),
        ("", None),
        ("invalid", None),
        (None, None),
        ("$-10", None),
    ],
)
def test_clean_price(
    raw_value,
    expected_value,
) -> None:
    """USD prices should be converted into rupiah."""
    assert clean_price(raw_value) == expected_value


def test_clean_price_with_custom_exchange_rate() -> None:
    """A custom exchange rate should be supported."""
    assert clean_price(
        "$10",
        exchange_rate=15_000,
    ) == 150_000.0


def test_clean_price_rejects_invalid_exchange_rate() -> None:
    """Invalid exchange rates should produce no price."""
    assert clean_price(
        "$10",
        exchange_rate=0,
    ) is None


@pytest.mark.parametrize(
    ("raw_value", "expected_value"),
    [
        ("Rating: ⭐ 3.9 / 5", 3.9),
        ("Rating: 4.5 / 5", 4.5),
        ("4 / 5", 4.0),
        ("Rating: Not Rated", None),
        ("Rating: Invalid Rating / 5", None),
        ("Rating: 6 / 5", None),
        ("", None),
        (None, None),
    ],
)
def test_clean_rating(
    raw_value,
    expected_value,
) -> None:
    """Raw ratings should become float values."""
    assert clean_rating(raw_value) == expected_value


@pytest.mark.parametrize(
    ("raw_value", "expected_value"),
    [
        ("3 Colors", 3),
        ("1 Color", 1),
        ("10 Colors", 10),
        ("0 Colors", None),
        ("No Colors", None),
        ("", None),
        (None, None),
    ],
)
def test_clean_colors(
    raw_value,
    expected_value,
) -> None:
    """Color descriptions should become integers."""
    assert clean_colors(raw_value) == expected_value


@pytest.mark.parametrize(
    ("raw_value", "expected_value"),
    [
        ("Size: M", "M"),
        ("Size : XL", "XL"),
        ("size: S", "S"),
        ("L", "L"),
        ("", None),
        (None, None),
    ],
)
def test_clean_size(
    raw_value,
    expected_value,
) -> None:
    """Size prefixes should be removed."""
    assert clean_size(raw_value) == expected_value


@pytest.mark.parametrize(
    ("raw_value", "expected_value"),
    [
        ("Gender: Women", "Women"),
        ("Gender : Men", "Men"),
        ("gender: Unisex", "Unisex"),
        ("Women", "Women"),
        ("", None),
        (None, None),
    ],
)
def test_clean_gender(
    raw_value,
    expected_value,
) -> None:
    """Gender prefixes should be removed."""
    assert clean_gender(raw_value) == expected_value


def test_clean_timestamp() -> None:
    """Valid timestamps should be normalized."""
    result = clean_timestamp(
        "2026-07-19T10:00:00"
    )

    assert result == "2026-07-19T10:00:00"


@pytest.mark.parametrize(
    "raw_value",
    [
        None,
        "",
        "not-a-timestamp",
    ],
)
def test_clean_timestamp_rejects_invalid_value(
    raw_value,
) -> None:
    """Invalid timestamps should return None."""
    assert clean_timestamp(raw_value) is None


def test_remove_invalid_rows() -> None:
    """Rows containing null values should be removed."""
    dataframe = pd.DataFrame(
        [
            {
                "Title": "Product A",
                "Price": 100_000.0,
                "Rating": 4.0,
                "Colors": 2,
                "Size": "M",
                "Gender": "Women",
                "Timestamp": "2026-07-19T10:00:00",
            },
            {
                "Title": None,
                "Price": 100_000.0,
                "Rating": 4.0,
                "Colors": 2,
                "Size": "M",
                "Gender": "Women",
                "Timestamp": "2026-07-19T10:00:00",
            },
        ]
    )

    result = remove_invalid_rows(dataframe)

    assert len(result) == 1
    assert result.iloc[0]["Title"] == "Product A"


def test_remove_invalid_rows_rejects_empty_dataframe() -> None:
    """Empty dataframes should be rejected."""
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        remove_invalid_rows(pd.DataFrame())


def test_remove_duplicate_rows() -> None:
    """Identical products should only remain once."""
    dataframe = pd.DataFrame(
        [
            {
                "Title": "Product A",
                "Price": 100_000.0,
                "Rating": 4.0,
                "Colors": 2,
                "Size": "M",
                "Gender": "Women",
                "Timestamp": "2026-07-19T10:00:00",
            },
            {
                "Title": "Product A",
                "Price": 100_000.0,
                "Rating": 4.0,
                "Colors": 2,
                "Size": "M",
                "Gender": "Women",
                "Timestamp": "2026-07-19T10:01:00",
            },
        ]
    )

    result = remove_duplicate_rows(dataframe)

    assert len(result) == 1


def test_remove_duplicate_rows_rejects_empty_dataframe() -> None:
    """Empty dataframes should be rejected."""
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        remove_duplicate_rows(pd.DataFrame())


def test_enforce_data_types() -> None:
    """Final columns should use the required data types."""
    dataframe = pd.DataFrame(
        [
            {
                "Title": "Product A",
                "Price": 100_000,
                "Rating": 4,
                "Colors": 2,
                "Size": "M",
                "Gender": "Women",
                "Timestamp": "2026-07-19T10:00:00",
            }
        ]
    )

    result = enforce_data_types(dataframe)

    assert str(result["Title"].dtype) == "string"
    assert str(result["Price"].dtype) == "float64"
    assert str(result["Rating"].dtype) == "float64"
    assert str(result["Colors"].dtype) == "int64"
    assert str(result["Size"].dtype) == "string"
    assert str(result["Gender"].dtype) == "string"
    assert str(result["Timestamp"].dtype) == "string"


def test_enforce_data_types_rejects_empty_dataframe() -> None:
    """Empty dataframes should not be type-converted."""
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        enforce_data_types(pd.DataFrame())


def test_transform_products() -> None:
    """Raw products should become a clean DataFrame."""
    raw_products = [
        VALID_RAW_PRODUCT,
        {
            "Title": "Jacket 3",
            "Price": "$150.00",
            "Rating": "Rating: ⭐ 4.5 / 5",
            "Colors": "5 Colors",
            "Size": "Size: L",
            "Gender": "Gender: Men",
            "Timestamp": "2026-07-19T10:00:00",
        },
    ]

    result = transform_products(raw_products)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 2

    assert result.iloc[0]["Title"] == "T-shirt 2"
    assert result.iloc[0]["Price"] == 1_634_400.0
    assert result.iloc[0]["Rating"] == 3.9
    assert result.iloc[0]["Colors"] == 3
    assert result.iloc[0]["Size"] == "M"
    assert result.iloc[0]["Gender"] == "Women"


def test_transform_products_removes_invalid_rows() -> None:
    """Invalid products should be removed."""
    invalid_products = [
        {
            "Title": "Unknown Product",
            "Price": "$100.00",
            "Rating": "Rating: ⭐ 4.0 / 5",
            "Colors": "3 Colors",
            "Size": "Size: M",
            "Gender": "Gender: Men",
            "Timestamp": "2026-07-19T10:00:00",
        },
        {
            "Title": "Product A",
            "Price": "Price Unavailable",
            "Rating": "Rating: ⭐ 4.0 / 5",
            "Colors": "3 Colors",
            "Size": "Size: M",
            "Gender": "Gender: Men",
            "Timestamp": "2026-07-19T10:00:00",
        },
        {
            "Title": "Product B",
            "Price": "$100.00",
            "Rating": "Rating: Not Rated",
            "Colors": "3 Colors",
            "Size": "Size: M",
            "Gender": "Gender: Men",
            "Timestamp": "2026-07-19T10:00:00",
        },
        VALID_RAW_PRODUCT,
    ]

    result = transform_products(invalid_products)

    assert len(result) == 1
    assert result.iloc[0]["Title"] == "T-shirt 2"


def test_transform_products_removes_duplicates() -> None:
    """Duplicate products should only appear once."""
    duplicate_product = VALID_RAW_PRODUCT.copy()
    duplicate_product["Timestamp"] = (
        "2026-07-19T10:01:00"
    )

    result = transform_products(
        [
            VALID_RAW_PRODUCT,
            duplicate_product,
        ]
    )

    assert len(result) == 1


def test_transform_products_rejects_all_invalid_rows() -> None:
    """Transformation should fail when no valid products remain."""
    raw_products = [
        {
            "Title": "Unknown Product",
            "Price": "Price Unavailable",
            "Rating": "Rating: Not Rated",
            "Colors": "No Colors",
            "Size": "",
            "Gender": "",
            "Timestamp": "invalid",
        }
    ]

    with pytest.raises(
        ValueError,
        match="No valid product data remains",
    ):
        transform_products(raw_products)


def test_validate_transformed_data() -> None:
    """A valid transformed DataFrame should pass validation."""
    dataframe = transform_products(
        [VALID_RAW_PRODUCT]
    )

    validate_transformed_data(dataframe)


def test_validate_transformed_data_rejects_empty_data() -> None:
    """Empty transformed data should be rejected."""
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        validate_transformed_data(pd.DataFrame())