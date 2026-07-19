"""Unit tests for the loading module."""

from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from utils.load import (
    DEFAULT_CSV_PATH,
    DEFAULT_GOOGLE_RANGE,
    DEFAULT_POSTGRES_TABLE,
    convert_to_native_value,
    create_google_sheets_service,
    create_postgresql_engine,
    dataframe_to_sheet_values,
    get_load_configuration,
    load_all_repositories,
    save_to_csv,
    save_to_google_sheets,
    save_to_postgresql,
    validate_load_data,
    validate_table_name,
)


VALID_DATAFRAME = pd.DataFrame(
    [
        {
            "Title": "T-shirt 2",
            "Price": 1_634_400.0,
            "Rating": 3.9,
            "Colors": 3,
            "Size": "M",
            "Gender": "Women",
            "Timestamp": "2026-07-19T10:00:00",
        },
        {
            "Title": "Jacket 3",
            "Price": 2_400_000.0,
            "Rating": 4.5,
            "Colors": 5,
            "Size": "L",
            "Gender": "Men",
            "Timestamp": "2026-07-19T10:00:00",
        },
    ]
)


def test_load_constants() -> None:
    """Default load configuration should be correct."""
    assert DEFAULT_CSV_PATH == "products.csv"
    assert DEFAULT_GOOGLE_RANGE == "products!A1"
    assert DEFAULT_POSTGRES_TABLE == "fashion_products"


def test_validate_load_data() -> None:
    """A valid DataFrame should be copied."""
    result = validate_load_data(
        VALID_DATAFRAME
    )

    assert isinstance(result, pd.DataFrame)
    assert result.equals(VALID_DATAFRAME)
    assert result is not VALID_DATAFRAME


def test_validate_load_data_rejects_non_dataframe() -> None:
    """Non-DataFrame input should be rejected."""
    with pytest.raises(
        TypeError,
        match="must be a pandas DataFrame",
    ):
        validate_load_data(
            [{"Title": "Product"}]
        )


def test_validate_load_data_rejects_empty_dataframe() -> None:
    """Empty DataFrames should be rejected."""
    with pytest.raises(
        ValueError,
        match="must not be empty",
    ):
        validate_load_data(pd.DataFrame())


def test_validate_load_data_rejects_missing_columns() -> None:
    """Incomplete DataFrames should be rejected."""
    incomplete_data = pd.DataFrame(
        [
            {
                "Title": "Product A",
                "Price": 100_000.0,
            }
        ]
    )

    with pytest.raises(
        ValueError,
        match="Missing required columns",
    ):
        validate_load_data(incomplete_data)


def test_validate_load_data_rejects_null_values() -> None:
    """Rows containing null values should be rejected."""
    invalid_data = VALID_DATAFRAME.copy()
    invalid_data.loc[0, "Title"] = None

    with pytest.raises(
        ValueError,
        match="must not contain null",
    ):
        validate_load_data(invalid_data)


def test_validate_load_data_rejects_duplicates() -> None:
    """Duplicate rows should be rejected."""
    duplicate_data = pd.concat(
        [
            VALID_DATAFRAME.iloc[[0]],
            VALID_DATAFRAME.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="must not contain duplicate",
    ):
        validate_load_data(duplicate_data)


def test_save_to_csv(
    tmp_path: Path,
) -> None:
    """Clean data should be saved into a CSV file."""
    output_path = tmp_path / "products.csv"

    result = save_to_csv(
        VALID_DATAFRAME,
        output_path,
    )

    assert result == output_path
    assert output_path.exists()

    loaded_data = pd.read_csv(output_path)

    assert len(loaded_data) == 2
    assert list(loaded_data.columns) == list(
        VALID_DATAFRAME.columns
    )
    assert loaded_data.iloc[0]["Title"] == "T-shirt 2"


def test_save_to_csv_creates_parent_directory(
    tmp_path: Path,
) -> None:
    """Missing output directories should be created."""
    output_path = (
        tmp_path
        / "nested"
        / "directory"
        / "products.csv"
    )

    save_to_csv(
        VALID_DATAFRAME,
        output_path,
    )

    assert output_path.exists()


@pytest.mark.parametrize(
    "invalid_path",
    [
        "products.txt",
        "products.json",
        "products",
    ],
)
def test_save_to_csv_rejects_invalid_extension(
    invalid_path: str,
) -> None:
    """Only .csv output paths should be accepted."""
    with pytest.raises(
        ValueError,
        match=".csv extension",
    ):
        save_to_csv(
            VALID_DATAFRAME,
            invalid_path,
        )


def test_save_to_csv_rejects_none_path() -> None:
    """A missing CSV path should be rejected."""
    with pytest.raises(
        ValueError,
        match="must not be None",
    ):
        save_to_csv(
            VALID_DATAFRAME,
            None,
        )


@patch("utils.load.service_account.Credentials")
@patch("utils.load.build")
def test_create_google_sheets_service(
    mock_build: Mock,
    mock_credentials_class: Mock,
    tmp_path: Path,
) -> None:
    """Google Sheets service should use service account credentials."""
    credentials_file = (
        tmp_path
        / "google-sheets-api.json"
    )
    credentials_file.write_text(
        "{}",
        encoding="utf-8",
    )

    mock_credentials = Mock()

    mock_credentials_class\
        .from_service_account_file\
        .return_value = mock_credentials

    mock_service = Mock()
    mock_build.return_value = mock_service

    result = create_google_sheets_service(
        credentials_file
    )

    assert result is mock_service

    mock_credentials_class\
        .from_service_account_file\
        .assert_called_once()

    mock_build.assert_called_once_with(
        "sheets",
        "v4",
        credentials=mock_credentials,
        cache_discovery=False,
    )


def test_create_google_sheets_service_rejects_none() -> None:
    """Missing credentials paths should be rejected."""
    with pytest.raises(
        ValueError,
        match="must not be None",
    ):
        create_google_sheets_service(None)


def test_create_google_sheets_service_rejects_missing_file(
    tmp_path: Path,
) -> None:
    """Missing credential files should be rejected."""
    missing_file = (
        tmp_path
        / "missing.json"
    )

    with pytest.raises(
        FileNotFoundError,
        match="was not found",
    ):
        create_google_sheets_service(
            missing_file
        )


@pytest.mark.parametrize(
    ("raw_value", "expected_value"),
    [
        (1, 1),
        (3.9, 3.9),
        ("Women", "Women"),
        (None, ""),
    ],
)
def test_convert_to_native_value(
    raw_value,
    expected_value,
) -> None:
    """Values should become API-compatible Python values."""
    assert (
        convert_to_native_value(raw_value)
        == expected_value
    )


def test_dataframe_to_sheet_values() -> None:
    """DataFrames should become nested row lists."""
    values = dataframe_to_sheet_values(
        VALID_DATAFRAME
    )

    assert values[0] == [
        "Title",
        "Price",
        "Rating",
        "Colors",
        "Size",
        "Gender",
        "Timestamp",
    ]

    assert len(values) == 3
    assert values[1][0] == "T-shirt 2"
    assert values[1][1] == 1_634_400.0


def test_save_to_google_sheets() -> None:
    """Google Sheets loader should clear and update data."""
    mock_service = Mock()
    mock_values_resource = Mock()

    mock_service\
        .spreadsheets\
        .return_value\
        .values\
        .return_value = mock_values_resource

    mock_clear_request = Mock()
    mock_update_request = Mock()

    mock_values_resource.clear.return_value = (
        mock_clear_request
    )
    mock_values_resource.update.return_value = (
        mock_update_request
    )

    mock_clear_request.execute.return_value = {}
    mock_update_request.execute.return_value = {
        "updatedRows": 3,
    }

    result = save_to_google_sheets(
        data=VALID_DATAFRAME,
        spreadsheet_id="spreadsheet-id",
        sheet_range="products!A1",
        service=mock_service,
    )

    assert result == {
        "updatedRows": 3,
    }

    mock_values_resource.clear.assert_called_once_with(
        spreadsheetId="spreadsheet-id",
        range="products!A1",
        body={},
    )

    mock_values_resource.update.assert_called_once()

    mock_clear_request.execute.assert_called_once()
    mock_update_request.execute.assert_called_once()


def test_save_to_google_sheets_rejects_empty_id() -> None:
    """Empty spreadsheet IDs should be rejected."""
    with pytest.raises(
        ValueError,
        match="spreadsheet ID must not be empty",
    ):
        save_to_google_sheets(
            data=VALID_DATAFRAME,
            spreadsheet_id="",
            service=Mock(),
        )


def test_save_to_google_sheets_rejects_empty_range() -> None:
    """Empty Google Sheets ranges should be rejected."""
    with pytest.raises(
        ValueError,
        match="range must not be empty",
    ):
        save_to_google_sheets(
            data=VALID_DATAFRAME,
            spreadsheet_id="spreadsheet-id",
            sheet_range="",
            service=Mock(),
        )


def test_save_to_google_sheets_handles_api_error() -> None:
    """Google Sheets API failures should be wrapped."""
    mock_service = Mock()

    mock_service.spreadsheets.side_effect = RuntimeError(
        "Google API failed"
    )

    with pytest.raises(
        RuntimeError,
        match="Failed to save data to Google Sheets",
    ):
        save_to_google_sheets(
            data=VALID_DATAFRAME,
            spreadsheet_id="spreadsheet-id",
            service=mock_service,
        )


@pytest.mark.parametrize(
    "valid_name",
    [
        "fashion_products",
        "products2026",
        "_products",
        "FashionProducts",
    ],
)
def test_validate_table_name(
    valid_name: str,
) -> None:
    """Valid SQL table names should be accepted."""
    assert validate_table_name(valid_name) == valid_name


@pytest.mark.parametrize(
    "invalid_name",
    [
        "",
        "123products",
        "fashion-products",
        "fashion products",
        "products;drop_table",
    ],
)
def test_validate_table_name_rejects_invalid_name(
    invalid_name: str,
) -> None:
    """Unsafe SQL table names should be rejected."""
    with pytest.raises(ValueError):
        validate_table_name(invalid_name)


@patch("utils.load.create_engine")
def test_create_postgresql_engine(
    mock_create_engine: Mock,
) -> None:
    """PostgreSQL engine should use pool pre-ping."""
    mock_engine = Mock()
    mock_create_engine.return_value = mock_engine

    result = create_postgresql_engine(
        "postgresql+psycopg2://user:pass@localhost/db"
    )

    assert result is mock_engine

    mock_create_engine.assert_called_once_with(
        "postgresql+psycopg2://user:pass@localhost/db",
        pool_pre_ping=True,
    )


def test_create_postgresql_engine_rejects_empty_url() -> None:
    """Empty database URLs should be rejected."""
    with pytest.raises(
        ValueError,
        match="database URL must not be empty",
    ):
        create_postgresql_engine("")


def test_save_to_postgresql_with_existing_engine() -> None:
    """Data should be written using a supplied engine."""
    mock_engine = Mock()

    with patch.object(
        pd.DataFrame,
        "to_sql",
        return_value=None,
    ) as mock_to_sql:
        rows = save_to_postgresql(
            data=VALID_DATAFRAME,
            table_name="fashion_products",
            engine=mock_engine,
        )

    assert rows == 2

    mock_to_sql.assert_called_once_with(
        name="fashion_products",
        con=mock_engine,
        if_exists="replace",
        index=False,
        method="multi",
    )

    mock_engine.dispose.assert_not_called()


@patch("utils.load.create_postgresql_engine")
def test_save_to_postgresql_creates_and_disposes_engine(
    mock_create_engine_function: Mock,
) -> None:
    """Internally created engines should be disposed."""
    mock_engine = Mock()
    mock_create_engine_function.return_value = (
        mock_engine
    )

    with patch.object(
        pd.DataFrame,
        "to_sql",
        return_value=None,
    ):
        rows = save_to_postgresql(
            data=VALID_DATAFRAME,
            database_url=(
                "postgresql+psycopg2://"
                "user:pass@localhost/db"
            ),
        )

    assert rows == 2

    mock_create_engine_function.assert_called_once()
    mock_engine.dispose.assert_called_once()


def test_save_to_postgresql_rejects_missing_connection() -> None:
    """A URL is required when no engine is supplied."""
    with pytest.raises(
        ValueError,
        match="database URL is required",
    ):
        save_to_postgresql(
            data=VALID_DATAFRAME,
            database_url=None,
            engine=None,
        )


def test_save_to_postgresql_rejects_invalid_if_exists() -> None:
    """Unsupported table behavior should be rejected."""
    with pytest.raises(
        ValueError,
        match="if_exists must be one of",
    ):
        save_to_postgresql(
            data=VALID_DATAFRAME,
            database_url="postgresql://example",
            if_exists="invalid",
        )


def test_save_to_postgresql_handles_database_error() -> None:
    """Database failures should be wrapped."""
    mock_engine = Mock()

    with patch.object(
        pd.DataFrame,
        "to_sql",
        side_effect=RuntimeError(
            "Database failed"
        ),
    ):
        with pytest.raises(
            RuntimeError,
            match="Failed to save data to PostgreSQL",
        ):
            save_to_postgresql(
                data=VALID_DATAFRAME,
                engine=mock_engine,
            )


@patch("utils.load.save_to_postgresql")
@patch("utils.load.save_to_google_sheets")
@patch("utils.load.save_to_csv")
def test_load_all_repositories(
    mock_save_csv: Mock,
    mock_save_google: Mock,
    mock_save_postgresql: Mock,
) -> None:
    """All configured repositories should be loaded."""
    mock_save_csv.return_value = Path(
        "products.csv"
    )
    mock_save_google.return_value = {
        "updatedRows": 3,
    }
    mock_save_postgresql.return_value = 2

    result = load_all_repositories(
        data=VALID_DATAFRAME,
        spreadsheet_id="spreadsheet-id",
        database_url="postgresql://database-url",
    )

    assert result == {
        "csv_path": "products.csv",
        "google_sheets": {
            "updatedRows": 3,
        },
        "postgresql_rows": 2,
    }

    mock_save_csv.assert_called_once()
    mock_save_google.assert_called_once()
    mock_save_postgresql.assert_called_once()


def test_load_all_repositories_requires_spreadsheet_id() -> None:
    """Loading all repositories requires Sheets configuration."""
    with pytest.raises(
        ValueError,
        match="Spreadsheet ID is required",
    ):
        load_all_repositories(
            data=VALID_DATAFRAME,
            spreadsheet_id=None,
            database_url="postgresql://database-url",
        )


def test_load_all_repositories_requires_database_url() -> None:
    """Loading all repositories requires PostgreSQL config."""
    with pytest.raises(
        ValueError,
        match="database URL is required",
    ):
        load_all_repositories(
            data=VALID_DATAFRAME,
            spreadsheet_id="spreadsheet-id",
            database_url=None,
        )


@patch.dict(
    "os.environ",
    {
        "GOOGLE_SHEETS_SPREADSHEET_ID": (
            "spreadsheet-id"
        ),
        "GOOGLE_SHEETS_RANGE": "products!A1",
        "GOOGLE_SERVICE_ACCOUNT_FILE": (
            "google-sheets-api.json"
        ),
        "POSTGRESQL_URL": (
            "postgresql+psycopg2://"
            "user:pass@localhost/db"
        ),
        "POSTGRESQL_TABLE": "fashion_products",
    },
    clear=True,
)
def test_get_load_configuration() -> None:
    """Repository configuration should come from environment."""
    configuration = get_load_configuration()

    assert configuration == {
        "spreadsheet_id": "spreadsheet-id",
        "sheet_range": "products!A1",
        "credentials_file": (
            "google-sheets-api.json"
        ),
        "database_url": (
            "postgresql+psycopg2://"
            "user:pass@localhost/db"
        ),
        "table_name": "fashion_products",
    }