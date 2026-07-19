"""Module for loading transformed Fashion Studio product data."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import pandas as pd
from google.oauth2 import service_account
from googleapiclient.discovery import build
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from utils.transform import REQUIRED_COLUMNS


GOOGLE_SHEETS_SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
]

DEFAULT_CSV_PATH = "products.csv"
DEFAULT_GOOGLE_RANGE = "products!A1"
DEFAULT_POSTGRES_TABLE = "fashion_products"


def validate_load_data(data: Any) -> pd.DataFrame:
    """Validate data before loading it into a repository.

    Args:
        data: Transformed product data.

    Returns:
        A copied and validated DataFrame.

    Raises:
        TypeError: If input is not a pandas DataFrame.
        ValueError: If data is empty, incomplete, or contains null values.
        RuntimeError: If validation unexpectedly fails.
    """
    try:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                "Load data must be a pandas DataFrame."
            )

        if data.empty:
            raise ValueError(
                "Load data must not be empty."
            )

        missing_columns = [
            column
            for column in REQUIRED_COLUMNS
            if column not in data.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing required columns for loading: "
                f"{', '.join(missing_columns)}"
            )

        validated_data = data[REQUIRED_COLUMNS].copy()

        if validated_data.isnull().any().any():
            raise ValueError(
                "Load data must not contain null values."
            )

        if validated_data.duplicated().any():
            raise ValueError(
                "Load data must not contain duplicate rows."
            )

        return validated_data

    except (TypeError, ValueError):
        raise

    except Exception as error:
        raise RuntimeError(
            f"Failed to validate load data: {error}"
        ) from error


def save_to_csv(
    data: pd.DataFrame,
    output_path: str | Path = DEFAULT_CSV_PATH,
) -> Path:
    """Save transformed product data into a CSV file.

    Args:
        data: Clean product DataFrame.
        output_path: Destination CSV file path.

    Returns:
        Path to the generated CSV file.

    Raises:
        ValueError: If output path is invalid.
        RuntimeError: If writing the CSV file fails.
    """
    if output_path is None:
        raise ValueError(
            "CSV output path must not be None."
        )

    output_path = Path(output_path)

    if output_path.suffix.lower() != ".csv":
        raise ValueError(
            "CSV output path must use the .csv extension."
        )

    validated_data = validate_load_data(data)

    try:
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        validated_data.to_csv(
            output_path,
            index=False,
            encoding="utf-8",
        )

        return output_path

    except (OSError, PermissionError) as error:
        raise RuntimeError(
            f"Failed to save CSV file: {error}"
        ) from error

    except Exception as error:
        raise RuntimeError(
            f"Unexpected error while saving CSV: {error}"
        ) from error


def create_google_sheets_service(
    credentials_file: str | Path,
):
    """Create an authenticated Google Sheets API service.

    Args:
        credentials_file: Path to the service account JSON file.

    Returns:
        Authenticated Google Sheets API service.

    Raises:
        ValueError: If credentials path is missing.
        FileNotFoundError: If credentials file does not exist.
        RuntimeError: If service creation fails.
    """
    if credentials_file is None:
        raise ValueError(
            "Google credentials file must not be None."
        )

    credentials_path = Path(credentials_file)

    if not credentials_path.exists():
        raise FileNotFoundError(
            f"Google credentials file was not found: "
            f"{credentials_path}"
        )

    try:
        credentials = (
            service_account.Credentials
            .from_service_account_file(
                str(credentials_path),
                scopes=GOOGLE_SHEETS_SCOPES,
            )
        )

        service = build(
            "sheets",
            "v4",
            credentials=credentials,
            cache_discovery=False,
        )

        return service

    except Exception as error:
        raise RuntimeError(
            f"Failed to create Google Sheets service: {error}"
        ) from error


def convert_to_native_value(value: Any) -> Any:
    """Convert pandas or NumPy values into API-compatible values.

    Args:
        value: Cell value from a DataFrame.

    Returns:
        Native Python value suitable for serialization.
    """
    if pd.isna(value):
        return ""

    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass

    return value


def dataframe_to_sheet_values(
    data: pd.DataFrame,
) -> list[list[Any]]:
    """Convert a DataFrame into Google Sheets row values.

    The first row contains column headers.

    Args:
        data: Product DataFrame.

    Returns:
        Nested list containing headers and row values.
    """
    validated_data = validate_load_data(data)

    try:
        headers = list(validated_data.columns)

        rows = [
            [
                convert_to_native_value(value)
                for value in row
            ]
            for row in validated_data.itertuples(
                index=False,
                name=None,
            )
        ]

        return [headers, *rows]

    except Exception as error:
        raise RuntimeError(
            f"Failed to convert data for Google Sheets: {error}"
        ) from error


def save_to_google_sheets(
    data: pd.DataFrame,
    spreadsheet_id: str,
    credentials_file: str | Path = (
        "google-sheets-api.json"
    ),
    sheet_range: str = DEFAULT_GOOGLE_RANGE,
    service: Any | None = None,
) -> dict[str, Any]:
    """Save transformed data into Google Sheets.

    Args:
        data: Clean product DataFrame.
        spreadsheet_id: Google Sheets spreadsheet ID.
        credentials_file: Service account JSON path.
        sheet_range: Target worksheet and starting cell.
        service: Optional pre-created Sheets API service.

    Returns:
        Google Sheets API update response.

    Raises:
        ValueError: If spreadsheet ID or range is missing.
        RuntimeError: If the Google Sheets operation fails.
    """
    if not spreadsheet_id or not spreadsheet_id.strip():
        raise ValueError(
            "Google Sheets spreadsheet ID must not be empty."
        )

    if not sheet_range or not sheet_range.strip():
        raise ValueError(
            "Google Sheets range must not be empty."
        )

    values = dataframe_to_sheet_values(data)

    try:
        active_service = service or (
            create_google_sheets_service(
                credentials_file
            )
        )

        values_resource = (
            active_service
            .spreadsheets()
            .values()
        )

        values_resource.clear(
            spreadsheetId=spreadsheet_id,
            range=sheet_range,
            body={},
        ).execute()

        response = values_resource.update(
            spreadsheetId=spreadsheet_id,
            range=sheet_range,
            valueInputOption="RAW",
            body={
                "values": values,
            },
        ).execute()

        return response

    except Exception as error:
        raise RuntimeError(
            f"Failed to save data to Google Sheets: {error}"
        ) from error


def validate_table_name(table_name: str) -> str:
    """Validate a PostgreSQL table name.

    Args:
        table_name: Target PostgreSQL table name.

    Returns:
        Validated table name.

    Raises:
        ValueError: If the table name is invalid.
    """
    if not table_name or not table_name.strip():
        raise ValueError(
            "PostgreSQL table name must not be empty."
        )

    normalized_table_name = table_name.strip()

    if not re.fullmatch(
        r"[A-Za-z_][A-Za-z0-9_]*",
        normalized_table_name,
    ):
        raise ValueError(
            "PostgreSQL table name may only contain "
            "letters, numbers, and underscores, and must "
            "not start with a number."
        )

    return normalized_table_name


def create_postgresql_engine(
    database_url: str,
) -> Engine:
    """Create a SQLAlchemy PostgreSQL engine.

    Args:
        database_url: PostgreSQL SQLAlchemy connection URL.

    Returns:
        SQLAlchemy engine.

    Raises:
        ValueError: If database URL is empty.
        RuntimeError: If engine creation fails.
    """
    if not database_url or not database_url.strip():
        raise ValueError(
            "PostgreSQL database URL must not be empty."
        )

    try:
        return create_engine(
            database_url,
            pool_pre_ping=True,
        )

    except Exception as error:
        raise RuntimeError(
            f"Failed to create PostgreSQL engine: {error}"
        ) from error


def save_to_postgresql(
    data: pd.DataFrame,
    database_url: str | None = None,
    table_name: str = DEFAULT_POSTGRES_TABLE,
    if_exists: str = "replace",
    engine: Engine | Any | None = None,
) -> int:
    """Save transformed data into PostgreSQL.

    Args:
        data: Clean product DataFrame.
        database_url: PostgreSQL SQLAlchemy connection URL.
        table_name: Destination database table.
        if_exists: Behavior when the table already exists.
        engine: Optional existing SQLAlchemy engine.

    Returns:
        Number of rows submitted to PostgreSQL.

    Raises:
        ValueError: If configuration is invalid.
        RuntimeError: If database loading fails.
    """
    validated_data = validate_load_data(data)
    validated_table_name = validate_table_name(
        table_name
    )

    allowed_if_exists = {
        "fail",
        "replace",
        "append",
    }

    if if_exists not in allowed_if_exists:
        raise ValueError(
            "if_exists must be one of: "
            "fail, replace, or append."
        )

    if engine is None and (
        database_url is None
        or not database_url.strip()
    ):
        raise ValueError(
            "PostgreSQL database URL is required "
            "when no engine is provided."
        )

    active_engine = engine

    try:
        if active_engine is None:
            active_engine = create_postgresql_engine(
                database_url
            )

        validated_data.to_sql(
            name=validated_table_name,
            con=active_engine,
            if_exists=if_exists,
            index=False,
            method="multi",
        )

        return len(validated_data)

    except Exception as error:
        raise RuntimeError(
            f"Failed to save data to PostgreSQL: {error}"
        ) from error

    finally:
        if engine is None and active_engine is not None:
            active_engine.dispose()


def load_all_repositories(
    data: pd.DataFrame,
    csv_path: str | Path = DEFAULT_CSV_PATH,
    spreadsheet_id: str | None = None,
    credentials_file: str | Path = (
        "google-sheets-api.json"
    ),
    sheet_range: str = DEFAULT_GOOGLE_RANGE,
    database_url: str | None = None,
    table_name: str = DEFAULT_POSTGRES_TABLE,
) -> dict[str, Any]:
    """Load product data into CSV, Google Sheets, and PostgreSQL.

    Args:
        data: Clean transformed product data.
        csv_path: Target CSV path.
        spreadsheet_id: Google Sheets spreadsheet ID.
        credentials_file: Google service account file.
        sheet_range: Google Sheets destination range.
        database_url: PostgreSQL connection URL.
        table_name: PostgreSQL table name.

    Returns:
        Summary of all completed load operations.

    Raises:
        ValueError: If required configuration is missing.
        RuntimeError: If one of the load operations fails.
    """
    if not spreadsheet_id:
        raise ValueError(
            "Spreadsheet ID is required to load "
            "all repositories."
        )

    if not database_url:
        raise ValueError(
            "PostgreSQL database URL is required to "
            "load all repositories."
        )

    try:
        csv_result = save_to_csv(
            data=data,
            output_path=csv_path,
        )

        google_result = save_to_google_sheets(
            data=data,
            spreadsheet_id=spreadsheet_id,
            credentials_file=credentials_file,
            sheet_range=sheet_range,
        )

        postgresql_rows = save_to_postgresql(
            data=data,
            database_url=database_url,
            table_name=table_name,
        )

        return {
            "csv_path": str(csv_result),
            "google_sheets": google_result,
            "postgresql_rows": postgresql_rows,
        }

    except Exception as error:
        raise RuntimeError(
            f"Failed to load all repositories: {error}"
        ) from error


def get_load_configuration() -> dict[str, str | None]:
    """Read loading configuration from environment variables.

    Returns:
        Dictionary containing external repository configuration.
    """
    return {
        "spreadsheet_id": os.getenv(
            "GOOGLE_SHEETS_SPREADSHEET_ID"
        ),
        "sheet_range": os.getenv(
            "GOOGLE_SHEETS_RANGE",
            DEFAULT_GOOGLE_RANGE,
        ),
        "credentials_file": os.getenv(
            "GOOGLE_SERVICE_ACCOUNT_FILE",
            "google-sheets-api.json",
        ),
        "database_url": os.getenv(
            "POSTGRESQL_URL"
        ),
        "table_name": os.getenv(
            "POSTGRESQL_TABLE",
            DEFAULT_POSTGRES_TABLE,
        ),
    }