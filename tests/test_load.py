"""Initial smoke tests for utils.load."""

from utils import load


def test_save_to_csv_is_callable() -> None:
    """The CSV loader entry point should already be exposed."""
    assert callable(load.save_to_csv)
