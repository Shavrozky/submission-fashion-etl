"""Unit tests for the extraction module."""

from unittest.mock import Mock, patch

import pytest
import requests

from utils.extract import (
    BASE_URL,
    TOTAL_PAGES,
    build_page_url,
    create_session,
    extract_product_card,
    fetching_content,
    parse_products,
    scrape_main,
    scrape_page,
)


SAMPLE_HTML = """
<html>
    <body>
        <div class="collection-card">
            <div style="position: relative;"></div>

            <div class="product-details">
                <h3 class="product-title">T-shirt 2</h3>

                <div class="price-container">
                    <span class="price">$102.15</span>
                </div>

                <p>Rating: ⭐ 3.9 / 5</p>
                <p>3 Colors</p>
                <p>Size: M</p>
                <p>Gender: Women</p>
            </div>
        </div>

        <div class="collection-card">
            <div style="position: relative;"></div>

            <div class="product-details">
                <h3 class="product-title">Pants 16</h3>

                <div class="price-container">
                    <p class="price">Price Unavailable</p>
                </div>

                <p>Rating: Not Rated</p>
                <p>8 Colors</p>
                <p>Size: S</p>
                <p>Gender: Men</p>
            </div>
        </div>
    </body>
</html>
"""


def test_extract_constants() -> None:
    """Extraction configuration should target the correct source."""
    assert BASE_URL == "https://fashion-studio.dicoding.dev/"
    assert TOTAL_PAGES == 50


def test_create_session() -> None:
    """Session should be created with default headers."""
    session = create_session()

    assert isinstance(session, requests.Session)
    assert "User-Agent" in session.headers

    session.close()


@pytest.mark.parametrize(
    ("page_number", "expected_url"),
    [
        (1, "https://fashion-studio.dicoding.dev/"),
        (2, "https://fashion-studio.dicoding.dev/page2"),
        (50, "https://fashion-studio.dicoding.dev/page50"),
    ],
)
def test_build_page_url(
    page_number: int,
    expected_url: str,
) -> None:
    """Page URLs should follow the website pagination pattern."""
    assert build_page_url(page_number) == expected_url


def test_build_page_url_rejects_invalid_page() -> None:
    """Page numbers below one should be rejected."""
    with pytest.raises(
        ValueError,
        match="Page number must be greater than or equal to 1",
    ):
        build_page_url(0)


def test_fetching_content_success() -> None:
    """Fetching content should return response bytes."""
    mock_session = Mock()
    mock_response = Mock()

    mock_response.content = b"<html><body>Test</body></html>"
    mock_response.raise_for_status.return_value = None
    mock_session.get.return_value = mock_response

    result = fetching_content(
        url="https://example.com",
        session=mock_session,
    )

    assert result == b"<html><body>Test</body></html>"

    mock_session.get.assert_called_once_with(
        "https://example.com",
        timeout=15,
    )

    mock_response.raise_for_status.assert_called_once()


def test_fetching_content_rejects_empty_url() -> None:
    """Empty URLs should be rejected before making a request."""
    with pytest.raises(
        ValueError,
        match="URL must not be empty",
    ):
        fetching_content("")


def test_fetching_content_handles_request_error() -> None:
    """Request failures should be converted into RuntimeError."""
    mock_session = Mock()

    mock_session.get.side_effect = (
        requests.exceptions.ConnectionError(
            "Connection failed"
        )
    )

    with pytest.raises(
        RuntimeError,
        match="Failed to fetch content",
    ):
        fetching_content(
            url="https://example.com",
            session=mock_session,
        )


def test_parse_products() -> None:
    """Product cards should be converted into raw dictionaries."""
    products = parse_products(
        SAMPLE_HTML,
        extracted_at="2026-07-19T10:00:00",
    )

    assert len(products) == 2

    assert products[0] == {
        "Title": "T-shirt 2",
        "Price": "$102.15",
        "Rating": "Rating: ⭐ 3.9 / 5",
        "Colors": "3 Colors",
        "Size": "Size: M",
        "Gender": "Gender: Women",
        "Timestamp": "2026-07-19T10:00:00",
    }

    assert products[1] == {
        "Title": "Pants 16",
        "Price": "Price Unavailable",
        "Rating": "Rating: Not Rated",
        "Colors": "8 Colors",
        "Size": "Size: S",
        "Gender": "Gender: Men",
        "Timestamp": "2026-07-19T10:00:00",
    }


def test_parse_products_rejects_empty_html() -> None:
    """Empty HTML should not be parsed."""
    with pytest.raises(
        ValueError,
        match="HTML content must not be empty",
    ):
        parse_products(b"")


def test_extract_product_card_rejects_none() -> None:
    """A missing product card should be rejected."""
    with pytest.raises(
        ValueError,
        match="Product card must not be None",
    ):
        extract_product_card(
            None,
            "2026-07-19T10:00:00",
        )


@patch("utils.extract.fetching_content")
def test_scrape_page(
    mock_fetching_content: Mock,
) -> None:
    """One page should be fetched and parsed."""
    mock_fetching_content.return_value = SAMPLE_HTML.encode(
        "utf-8"
    )

    products = scrape_page(
        page_number=1,
        session=Mock(),
    )

    assert len(products) == 2
    assert products[0]["Title"] == "T-shirt 2"
    assert products[1]["Price"] == "Price Unavailable"

    mock_fetching_content.assert_called_once()


@patch("utils.extract.scrape_page")
def test_scrape_main(
    mock_scrape_page: Mock,
) -> None:
    """Multiple pages should be combined into one list."""
    mock_scrape_page.side_effect = [
        [{"Title": "Product 1"}],
        [{"Title": "Product 2"}],
        [{"Title": "Product 3"}],
    ]

    mock_session = Mock()

    products = scrape_main(
        start_page=1,
        end_page=3,
        session=mock_session,
    )

    assert len(products) == 3

    assert products == [
        {"Title": "Product 1"},
        {"Title": "Product 2"},
        {"Title": "Product 3"},
    ]

    assert mock_scrape_page.call_count == 3


def test_scrape_main_rejects_invalid_start_page() -> None:
    """Start page below one should be rejected."""
    with pytest.raises(
        ValueError,
        match="Start page must be at least 1",
    ):
        scrape_main(
            start_page=0,
            end_page=2,
        )


def test_scrape_main_rejects_invalid_page_range() -> None:
    """End page below start page should be rejected."""
    with pytest.raises(
        ValueError,
        match=(
            "End page must be greater than or equal "
            "to start page"
        ),
    ):
        scrape_main(
            start_page=3,
            end_page=2,
        )


@patch("utils.extract.scrape_page")
def test_scrape_main_handles_scraping_failure(
    mock_scrape_page: Mock,
) -> None:
    """Scraping failures should be wrapped in RuntimeError."""
    mock_scrape_page.side_effect = RuntimeError(
        "Page failed"
    )

    with pytest.raises(
        RuntimeError,
        match="An error occurred during scraping",
    ):
        scrape_main(
            start_page=1,
            end_page=2,
            session=Mock(),
        )