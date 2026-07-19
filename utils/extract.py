"""Module for extracting product data from Fashion Studio."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_URL = "https://fashion-studio.dicoding.dev/"
TOTAL_PAGES = 50
REQUEST_TIMEOUT = 15

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    )
}


def create_session() -> requests.Session:
    """Create a requests session with retry configuration.

    Returns:
        A configured requests.Session instance.

    Raises:
        RuntimeError: If the HTTP session cannot be created.
    """
    try:
        retry_strategy = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )

        adapter = HTTPAdapter(max_retries=retry_strategy)

        session = requests.Session()
        session.headers.update(DEFAULT_HEADERS)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    except Exception as error:
        raise RuntimeError(
            f"Failed to create HTTP session: {error}"
        ) from error


def build_page_url(page_number: int) -> str:
    """Build the Fashion Studio URL for a page number.

    Page 1 uses the website root URL, while subsequent pages use
    paths such as /page2, /page3, and so on.

    Args:
        page_number: Page number starting from 1.

    Returns:
        URL for the requested page.

    Raises:
        ValueError: If the page number is below 1.
    """
    if page_number < 1:
        raise ValueError("Page number must be greater than or equal to 1.")

    if page_number == 1:
        return BASE_URL

    return f"{BASE_URL}page{page_number}"


def fetching_content(
    url: str,
    session: requests.Session | None = None,
    timeout: int = REQUEST_TIMEOUT,
) -> bytes:
    """Fetch HTML content from a URL.

    Args:
        url: Website URL to fetch.
        session: Optional requests session.
        timeout: Maximum request duration in seconds.

    Returns:
        Raw HTML content as bytes.

    Raises:
        ValueError: If the URL is empty.
        RuntimeError: If the request fails.
    """
    if not url or not url.strip():
        raise ValueError("URL must not be empty.")

    active_session = session or create_session()

    try:
        response = active_session.get(url, timeout=timeout)
        response.raise_for_status()

        return response.content

    except requests.exceptions.RequestException as error:
        raise RuntimeError(
            f"Failed to fetch content from {url}: {error}"
        ) from error


def extract_text(
    element: Any,
    selector: str,
    default: str | None = None,
) -> str | None:
    """Extract cleaned text from a child element.

    Args:
        element: BeautifulSoup element containing product data.
        selector: CSS selector used to find the child element.
        default: Value returned when the element is not found.

    Returns:
        Cleaned text or the provided default value.
    """
    try:
        selected_element = element.select_one(selector)

        if selected_element is None:
            return default

        text = selected_element.get_text(" ", strip=True)

        return text if text else default

    except (AttributeError, TypeError):
        return default


def extract_product_card(
    card: Any,
    extracted_at: str,
) -> dict[str, Any]:
    """Extract raw product attributes from one product card.

    Args:
        card: BeautifulSoup element representing one product card.
        extracted_at: ISO timestamp representing extraction time.

    Returns:
        Dictionary containing raw product values.

    Raises:
        ValueError: If the product card is missing.
        RuntimeError: If unexpected parsing failure occurs.
    """
    if card is None:
        raise ValueError("Product card must not be None.")

    try:
        title = extract_text(card, ".product-title")

        price_container = card.select_one(".price-container")
        price = (
            price_container.get_text(" ", strip=True)
            if price_container is not None
            else None
        )

        product_details = card.select(".product-details p")

        detail_texts = [
            detail.get_text(" ", strip=True)
            for detail in product_details
        ]

        rating = next(
            (
                text
                for text in detail_texts
                if text.lower().startswith("rating")
            ),
            None,
        )

        colors = next(
            (
                text
                for text in detail_texts
                if "color" in text.lower()
            ),
            None,
        )

        size = next(
            (
                text
                for text in detail_texts
                if text.lower().startswith("size")
            ),
            None,
        )

        gender = next(
            (
                text
                for text in detail_texts
                if text.lower().startswith("gender")
            ),
            None,
        )

        return {
            "Title": title,
            "Price": price,
            "Rating": rating,
            "Colors": colors,
            "Size": size,
            "Gender": gender,
            "Timestamp": extracted_at,
        }

    except Exception as error:
        raise RuntimeError(
            f"Failed to extract product card: {error}"
        ) from error


def parse_products(
    html_content: bytes | str,
    extracted_at: str | None = None,
) -> list[dict[str, Any]]:
    """Parse all product cards from one HTML page.

    Args:
        html_content: Raw HTML content.
        extracted_at: Optional extraction timestamp.

    Returns:
        List of raw product dictionaries.

    Raises:
        ValueError: If HTML content is empty.
        RuntimeError: If parsing fails.
    """
    if html_content is None or not html_content:
        raise ValueError("HTML content must not be empty.")

    timestamp = extracted_at or datetime.now().isoformat()

    try:
        soup = BeautifulSoup(html_content, "html.parser")
        product_cards = soup.select(".collection-card")

        products = [
            extract_product_card(card, timestamp)
            for card in product_cards
        ]

        return products

    except Exception as error:
        raise RuntimeError(
            f"Failed to parse product HTML: {error}"
        ) from error


def scrape_page(
    page_number: int,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Scrape products from one Fashion Studio page.

    Args:
        page_number: Page number starting from 1.
        session: Optional shared HTTP session.

    Returns:
        List of raw product dictionaries from the page.

    Raises:
        RuntimeError: If fetching or parsing the page fails.
    """
    url = build_page_url(page_number)

    try:
        html_content = fetching_content(
            url=url,
            session=session,
        )

        extracted_at = datetime.now().isoformat()

        return parse_products(
            html_content=html_content,
            extracted_at=extracted_at,
        )

    except Exception as error:
        raise RuntimeError(
            f"Failed to scrape page {page_number}: {error}"
        ) from error


def scrape_main(
    start_page: int = 1,
    end_page: int = TOTAL_PAGES,
    session: requests.Session | None = None,
) -> list[dict[str, Any]]:
    """Scrape product data from multiple Fashion Studio pages.

    Args:
        start_page: First page to scrape.
        end_page: Last page to scrape.
        session: Optional shared HTTP session.

    Returns:
        Combined list of raw product dictionaries.

    Raises:
        ValueError: If the page range is invalid.
        RuntimeError: If the scraping process fails.
    """
    if start_page < 1:
        raise ValueError("Start page must be at least 1.")

    if end_page < start_page:
        raise ValueError(
            "End page must be greater than or equal to start page."
        )

    products: list[dict[str, Any]] = []
    active_session = session or create_session()

    try:
        for page_number in range(start_page, end_page + 1):
            page_products = scrape_page(
                page_number=page_number,
                session=active_session,
            )

            products.extend(page_products)

            print(
                f"Page {page_number} extracted: "
                f"{len(page_products)} products"
            )

        return products

    except Exception as error:
        raise RuntimeError(
            f"An error occurred during scraping: {error}"
        ) from error

    finally:
        if session is None:
            active_session.close()


def extract_products() -> list[dict[str, Any]]:
    """Extract all products from Fashion Studio.

    Returns:
        Raw product data from pages 1 through 50.
    """
    return scrape_main(
        start_page=1,
        end_page=TOTAL_PAGES,
    )