"""
SICP Content Scraper for MIT OCW

A polite web scraper that fetches Structure and Interpretation of Computer Programs
content from MIT OpenCourseWare for research and educational purposes.

Features:
- Rate limiting with configurable delays
- Retry logic with exponential backoff
- Local caching of downloaded content
- Clean text extraction from HTML
- Streaming and batch interfaces
- Structured data output by chapter/section
"""

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator, Optional, List, Dict, Any
from urllib.parse import urljoin, urlparse

try:
    import requests
    from bs4 import BeautifulSoup, NavigableString
except ImportError as e:
    raise ImportError(
        "Required packages not installed. Please run:\n"
        "  pip install requests beautifulsoup4 lxml"
    ) from e

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class SICPSection:
    """Represents a section of SICP content."""
    chapter: int
    section: str
    title: str
    content: str
    url: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return asdict(self)


class RateLimiter:
    """Implements rate limiting for polite scraping."""

    def __init__(self, min_delay: float = 1.0, max_delay: float = 3.0):
        """
        Initialize rate limiter.

        Args:
            min_delay: Minimum delay between requests in seconds
            max_delay: Maximum delay between requests in seconds
        """
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.last_request_time: Optional[float] = None

    def wait(self) -> None:
        """Wait appropriate time before next request."""
        if self.last_request_time is not None:
            elapsed = time.time() - self.last_request_time
            # Use variable delay to appear more human-like
            import random
            delay = random.uniform(self.min_delay, self.max_delay)
            if elapsed < delay:
                sleep_time = delay - elapsed
                logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f}s")
                time.sleep(sleep_time)
        self.last_request_time = time.time()


class ContentCache:
    """Local file-based cache for downloaded content."""

    def __init__(self, cache_dir: str = ".sicp_cache"):
        """
        Initialize cache.

        Args:
            cache_dir: Directory to store cached content
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / "metadata.json"
        self._load_metadata()

    def _load_metadata(self) -> None:
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    self.metadata = json.load(f)
            except (json.JSONDecodeError, IOError):
                self.metadata = {}
        else:
            self.metadata = {}

    def _save_metadata(self) -> None:
        """Save cache metadata to disk."""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, indent=2)

    def _url_to_key(self, url: str) -> str:
        """Convert URL to cache key."""
        return hashlib.md5(url.encode()).hexdigest()

    def get(self, url: str) -> Optional[str]:
        """
        Retrieve cached content for URL.

        Args:
            url: The URL to look up

        Returns:
            Cached content if available, None otherwise
        """
        key = self._url_to_key(url)
        cache_file = self.cache_dir / f"{key}.html"

        if cache_file.exists() and key in self.metadata:
            logger.debug(f"Cache hit for {url}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                return f.read()
        return None

    def set(self, url: str, content: str) -> None:
        """
        Store content in cache.

        Args:
            url: The URL as key
            content: The content to cache
        """
        key = self._url_to_key(url)
        cache_file = self.cache_dir / f"{key}.html"

        with open(cache_file, 'w', encoding='utf-8') as f:
            f.write(content)

        self.metadata[key] = {
            'url': url,
            'timestamp': time.time(),
            'size': len(content)
        }
        self._save_metadata()
        logger.debug(f"Cached content for {url}")

    def clear(self) -> None:
        """Clear all cached content."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = {}
        self._save_metadata()
        logger.info("Cache cleared")


class SICPScraper:
    """
    Scraper for SICP content from MIT OCW.

    Implements polite scraping practices including rate limiting,
    retry logic, and local caching.
    """

    # Known SICP URLs - primary and fallbacks
    BASE_URLS = [
        "https://mitp-content-server.mit.edu/books/content/sectbyfn/books_pres_0/6515/sicp.zip/full-text/book/",
        "https://mitpress.mit.edu/sites/default/files/sicp/full-text/book/",
        "https://web.mit.edu/6.001/6.037/sicp/full-text/book/",
    ]

    # Chapter structure for SICP with accurate page numbers
    # Format: chapter_intro_page, [(section_num, title, page_num), ...]
    CHAPTER_STRUCTURE = {
        1: {
            'title': 'Building Abstractions with Procedures',
            'intro_page': 9,
            'sections': [
                ('1.1', 'The Elements of Programming', 10),
                ('1.2', 'Procedures and the Processes They Generate', 11),
                ('1.3', 'Formulating Abstractions with Higher-Order Procedures', 12),
            ]
        },
        2: {
            'title': 'Building Abstractions with Data',
            'intro_page': 13,
            'sections': [
                ('2.1', 'Introduction to Data Abstraction', 14),
                ('2.2', 'Hierarchical Data and the Closure Property', 15),
                ('2.3', 'Symbolic Data', 16),
                ('2.4', 'Multiple Representations for Abstract Data', 17),
                ('2.5', 'Systems with Generic Operations', 18),
            ]
        },
        3: {
            'title': 'Modularity, Objects, and State',
            'intro_page': 19,
            'sections': [
                ('3.1', 'Assignment and Local State', 20),
                ('3.2', 'The Environment Model of Evaluation', 21),
                ('3.3', 'Modeling with Mutable Data', 22),
                ('3.4', 'Concurrency: Time Is of the Essence', 23),
                ('3.5', 'Streams', 24),
            ]
        },
        4: {
            'title': 'Metalinguistic Abstraction',
            'intro_page': 25,
            'sections': [
                ('4.1', 'The Metacircular Evaluator', 26),
                ('4.2', 'Variations on a Scheme -- Lazy Evaluation', 27),
                ('4.3', 'Variations on a Scheme -- Nondeterministic Computing', 28),
                ('4.4', 'Logic Programming', 29),
            ]
        },
        5: {
            'title': 'Computing with Register Machines',
            'intro_page': 30,
            'sections': [
                ('5.1', 'Designing Register Machines', 31),
                ('5.2', 'A Register-Machine Simulator', 32),
                ('5.3', 'Storage Allocation and Garbage Collection', 33),
                ('5.4', 'The Explicit-Control Evaluator', 34),
                ('5.5', 'Compilation', 35),
            ]
        },
    }

    # URL patterns for different SICP mirrors
    URL_PATTERNS = {
        'mitpress': 'book-Z-H-{page}.html',
        'index': 'book.html',
    }

    # Page number mapping based on actual SICP structure
    CHAPTER_PAGES = {
        1: [9, 10, 11, 12],   # Chapter 1: intro + 3 sections
        2: [13, 14, 15, 16, 17, 18],  # Chapter 2: intro + 5 sections
        3: [19, 20, 21, 22, 23, 24],  # Chapter 3: intro + 5 sections
        4: [25, 26, 27, 28, 29],  # Chapter 4: intro + 4 sections
        5: [30, 31, 32, 33, 34, 35],  # Chapter 5: intro + 5 sections
    }

    def __init__(
        self,
        base_url: Optional[str] = None,
        cache_dir: str = ".sicp_cache",
        min_delay: float = 1.0,
        max_delay: float = 3.0,
        max_retries: int = 3,
        timeout: int = 30,
        user_agent: Optional[str] = None
    ):
        """
        Initialize the SICP scraper.

        Args:
            base_url: Base URL for SICP content (uses defaults if None)
            cache_dir: Directory for caching downloaded content
            min_delay: Minimum delay between requests in seconds
            max_delay: Maximum delay between requests in seconds
            max_retries: Maximum number of retry attempts
            timeout: Request timeout in seconds
            user_agent: Custom user agent string
        """
        self.base_url = base_url
        self.cache = ContentCache(cache_dir)
        self.rate_limiter = RateLimiter(min_delay, max_delay)
        self.max_retries = max_retries
        self.timeout = timeout

        # Set up session with appropriate headers
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': user_agent or (
                'Mozilla/5.0 (compatible; SICPResearchBot/1.0; '
                '+educational-research; respects-robots.txt)'
            ),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
        })

        self._working_base_url: Optional[str] = None

    def _fetch_with_retry(self, url: str) -> Optional[str]:
        """
        Fetch URL content with retry logic and exponential backoff.

        Args:
            url: URL to fetch

        Returns:
            Response content if successful, None otherwise
        """
        # Check cache first
        cached = self.cache.get(url)
        if cached is not None:
            return cached

        last_exception = None

        for attempt in range(self.max_retries):
            try:
                # Apply rate limiting
                self.rate_limiter.wait()

                logger.info(f"Fetching {url} (attempt {attempt + 1}/{self.max_retries})")
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()

                content = response.text

                # Cache successful response
                self.cache.set(url, content)

                return content

            except requests.exceptions.HTTPError as e:
                last_exception = e
                if response.status_code == 404:
                    logger.warning(f"Page not found: {url}")
                    return None
                elif response.status_code == 429:
                    # Rate limited - wait longer
                    wait_time = 2 ** (attempt + 2)
                    logger.warning(f"Rate limited, waiting {wait_time}s")
                    time.sleep(wait_time)
                else:
                    logger.error(f"HTTP error {response.status_code}: {e}")

            except requests.exceptions.ConnectionError as e:
                last_exception = e
                logger.warning(f"Connection error: {e}")

            except requests.exceptions.Timeout as e:
                last_exception = e
                logger.warning(f"Timeout error: {e}")

            except requests.exceptions.RequestException as e:
                last_exception = e
                logger.error(f"Request error: {e}")

            # Exponential backoff
            if attempt < self.max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time}s...")
                time.sleep(wait_time)

        logger.error(f"Failed to fetch {url} after {self.max_retries} attempts")
        if last_exception:
            logger.error(f"Last error: {last_exception}")
        return None

    def _find_working_base_url(self) -> Optional[str]:
        """Find a working base URL from the known list."""
        if self._working_base_url:
            return self._working_base_url

        if self.base_url:
            # Test provided URL
            test_url = urljoin(self.base_url, 'book.html')
            content = self._fetch_with_retry(test_url)
            if content and ('SICP' in content.upper() or 'Structure and Interpretation' in content):
                self._working_base_url = self.base_url
                return self.base_url

        # Try known URLs
        for base in self.BASE_URLS:
            try:
                test_url = urljoin(base, 'book.html')
                content = self._fetch_with_retry(test_url)
                if content and ('SICP' in content.upper() or 'Structure and Interpretation' in content):
                    self._working_base_url = base
                    logger.info(f"Using base URL: {base}")
                    return base
            except Exception as e:
                logger.debug(f"Base URL {base} failed: {e}")
                continue

        return None

    def _extract_clean_text(self, html_content: str) -> str:
        """
        Extract clean text from HTML, removing navigation and boilerplate.

        Args:
            html_content: Raw HTML content

        Returns:
            Cleaned text content
        """
        soup = BeautifulSoup(html_content, 'lxml')

        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'nav', 'header',
                                       'footer', 'aside', 'noscript']):
            element.decompose()

        # Remove navigation links (common patterns)
        for element in soup.find_all(['a', 'div', 'span'],
                                      class_=re.compile(r'nav|menu|sidebar|footer|header', re.I)):
            element.decompose()

        # Remove elements with navigation-like text
        nav_patterns = [
            re.compile(r'^\s*(previous|next|contents|index|back|forward)\s*$', re.I),
            re.compile(r'^\s*\[.*\]\s*$'),  # [Previous] [Next] style links
        ]
        for a in soup.find_all('a'):
            text = a.get_text(strip=True)
            for pattern in nav_patterns:
                if pattern.match(text):
                    a.decompose()
                    break

        # Find main content area
        main_content = None
        for selector in ['main', 'article', '.content', '#content',
                         '.chapter', '.section', 'body']:
            if selector.startswith('.'):
                main_content = soup.find(class_=selector[1:])
            elif selector.startswith('#'):
                main_content = soup.find(id=selector[1:])
            else:
                main_content = soup.find(selector)
            if main_content:
                break

        if not main_content:
            main_content = soup.body or soup

        # Extract and clean text
        text_parts = []

        for element in main_content.descendants:
            if isinstance(element, NavigableString):
                text = str(element).strip()
                if text:
                    text_parts.append(text)
            elif element.name in ['p', 'div', 'br', 'h1', 'h2', 'h3',
                                   'h4', 'h5', 'h6', 'li']:
                text_parts.append('\n')

        # Join and clean up
        text = ' '.join(text_parts)

        # Clean up whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n ', '\n', text)
        text = text.strip()

        return text

    def _extract_title(self, html_content: str) -> str:
        """Extract the title from HTML content."""
        soup = BeautifulSoup(html_content, 'lxml')

        # Try various title sources
        title_element = (
            soup.find('h1') or
            soup.find('h2') or
            soup.find('title') or
            soup.find(class_=re.compile(r'title', re.I))
        )

        if title_element:
            return title_element.get_text(strip=True)
        return "Unknown Title"

    def _parse_chapter_section(self, url: str, title: str) -> tuple:
        """
        Parse chapter and section from URL or title.

        Returns:
            Tuple of (chapter_num, section_str)
        """
        # Try to extract from URL
        match = re.search(r'(\d+)[._-](\d+)', url)
        if match:
            return int(match.group(1)), f"{match.group(1)}.{match.group(2)}"

        # Try to extract from title
        match = re.search(r'(\d+)[._](\d+)', title)
        if match:
            return int(match.group(1)), f"{match.group(1)}.{match.group(2)}"

        # Try chapter only
        match = re.search(r'chapter\s*(\d+)', title, re.I)
        if match:
            return int(match.group(1)), f"{match.group(1)}"

        return 0, "0"

    def fetch_page(self, url: str) -> Optional[SICPSection]:
        """
        Fetch and parse a single SICP page.

        Args:
            url: Full URL to the page

        Returns:
            SICPSection object if successful, None otherwise
        """
        content = self._fetch_with_retry(url)
        if not content:
            return None

        title = self._extract_title(content)
        clean_text = self._extract_clean_text(content)
        chapter, section = self._parse_chapter_section(url, title)

        return SICPSection(
            chapter=chapter,
            section=section,
            title=title,
            content=clean_text,
            url=url
        )

    def fetch_chapter(self, chapter_num: int) -> List[SICPSection]:
        """
        Fetch all sections of a chapter (batch interface).

        Args:
            chapter_num: Chapter number (1-5)

        Returns:
            List of SICPSection objects for the chapter
        """
        if chapter_num not in self.CHAPTER_STRUCTURE:
            logger.error(f"Invalid chapter number: {chapter_num}")
            return []

        sections = []
        base_url = self._find_working_base_url()
        chapter_info = self.CHAPTER_STRUCTURE[chapter_num]

        if not base_url:
            logger.error("Could not find working SICP URL")
            # Return chapter info from our structure
            return [SICPSection(
                chapter=chapter_num,
                section=str(chapter_num),
                title=chapter_info['title'],
                content=f"Chapter {chapter_num}: {chapter_info['title']}\n\n"
                        f"Sections:\n" +
                        '\n'.join(f"  {s[0]}: {s[1]}" for s in chapter_info['sections']),
                url=""
            )]

        # Fetch chapter introduction page
        intro_page = chapter_info.get('intro_page')
        if intro_page:
            url = urljoin(base_url, f"book-Z-H-{intro_page}.html")
            section = self.fetch_page(url)
            if section:
                section.chapter = chapter_num
                section.section = str(chapter_num)
                section.title = chapter_info['title']
                sections.append(section)

        # Fetch each section
        for section_num, section_title, page_num in chapter_info['sections']:
            url = urljoin(base_url, f"book-Z-H-{page_num}.html")
            section = self.fetch_page(url)
            if section:
                section.chapter = chapter_num
                section.section = section_num
                section.title = section_title
                sections.append(section)

        return sections

    def stream_chapter(self, chapter_num: int) -> Iterator[SICPSection]:
        """
        Stream sections of a chapter one at a time (streaming interface).

        Args:
            chapter_num: Chapter number (1-5)

        Yields:
            SICPSection objects as they are fetched
        """
        if chapter_num not in self.CHAPTER_STRUCTURE:
            logger.error(f"Invalid chapter number: {chapter_num}")
            return

        base_url = self._find_working_base_url()
        chapter_info = self.CHAPTER_STRUCTURE[chapter_num]

        if not base_url:
            logger.warning("Could not find working SICP URL, yielding structure info")
            yield SICPSection(
                chapter=chapter_num,
                section=str(chapter_num),
                title=chapter_info['title'],
                content=f"Chapter {chapter_num}: {chapter_info['title']}\n\n"
                        f"Sections:\n" +
                        '\n'.join(f"  {s[0]}: {s[1]}" for s in chapter_info['sections']),
                url=""
            )
            return

        # Stream chapter introduction
        intro_page = chapter_info.get('intro_page')
        if intro_page:
            url = urljoin(base_url, f"book-Z-H-{intro_page}.html")
            section = self.fetch_page(url)
            if section:
                section.chapter = chapter_num
                section.section = str(chapter_num)
                section.title = chapter_info['title']
                yield section

        # Stream each section
        for section_num, section_title, page_num in chapter_info['sections']:
            url = urljoin(base_url, f"book-Z-H-{page_num}.html")
            section = self.fetch_page(url)
            if section:
                section.chapter = chapter_num
                section.section = section_num
                section.title = section_title
                yield section

    def fetch_all(self) -> List[SICPSection]:
        """
        Fetch all chapters (batch interface).

        Returns:
            List of all SICPSection objects
        """
        all_sections = []
        for chapter_num in range(1, 6):
            logger.info(f"Fetching Chapter {chapter_num}...")
            sections = self.fetch_chapter(chapter_num)
            all_sections.extend(sections)
        return all_sections

    def stream_all(self) -> Iterator[SICPSection]:
        """
        Stream all chapters (streaming interface).

        Yields:
            SICPSection objects as they are fetched
        """
        for chapter_num in range(1, 6):
            logger.info(f"Streaming Chapter {chapter_num}...")
            yield from self.stream_chapter(chapter_num)

    def get_table_of_contents(self) -> Dict[int, Dict[str, Any]]:
        """
        Get the table of contents structure.

        Returns:
            Dictionary of chapter structures with simplified section info
        """
        toc = {}
        for ch_num, ch_info in self.CHAPTER_STRUCTURE.items():
            toc[ch_num] = {
                'title': ch_info['title'],
                'sections': [(s[0], s[1]) for s in ch_info['sections']]
            }
        return toc

    def search_content(self, query: str, chapter: Optional[int] = None) -> List[SICPSection]:
        """
        Search cached content for a query string.

        Args:
            query: Search query
            chapter: Optional chapter to limit search

        Returns:
            List of matching sections
        """
        results = []
        query_lower = query.lower()

        # Search through cached files
        for cache_file in self.cache.cache_dir.glob("*.html"):
            if cache_file.name == "metadata.json":
                continue

            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    content = f.read()

                if query_lower in content.lower():
                    # Parse the content
                    clean_text = self._extract_clean_text(content)
                    title = self._extract_title(content)
                    ch, sec = self._parse_chapter_section(str(cache_file), title)

                    if chapter is None or ch == chapter:
                        results.append(SICPSection(
                            chapter=ch,
                            section=sec,
                            title=title,
                            content=clean_text,
                            url=""
                        ))
            except Exception as e:
                logger.debug(f"Error searching {cache_file}: {e}")

        return results

    def export_to_json(self, sections: List[SICPSection], filepath: str) -> None:
        """
        Export sections to JSON file.

        Args:
            sections: List of SICPSection objects
            filepath: Output file path
        """
        data = [s.to_dict() for s in sections]
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Exported {len(sections)} sections to {filepath}")

    def close(self) -> None:
        """Close the session and clean up resources."""
        self.session.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def demonstrate_chapter_1():
    """
    Demonstrate fetching Chapter 1 of SICP.

    This function shows both streaming and batch interfaces.
    """
    print("=" * 70)
    print("SICP Scraper Demonstration - Chapter 1")
    print("=" * 70)
    print()

    # Create scraper with polite settings
    scraper = SICPScraper(
        min_delay=1.5,  # At least 1.5 seconds between requests
        max_delay=3.0,  # Up to 3 seconds
        max_retries=3,
        cache_dir=".sicp_cache"
    )

    try:
        # Show table of contents first
        print("Table of Contents for Chapter 1:")
        print("-" * 40)
        toc = scraper.get_table_of_contents()
        chapter_1 = toc[1]
        print(f"Chapter 1: {chapter_1['title']}")
        for section_num, section_title in chapter_1['sections']:
            print(f"  {section_num}: {section_title}")
        print()

        # Demonstrate streaming interface
        print("Demonstrating Streaming Interface:")
        print("-" * 40)
        section_count = 0

        for section in scraper.stream_chapter(1):
            section_count += 1
            print(f"\nSection: {section.section}")
            print(f"Title: {section.title}")
            print(f"Content preview: {section.content[:200]}...")
            print()

            # Only fetch a few sections for demonstration
            if section_count >= 2:
                print("(Stopping after 2 sections for demonstration)")
                break

        # Demonstrate batch interface
        print("\nDemonstrating Batch Interface:")
        print("-" * 40)

        # Fetch will use cache for already-fetched pages
        sections = scraper.fetch_chapter(1)
        print(f"Fetched {len(sections)} sections from Chapter 1")

        # Show summary
        for section in sections[:3]:  # Show first 3
            print(f"  - {section.section}: {section.title}")

        if len(sections) > 3:
            print(f"  ... and {len(sections) - 3} more sections")

        # Export demonstration
        if sections:
            export_path = ".sicp_cache/chapter1_demo.json"
            scraper.export_to_json(sections[:2], export_path)
            print(f"\nExported sample to: {export_path}")

        print("\n" + "=" * 70)
        print("Demonstration complete!")
        print("Note: Content is cached locally for subsequent runs.")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
    except Exception as e:
        logger.error(f"Error during demonstration: {e}")
        raise
    finally:
        scraper.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SICP Content Scraper for research/educational purposes"
    )
    parser.add_argument(
        "--chapter", "-c",
        type=int,
        choices=[1, 2, 3, 4, 5],
        default=1,
        help="Chapter to fetch (default: 1)"
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Fetch all chapters"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output JSON file path"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=".sicp_cache",
        help="Cache directory (default: .sicp_cache)"
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear the cache before fetching"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demonstration mode"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.demo:
        demonstrate_chapter_1()
    else:
        with SICPScraper(cache_dir=args.cache_dir) as scraper:
            if args.clear_cache:
                scraper.cache.clear()

            if args.all:
                print("Fetching all chapters...")
                sections = scraper.fetch_all()
            else:
                print(f"Fetching Chapter {args.chapter}...")
                sections = scraper.fetch_chapter(args.chapter)

            print(f"Retrieved {len(sections)} sections")

            if args.output:
                scraper.export_to_json(sections, args.output)
            else:
                # Print summary
                for section in sections:
                    print(f"\n[{section.chapter}.{section.section}] {section.title}")
                    if section.content:
                        preview = section.content[:150].replace('\n', ' ')
                        print(f"  {preview}...")
