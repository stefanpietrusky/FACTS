"""
title: FACTS V2.5 [FILTERING AND ANALYSIS OF CONTENT IN TEXTUAL SOURCES]
author: xxx
version: 1.0
"""

import os
import io
import logging
import requests
import re
import threading
import time
import uuid
import json
import xml.etree.ElementTree as ET
import subprocess
import unicodedata
from datetime import datetime

import fitz

from flask import Flask, request, jsonify, Response

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from selenium.common.exceptions import TimeoutException, WebDriverException
from bs4 import BeautifulSoup

from difflib import SequenceMatcher

import pandas as pd
import nltk
from nltk.corpus import stopwords
from gensim.models import LdaModel, CoherenceModel
from gensim.corpora import Dictionary
from wordcloud import WordCloud
from urllib.parse import quote, urljoin
import pyLDAvis
import pyLDAvis.gensim_models
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import matplotlib
import matplotlib.pyplot as plt
import math

matplotlib.use("Agg")

# ---------------------------
# Logging & Configuration
# ---------------------------
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = Flask(__name__)
start_time = datetime.now()

# Globales Dictionary
download_progress = {}
analysis_progress = {}
sediment_progress = {}

# Base folder for downloads (data acquisition)
DOWNLOAD_ROOT = os.path.join(os.getcwd(), "downloads")
if not os.path.exists(DOWNLOAD_ROOT):
    os.makedirs(DOWNLOAD_ROOT)

# Base folder for analysis results (data analysis)
DATA_ROOT = os.path.join(os.getcwd(), "data")
if not os.path.exists(DATA_ROOT):
    os.makedirs(DATA_ROOT)

# Base folder for analysis results (data analysis)
ANALYSIS_ROOT = os.path.join(os.getcwd(), "analysis")
if not os.path.exists(ANALYSIS_ROOT):
    os.makedirs(ANALYSIS_ROOT)

# ---------------------------
# Process exclusivity
# ---------------------------
current_process = {"download": False, "analysis": False, "sediment": False}
process_lock = threading.Lock()

# ---------------------------
# Constants
# ---------------------------
GECKO_DRIVER_PATH = (
    r"\geckodriver.exe"
)
PDF_BASE_URL = "https://files.eric.ed.gov/fulltext/"

# ---------------------------
# Functions: Data acquisition
# ---------------------------

def is_pdf_file(filepath):
    try:
        with open(filepath, "rb") as f:
            return f.read(4) == b"%PDF"
    except Exception:
        return False

def resolve_eric_pdf_url(detail_url, session=None):
    session = session or requests.Session()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            " AppleWebKit/537.36 (KHTML, like Gecko)"
            " Chrome/125.0.0.0 Safari/537.36"
        ),
        "Referer": "https://eric.ed.gov/",
        "Accept": "application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive",
        "Accept-Language": "en-US,en;q=0.9,de-DE;q=0.8,de;q=0.7",
    }
    try:
        resp = session.get(detail_url, timeout=30, headers=headers)
        resp.raise_for_status()
    except Exception as e:
        logging.warning(f"Detail page not available: {e}")
        return None

    html = resp.text
    soup = BeautifulSoup(html, "html.parser")

    link = soup.find("a", id="downloadFullText") or soup.find(
        "a", string=re.compile(r"Download Full Text", re.I)
    )
    if link and link.get("href"):
        return urljoin(detail_url, link["href"])

    link2 = soup.find("a", string=re.compile(r"Full\s*Text\s*PDF", re.I))
    if link2 and link2.get("href"):
        return urljoin(detail_url, link2["href"])

    iframe = soup.find("iframe", src=re.compile(r"\.pdf($|\?)", re.I))
    if iframe and iframe.get("src"):
        return urljoin(detail_url, iframe["src"])

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.lower().endswith(".pdf") and "/fulltext/" in href.lower():
            return urljoin(detail_url, href)

    m = re.search(r'"pdfUrl"\s*:\s*"([^"]+\.pdf[^"]*)"', html)
    if m:
        return urljoin(detail_url, m.group(1).replace(r"\/", "/"))

    m = re.match(r".*/fulltext/(EJ\d+)\.html", detail_url)
    if m:
        ej_id = m.group(1)
        direct_pdf = f"https://files.eric.ed.gov/fulltext/{ej_id}.pdf"
        logging.info(f"Try Direct PDF for EJ: {direct_pdf}")
        try:
            r = session.head(direct_pdf, timeout=10, headers=headers)
            if r.status_code == 200:
                return direct_pdf
        except Exception:
            pass

    logging.warning(f"NO PDF LINK on detail page: {detail_url}")
    return None

def download_pdf_generic(pdf_url, download_folder, file_name, session=None, progress_data=None):
    if not pdf_url:
        logging.warning("download_pdf_generic: No URL received.")
        return None

    if pdf_url.startswith("http://files.eric.ed.gov/"):
        pdf_url = pdf_url.replace("http://", "https://", 1)

    session = session or requests.Session()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "Referer": "https://eric.ed.gov/",
    }

    pdf_path = os.path.join(download_folder, file_name)
    logging.info(f"Start download: {pdf_url} → {pdf_path}")

    try:
        timeout = (10, 60)
        r = session.get(pdf_url, headers=headers, timeout=timeout, stream=True)
        logging.info(f"Response status code: {r.status_code}")
        r.raise_for_status()
    except Exception as e:
        logging.error(f"Error retrieving PDF: {e}")
        return None

    content_length = r.headers.get("Content-Length")
    if content_length:
        logging.info(f"Content-Length (Header): {content_length} bytes")

    try:
        total_written = 0
        with open(pdf_path, "wb") as f:
            for i, chunk in enumerate(r.iter_content(chunk_size=16_384), start=1):
                if progress_data and progress_data.get("abort"):
                    logging.info("Interruption detected during download, abort.")
                    try:
                        f.close()
                        os.remove(pdf_path)
                    except Exception:
                        pass
                    return None
                if not chunk:
                    continue
                f.write(chunk)
                total_written += len(chunk)
                if i % 10 == 0:
                    logging.info(
                        f"  ... {total_written} bytes written (after {i} chunks)"
                    )
            f.flush()
            os.fsync(f.fileno())

        logging.info(
            f"Finish writing: Total {total_written} bytes in '{pdf_path}'"
        )
        if total_written > 0 and os.path.exists(pdf_path):
            if is_pdf_file(pdf_path):
                logging.info("The file is a PDF and has been saved correctly.")
                return pdf_url
            else:
                bad_path = pdf_path + ".html"
                os.rename(pdf_path, bad_path)
                with open(bad_path, "r", encoding="utf-8", errors="ignore") as f:
                    head = f.read(500)
                    logging.error(f"No PDF received. The first 500 characters:\n{head}")
                return None
        else:
            logging.error("File exists but is empty!")
            return None

    except Exception as e:
        logging.error(f"Error writing file:{e}")
        return None

def generate_apa_citation(metadata, database):
    authors = metadata.get("authors", [])
    if isinstance(authors, list):
        author_str = ", ".join(authors) if authors else "Unknown author"
    else:
        author_str = authors or "Unknown author"
    year = metadata.get("year", "o.J.")
    title = metadata.get("title", "No title")
    citation = f"{author_str} ({year}). {title}. {database}. Retrieved from {metadata.get('url', 'n/a')}"
    return citation

def download_pdf_eric(
    paper_id, download_folder, session=None, file_name=None, max_retries=5
):
    pdf_url = f"https://files.eric.ed.gov/fulltext/{paper_id}.pdf"
    pdf_filename = (
        os.path.join(download_folder, file_name)
        if file_name
        else os.path.join(download_folder, f"{paper_id}.pdf")
    )

    session = session or requests.Session()

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            " AppleWebKit/537.36 (KHTML, like Gecko)"
            " Chrome/125.0.0.0 Safari/537.36"
        ),
        "Referer": "https://eric.ed.gov/",
        "Accept": "application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive",
        "Accept-Language": "en-US,en;q=0.9,de-DE;q=0.8,de;q=0.7",
    }

    for attempt in range(max_retries):
        try:
            logging.info(f"Attempt {attempt+1}: Download PDF: {pdf_url}")
            response = session.get(pdf_url, timeout=30, headers=headers)
            response.raise_for_status()

            if response.headers.get("Content-Type", "").lower() != "application/pdf":
                logging.warning(
                    f"No PDF response ({response.headers.get('Content-Type')}), new attempt in 3 seconds..."
                )
                time.sleep(3)
                continue

            with open(pdf_filename, "wb") as f:
                f.write(response.content)

            if is_pdf_file(pdf_filename):
                logging.info(f"PDF successfully saved: {pdf_filename}")
                return pdf_url
            else:
                logging.warning(
                    "File is not a valid PDF, new attempt in 3 seconds..."
                )
                os.rename(pdf_filename, pdf_filename + ".html")
                time.sleep(3)
                continue

        except requests.exceptions.RequestException as e:
            logging.error(f"Download error: {e}, new attempt in 3 seconds...")
            time.sleep(3)

    logging.error(
        f"All {max_retries} Attempts failed, PDF could not be downloaded."
    )
    return None

def extract_paper_data_eric(paper_div, base_url):
    raw_id = paper_div.get("id", "")
    paper_id = re.sub(r"^r_", "", raw_id)
    if not paper_id:
        logging.warning("No valid paper ID found, skip.")
        return None

    title_elem = paper_div.find("div", class_="r_t").find("a")
    title = title_elem.text.strip() if title_elem else "Unknown title"
    href = title_elem["href"] if title_elem and title_elem.has_attr("href") else None

    if href and href.startswith("/"):
        paper_url = urljoin(base_url, href)
    else:
        paper_url = f"{base_url}?id={paper_id}"

    metadata_elem = paper_div.find("div", class_="r_a")
    metadata_text = metadata_elem.text.strip() if metadata_elem else ""
    if "," in metadata_text:
        author_journal, pub_year = metadata_text.rsplit(",", 1)
        pub_year = pub_year.strip()
    else:
        author_journal, pub_year = metadata_text, "o.J."

    snippet_elem = paper_div.find("div", class_="r_s")
    snippet = snippet_elem.text.strip() if snippet_elem else ""

    return {
        "paper_id": paper_id,
        "title": title,
        "paper_url": paper_url,
        "author_journal": author_journal,
        "pub_year": pub_year,
        "snippet": snippet,
    }

def download_eric_selenium(
    query, target_year, num_papers, download_folder, progress_data
):
    global current_process
    options = Options()
    options.headless = True
    options.add_argument("--headless")
    options.set_preference("permissions.default.image", 2)
    service = Service(executable_path=GECKO_DRIVER_PATH)
    driver = webdriver.Firefox(service=service, options=options)

    driver.set_page_load_timeout(30)
    try:
        start_url = f"https://eric.ed.gov/?q={quote(query)}&ft=on"
        driver.get(start_url)
    except (TimeoutException, WebDriverException) as e:
        logging.error("Eric homepage could not be loaded: %s", e)
        progress_data.update(status="error", error="Eric homepage not reachable")
        driver.quit()
        return

    try:
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "r_i"))
        )
    except TimeoutException:
        logging.error("Search results did not load (timeout).")
        progress_data.update(status="error", error="Results unavailable")
        driver.quit()
        return

    os.makedirs(download_folder, exist_ok=True)
    references_file = os.path.join(download_folder, "sources.txt")
    downloaded_papers = 0
    eric_index = 1
    session = requests.Session()

    with open(references_file, "a", encoding="utf-8") as ref_file:
        while downloaded_papers < num_papers:
            if progress_data.get("abort"):
                progress_data["status"] = "aborted"
                logging.info("ERIC download aborted by user.")
                driver.quit()
                with process_lock:
                    current_process = None
                return

            soup = BeautifulSoup(driver.page_source, "html.parser")
            paper_divs = soup.find_all("div", class_="r_i")
            if not paper_divs:
                logging.info("No more hits on this page.")
                break

            for paper in paper_divs:
                if downloaded_papers >= num_papers:
                    break

                data = extract_paper_data_eric(paper, "https://eric.ed.gov/")
                if not data:
                    logging.debug("No valid metadata for a paper, skipping.")
                    continue

                if data["pub_year"] != str(target_year):
                    logging.info(
                        f"Skipping paper '{data.get('title', 'Unknown')}' due to year mismatch: found {data.get('pub_year')} vs requested {target_year}"
                    )
                    continue

                pdf_btn = paper.select_one("a[href*='/fulltext/'][href$='.pdf']")
                if pdf_btn:
                    href = pdf_btn["href"]
                    pdf_url = (
                        href
                        if href.lower().startswith("http")
                        else urljoin("https://files.eric.ed.gov", href)
                    )
                else:
                    pdf_url = resolve_eric_pdf_url(data["paper_url"], session)

                if pdf_url and pdf_url.startswith("http://files.eric.ed.gov/"):
                    pdf_url = pdf_url.replace("http://", "https://", 1)

                if not pdf_url:
                    logging.info(f"No PDF link found for '{data.get('title', 'Unknown')}', skipping.")
                    continue

                saved = download_pdf_generic(
                    pdf_url,
                    download_folder,
                    file_name=f"eric_{eric_index}.pdf",
                    session=session,
                    progress_data=progress_data,
                )
                if not saved:
                    logging.warning(f"PDF for '{data.get('title', 'Unknown')}' could not be saved.")
                    continue

                citation = generate_apa_citation(
                    {
                        "title": data["title"],
                        "authors": data["author_journal"],
                        "year": data["pub_year"],
                        "url": saved,
                    },
                    "ERIC",
                )
                ref_file.write(f"{eric_index}. {citation}\n\n")

                downloaded_papers += 1
                progress_data["completed"] = downloaded_papers
                progress_data["percent"] = int(downloaded_papers / num_papers * 100)
                eric_index += 1

            moved_next = False
            try:
                next_elem = None
                for candidate in driver.find_elements(By.TAG_NAME, "a"):
                    txt = candidate.text.strip()
                    if "next page" in txt.lower() or txt.lower().startswith("next") or ("»" in txt and "next" in txt.lower()):
                        next_elem = candidate
                        break

                if not next_elem:
                    for candidate in driver.find_elements(By.CSS_SELECTOR, "a[href*='&pg='], a[href*='?q='], a[href*='pg=']"):
                        txt = candidate.text.strip().lower()
                        if "next" in txt or "»" in candidate.text:
                            next_elem = candidate
                            break

                if next_elem:
                    logging.info(f"Navigation: clicking next page via link text '{next_elem.text.strip()}'.")
                    try:
                        next_elem.click()
                        moved_next = True
                    except Exception:
                        from selenium.webdriver.common.action_chains import ActionChains

                        ActionChains(driver).move_to_element(next_elem).click(next_elem).perform()
                        moved_next = True
                else:
                    logging.info("No further page found (no next link).")
            except Exception as e:
                logging.warning(f"Error finding/clicking the next page: {e}")

            if not moved_next:
                logging.info("Stopping pagination: no next page reachable.")
                break

            try:
                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "r_i"))
                )
                time.sleep(1)
            except TimeoutException:
                logging.info("Timeout while loading the next results page.")
                break

        logging.info("ERIC scraping completed.")
        if downloaded_papers == 0:
            progress_data["status"] = "error"
            progress_data["error"] = f"No hits found for year {target_year} or no valid PDFs downloaded."
        else:
            if progress_data.get("status") != "error":
                progress_data["status"] = "completed"
    driver.quit()

def download_pedocs(query, year, num_papers, download_folder, progress_data):
    global current_process
    options = Options()
    options.headless = True
    options.add_argument("--headless")
    options.set_preference("permissions.default.image", 2)

    service = Service(executable_path=GECKO_DRIVER_PATH)
    driver = webdriver.Firefox(service=service, options=options)

    driver.set_page_load_timeout(30)

    url = "https://www.pedocs.de"
    try:
        driver.get(url)
    except (TimeoutException, WebDriverException) as e:
        logging.error(
            "The peDOCs homepage could not be loaded "
            "(timeout or server not responding): %s",
            e,
        )
        progress_data["status"] = "error"
        progress_data["error"] = (
            "The peDOCs homepage could not be loaded "
            "(timeout or server unreachable)."
        )
        driver.quit()
        return

    try:
        search_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, "volltextsuche"))
        )
    except TimeoutException:
        logging.error("Search field not found (timeout).")
        progress_data["status"] = "error"
        progress_data["error"] = "Search field not found (timeout)."
        driver.quit()
        return

    search_box.clear()
    search_box.send_keys(query)
    search_box.send_keys(Keys.RETURN)

    try:
        WebDriverWait(driver, 10).until(
            lambda d: d.find_elements(By.XPATH, "//a[contains(@href, 'frontdoor.php')]")
        )
    except TimeoutException:
        logging.error("Results did not load (timeout).")
        progress_data["status"] = "error"
        progress_data["error"] = "Results could not be loaded (timeout)."
        driver.quit()
        return

    os.makedirs(download_folder, exist_ok=True)
    references_file = os.path.join(download_folder, "source_references.txt")

    paper_index = 1
    total_downloaded = 0

    try:
        while total_downloaded < num_papers:
            if progress_data.get("abort"):
                progress_data["status"] = "aborted"
                logging.info("peDOCs download aborted by user.")
                driver.quit()
                with process_lock:
                    current_process = None
                return

            soup = BeautifulSoup(driver.page_source, "html.parser")
            paper_links = [
                a["href"]
                for a in soup.find_all("a", href=True)
                if "frontdoor.php" in a["href"]
            ]
            base_url = "https://www.pedocs.de/"

            for rel_link in paper_links:
                if total_downloaded >= num_papers or progress_data.get("abort"):
                    break

                paper_url = base_url + rel_link
                logging.info(f"Visiting: {paper_url}")
                try:
                    paper_response = requests.get(paper_url, timeout=30)
                    paper_response.raise_for_status()
                except Exception as e:
                    logging.error(f"Error loading article {paper_url}: {e}")
                    continue

                paper_soup = BeautifulSoup(paper_response.content, "html.parser")
                year_elem = paper_soup.find("td", itemprop="datePublished")
                pub_year = year_elem.text.strip() if year_elem else None

                if pub_year != str(year):
                    continue

                title_elem = paper_soup.find("h1") or paper_soup.find("title")
                title = title_elem.text.strip() if title_elem else "No Title"

                pdf_link = paper_soup.find("a", class_="a5-book-list-item-fulltext")
                if not pdf_link:
                    logging.info(f"No PDF link for {paper_url}")
                    continue

                pdf_url = pdf_link["href"]
                if pdf_url.startswith("//"):
                    pdf_url = "https:" + pdf_url

                pdf_filename = os.path.join(
                    download_folder, f"pedocs_{paper_index}.pdf"
                )
                try:
                    r = requests.get(pdf_url, stream=True, timeout=30)
                    r.raise_for_status()
                    with open(pdf_filename, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                except Exception as e:
                    logging.error(f"PDF download error for {pdf_url}: {e}")
                    continue

                ref_row = paper_soup.find("th", scope="row", string="Quellenangabe")
                reference = (
                    ref_row.find_next("td").text.strip()
                    if ref_row
                    else "No source reference"
                )
                with open(references_file, "a", encoding="utf-8") as f:
                    f.write(f"{paper_index}. {reference}\n\n")

                total_downloaded += 1
                progress_data["completed"] = total_downloaded
                progress_data["percent"] = int((total_downloaded / num_papers) * 100)
                paper_index += 1

            if total_downloaded >= num_papers:
                break

            try:
                next_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable(
                        (
                            By.XPATH,
                            "//a[@class='svg-wrapper a5-svg-hover' and (@aria-label='Weiter' or @title='Weiter')]",
                        )
                    )
                )
                next_button.click()
                WebDriverWait(driver, 10).until(
                    lambda d: d.find_elements(
                        By.XPATH, "//a[contains(@href, 'frontdoor.php')]"
                    )
                )
            except (TimeoutException, WebDriverException) as e:
                logging.error("Next page could not be loaded: %s", e)
                progress_data["status"] = "error"
                progress_data["error"] = (
                    "The next results page could not be loaded "
                    "(timeout or server unreachable)."
                )
                break

        if not progress_data.get("error"):
            progress_data["status"] = "completed"

    except Exception as e:
        logging.error(f"Unknown error: {e}")
        progress_data["status"] = "error"
        progress_data["error"] = str(e)

    finally:
        driver.quit()

def download_arxiv(query, year, num_papers, download_folder, progress_data):
    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f'(ti:"{query}" OR abs:"{query}") AND '
        f"submittedDate:[{year}01010000 TO {year}12312359]",
        "start": 0,
        "max_results": num_papers,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
    }

    if progress_data.get("abort"):
        progress_data["status"] = "aborted"
        return

    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        logging.error("The arXiv API could not be reached (timeout).")
        progress_data["status"] = "error"
        progress_data["error"] = "Request to arXiv failed (timeout)."
        return
    except requests.exceptions.RequestException as e:
        logging.error(f"Error during arXiv query: {e}")
        progress_data["status"] = "error"
        progress_data["error"] = f"Request to arXiv failed: {e}"
        return

    try:
        root = ET.fromstring(response.content)
    except ET.ParseError:
        logging.error("Response from arXiv was invalid XML.")
        progress_data["status"] = "error"
        progress_data["error"] = (
            "Invalid response from arXiv (could not be parsed)."
        )
        return

    ns = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", ns)
    total = min(num_papers, len(entries))
    progress_data["total"] = total
    progress_data["completed"] = 0

    citations = []
    paper_index = 1

    for idx, entry in enumerate(entries[:total], start=1):
        if progress_data.get("abort"):
            progress_data["status"] = "aborted"
            logging.info("arXiv download aborted by user.")
            with process_lock:
                current_process = None
            return

        title_elem = entry.find("atom:title", ns)
        title = title_elem.text.strip() if title_elem is not None else "No Title"

        summary_elem = entry.find("atom:summary", ns)
        summary = summary_elem.text.strip() if summary_elem is not None else ""

        if query.lower() not in title.lower() and query.lower() not in summary.lower():
            continue

        published_elem = entry.find("atom:published", ns)
        published = published_elem.text.strip() if published_elem is not None else ""
        published_year = published[:4] if published else ""

        authors = [
            a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)
        ]

        pdf_url = None
        for link in entry.findall("atom:link", ns):
            if link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href")
                break

        if pdf_url:
            pdf_filename = os.path.join(download_folder, f"arxiv_{paper_index}.pdf")
            try:
                r = requests.get(pdf_url, stream=True, timeout=30)
                r.raise_for_status()
                os.makedirs(download_folder, exist_ok=True)
                with open(pdf_filename, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            except requests.exceptions.Timeout:
                logging.error(f"PDF download from {pdf_url} failed (timeout).")
            except requests.exceptions.RequestException as e:
                logging.error(f"PDF download error for {pdf_url}: {e}")

        metadata = {
            "title": title,
            "authors": authors,
            "year": published_year,
            "url": pdf_url or entry.find("atom:id", ns).text,
        }
        citation = generate_apa_citation(metadata, "arxiv")
        citations.append(citation)

        progress_data["completed"] = idx
        progress_data["percent"] = int((idx / total) * 100)
        paper_index += 1

    try:
        os.makedirs(download_folder, exist_ok=True)
        citations_file = os.path.join(download_folder, "source_references.txt")
        with open(citations_file, "a", encoding="utf-8") as f:
            for i, citation in enumerate(citations, start=1):
                f.write(f"{i}. {citation}\n\n")
    except Exception as e:
        logging.error(f"Error writing citation file: {e}")
        progress_data["status"] = "error"
        progress_data["error"] = f"Error saving citations: {e}"
        return

    progress_data["status"] = "completed"

def download_papers_background(download_id, database, query, year, num_papers):
    global current_process
    progress_data = download_progress.get(download_id, {})
    progress_data["status"] = "running"
    progress_data["abort"] = False
    download_progress[download_id] = progress_data

    db_folder = os.path.join(
        DOWNLOAD_ROOT, f"{database}_{re.sub(r'[^A-Za-z0-9]+', '_', query)}_{year}"
    )

    try:
        if database == "pedocs":
            download_pedocs(query, year, num_papers, db_folder, progress_data)
        elif database == "arxiv":
            download_arxiv(query, year, num_papers, db_folder, progress_data)
        elif database == "eric":
            download_eric_selenium(query, year, num_papers, db_folder, progress_data)
        else:
            progress_data["status"] = "error"
            progress_data["error"] = "Unknown database."
    except Exception as e:
        progress_data["status"] = "error"
        progress_data["error"] = str(e)
    finally:
        with process_lock:
            current_process["download"] = False

# ---------------------------
# Functions: Data analysis
# ---------------------------
@app.route("/upload_pdfs", methods=["POST"])
def upload_pdfs():
    if "files" not in request.files:
        return jsonify({"error": "No files found."}), 400
    files = request.files.getlist("files")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    upload_folder = os.path.join(DOWNLOAD_ROOT, f"upload_{timestamp}")
    os.makedirs(upload_folder, exist_ok=True)
    saved_files = []
    for file in files:
        if file.filename.endswith(".pdf"):
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)
            saved_files.append(file.filename)
    return jsonify(
        {
            "message": "Files uploaded",
            "folder": os.path.basename(upload_folder),
            "files": saved_files,
        }
    )

def extract_keywords(question, min_len=4):
    words = re.findall(r"\w+", question.lower(), flags=re.UNICODE)
    sw = set(german_stopwords) if 'german_stopwords' in globals() else set()
    return {w for w in words if w not in sw and len(w) >= min_len}

def extract_text_from_pdf(pdf_path, abort_data=None):
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        logging.error(f"[EXTRACT] Could not open PDF: {pdf_path} – {e}")
        return "", 0

    text = ""
    try:
        page_count = doc.page_count
        for page_num in range(page_count):
            if abort_data and abort_data.get("abort"):
                logging.info("[ABORT] PDF parsing aborted")
                break
            try:
                page = doc.load_page(page_num)
                text += page.get_text()
            except Exception as inner_e:
                logging.warning(
                    f"[EXTRACT] Error reading page {page_num} in {pdf_path}: {inner_e}"
                )
                continue
    finally:
        try:
            doc.close()
        except Exception:
            pass

    return text, page_count

def clean_text(
    text,
    remove_list_numbers: bool = True,
    remove_page_headers: bool = True
):
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00ad", "")
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    text = re.sub(r"(\w)[\--–]\s+(\w)", r"\1\2", text)
    if remove_page_headers:
        text = re.sub(r"(?i)\n\s*Page\s+\d+\s*\n", "\n", text)
        text = re.sub(r"\n\s*\d+\s*\n", "\n", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if remove_list_numbers:
        text = re.sub(r"(?<=\s|^)\d+\.\s+", "", text)

    return text

def query_llm_via_cli(input_text):
    try:
        process = subprocess.Popen(
            ["ollama", "run", "llama3.1p"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="ignore",
            bufsize=1,
        )
        stdout, stderr = process.communicate(input=f"{input_text}\n", timeout=180)
        if process.returncode != 0:
            logging.error(f"LLM error: {stderr.strip()}")
            return ""
        response = re.sub(r"\x1b\[.*?m", "", stdout)
        logging.info(f"Raw LLM response: {response.strip()}")
        return response.strip()
    except subprocess.TimeoutExpired:
        process.kill()
        return "Timeout for the model request"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

def validate_llm_response(response, expected_count):
    response = re.sub(r"(?i)^Here\s+are\s+the\s+answers.*?\n", "", response).strip()

    def parse_block(block_text):
        block_text = block_text.strip()
        a = re.search(r"Answer[:\-]\s*(.+?)(?:\n|$)", block_text, flags=re.IGNORECASE)
        e = re.search(r"Citation[:\-]\s*(.+?)(?:\n|$)", block_text, flags=re.IGNORECASE)
        ans = (a.group(1).strip() if a else "").strip()
        ev = (e.group(1).strip() if e else "").strip()

        if ans and re.fullmatch(r"(?i)no answer!?$", ans):
            return "NO ANSWER!", None
        if not ans:
            lines = [l.strip() for l in block_text.splitlines() if l.strip()]
            if lines:
                ans = lines[0]
            if len(lines) > 1 and not ev:
                ev = lines[1]

        if ans and re.fullmatch(r"(?i)no answer!?$", ans):
            return "NO ANSWER!", None

        if not ans:
            return "NO ANSWER!", None

        if not ev or len(ev.split()) < 2:
            return "NO ANSWER!", None

        return ans, ev

    blocks = []
    numbered = re.split(r"(?m)^\s*\d+\.\s*", response)
    if len(numbered) > 1:
        for b in numbered[1:]:
            if b.strip():
                blocks.append(b.strip())
    else:
        blocks = [b.strip() for b in re.split(r"\n\s*\n", response) if b.strip()]

    if len(blocks) < expected_count:
        blocks += [""] * (expected_count - len(blocks))
    else:
        blocks = blocks[:expected_count]

    out_lines = []
    for i, b in enumerate(blocks, start=1):
        ans_ev = parse_block(b)
        if isinstance(ans_ev, tuple):
            ans, ev = ans_ev
        else:
            ans, ev = ans_ev, None

        if ans == "NO ANSWER!":
            out_lines.append(f"{i}. NO ANSWER!")
        else:
            out_lines.append(f"{i}. Answer: {ans}\n   Citation: {ev}")
    return "\n".join(out_lines)

def split_text_by_sentences(text, chunk_size=4000):
    sentences = re.split(r"(?<=[.!?]) +", text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) > chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            current_chunk += " " + sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def analyze_long_text_in_chunks_and_save(
    cleaned_text, context_query, output_file, expected_count, progress_data=None, question_lines=None
):
    system_message = (
        "System: You are a conscientious assistant. Respond solely based on the provided excerpt. "
        "Rules:\n"
        "1) For each question provide exactly ONE sentence OR exactly 'NO ANSWER!'.\n"
        "2) If the excerpt does not contain sufficient information OR you cannot give a literal quote, respond 'NO ANSWER!'.\n"
        "3) After each answer there must be a line 'Citation:' with a short literal quote (max 20 words) from the excerpt; "
        "for 'NO ANSWER!' do not output a citation line.\n"
        "4) Do not fabricate citations. No prior knowledge, only the excerpt.\n"
        "Format per question:\n"
        "<Nr>. Answer: <one sentence or NO ANSWER!>\n"
        "   Citation: \"<literal quote from the excerpt>\""
    )
    full_prompt_header = f"{system_message}\n\n{context_query}"

    kw_sets = []
    if question_lines:
        kw_sets = [extract_keywords(q) for q in question_lines]

    def sentence_chunks(text, chunk_size=4000):
        sentences = re.split(r"(?<=[.!?]) +", text)
        chunk, chunks = "", []
        for s in sentences:
            if len(chunk) + len(s) > chunk_size:
                if chunk.strip():
                    chunks.append(chunk.strip())
                chunk = s
            else:
                chunk += " " + s
        if chunk.strip():
            chunks.append(chunk.strip())
        return chunks

    chunks = sentence_chunks(cleaned_text, chunk_size=4000)
    with open(output_file, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            if progress_data and progress_data.get("abort"):
                logging.info("[ABORT] Analysis loop aborted")
                return output_file

            if not chunk.strip():
                f.write(f"\n\nResult for section {i+1}:\n")
                f.write("\n".join(f"{j+1}. NO ANSWER!" for j in range(expected_count)))
                continue

            logging.info(f"Progress: section {i+1}/{len(chunks)}")

            chunk_low = chunk.lower()

            prompt = f"{full_prompt_header}\n\nSection {i+1}:\n{chunk}"
            analysis_result = query_llm_via_cli(prompt)

            validated_result = validate_llm_response(analysis_result, expected_count)

            blocks = list(re.finditer(r'(?ms)^\s*(\d+)\.\s*(.*?)(?=^\s*\d+\.|\Z)', validated_result))
            rebuilt_blocks = []

            for m in blocks:
                n = int(m.group(1))
                block_text = m.group(2).strip()

                if kw_sets and n <= len(kw_sets):
                    kws = kw_sets[n-1]
                    if kws and not any(k in chunk_low for k in kws):
                        rebuilt_blocks.append(f"{n}. NO ANSWER!")
                        continue

                first_line = block_text.splitlines()[0] if block_text else ""
                if re.search(r'(?i)^\s*(answer:)?\s*no answer!?$', first_line):
                    rebuilt_blocks.append(f"{n}. NO ANSWER!")
                    continue

                ev_m = re.search(r'(?im)^\s*Citation:\s*[\"“]?(.+?)[\"”]?\s*$', block_text)
                if not ev_m:
                    rebuilt_blocks.append(f"{n}. NO ANSWER!")
                    continue

                ev_text = ev_m.group(1).strip()
                ev_low  = ev_text.lower()

                meta_patterns = [
                    r"no(?:\s+\w+){0,3}\s+term",
                    r"nowhere\s+in\s+the\s+text",
                    r"not\s+mentioned",
                    r"no\s+literal\s+quote",
                    r"no\s+answer",
                ]
                if any(re.search(p, ev_low) for p in meta_patterns):
                    rebuilt_blocks.append(f"{n}. NO ANSWER!")
                    continue

                if len(ev_text) < 10 or ev_low not in chunk_low:
                    rebuilt_blocks.append(f"{n}. NO ANSWER!")
                    continue

                if kw_sets and n <= len(kw_sets):
                    kws = kw_sets[n-1]
                    if kws and not any(k in ev_low for k in kws):
                        rebuilt_blocks.append(f"{n}. NO ANSWER!")
                        continue

                rebuilt_blocks.append(f"{n}. {block_text}")

            while len(rebuilt_blocks) < expected_count:
                rebuilt_blocks.append(f"{len(rebuilt_blocks)+1}. NO ANSWER!")

            result_text = f"\n\nResult for section {i+1}:\n" + "\n".join(rebuilt_blocks) + "\n"
            f.write(result_text)

            logging.info("[CHECKED_RESULT] Section %d:\n%s", i + 1, result_text)

    return output_file

def run_analysis(analysis_id, pdf_directory, context_query, expected_count, question_lines=None):
    global current_process
    try:
        logging.info(
            f"[RUN_ANALYSIS] started: analysis_id={analysis_id}, pdf_directory={pdf_directory}"
        )
        if not os.path.exists(pdf_directory):
            msg = f"PDF directory does not exist: {pdf_directory}"
            logging.error(f"[RUN_ANALYSIS] {msg}")
            analysis_progress[analysis_id]["status"] = "error"
            analysis_progress[analysis_id]["error"] = msg
            return

        pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith(".pdf")]
        total_files = len(pdf_files)
        logging.info(
            f"[RUN_ANALYSIS] Found PDF files: {total_files} in {pdf_directory}"
        )

        if total_files == 0:
            analysis_progress[analysis_id]["status"] = "error"
            analysis_progress[analysis_id]["error"] = "No PDF files found."
            return

        pdf_dir_name = os.path.basename(pdf_directory)
        current_date = datetime.now().strftime("%Y-%m-%d")
        analysis_output_folder = os.path.join(
            DATA_ROOT, f"{pdf_dir_name}_{current_date}"
        )
        os.makedirs(analysis_output_folder, exist_ok=True)

        results_summary = []
        for idx, filename in enumerate(pdf_files):
            if analysis_progress[analysis_id].get("abort"):
                analysis_progress[analysis_id]["status"] = "aborted"
                logging.info("[RUN_ANALYSIS] Analysis aborted by user.")
                with process_lock:
                    current_process["analysis"] = False
                return

            pdf_path = os.path.join(pdf_directory, filename)
            logging.info(
                f"[RUN_ANALYSIS] Processing {filename} ({idx+1}/{total_files}) ..."
            )
            try:
                extracted_text, page_count = extract_text_from_pdf(
                    pdf_path, abort_data=analysis_progress[analysis_id]
                )
                cleaned = clean_text(extracted_text)

                analysis_output_path = os.path.join(
                    analysis_output_folder, f"analysis_result_paper{idx+1}.txt"
                )
                analyze_long_text_in_chunks_and_save(
                    cleaned,
                    context_query,
                    analysis_output_path,
                    expected_count,
                    progress_data=analysis_progress[analysis_id],
                    question_lines=question_lines,
                )

                results_summary.append(
                    f"Analysis for {filename} completed (result in {analysis_output_path})."
                )

                progress = int(((idx + 1) / total_files) * 100)
                analysis_progress[analysis_id]["percent"] = progress
                logging.info(f"[RUN_ANALYSIS] Progress: {progress}%")
            except Exception as e:
                logging.exception(
                    f"[RUN_ANALYSIS] Error processing {filename}: {e}"
                )
                continue

        if not analysis_progress[analysis_id].get("error"):
            analysis_progress[analysis_id]["status"] = "completed"
            analysis_progress[analysis_id]["result"] = "\n".join(results_summary)
            logging.info(f"[RUN_ANALYSIS] completed: {analysis_id}")
    finally:
        with process_lock:
            current_process["analysis"] = False
            logging.info(
                f"[RUN_ANALYSIS] current_process['analysis'] set to False."
            )

# ---------------------------
# Functions: Data analysis
# ---------------------------
def get_stopwords():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")
    return nltk.corpus.stopwords.words("english")

english_stopwords = get_stopwords()
custom_stopwords = set(english_stopwords).union(
    {
        "also",
        "well",
        "just",
        "really",
        "like",
        "you know",
        "actually",
        "basically",
        "maybe",
        "probably",
        "kind of",
        "sort of",
        "get",
        "goes",
        "good",
        "whole",
        "narrow",
        "for that",
        "per",
        "three",
        "have",
        "including",
        "etc",
        "among",
        "becomes",
    }
)

def tokenize_texts(text_data):
    tokenized_texts = [
        [word for word in text.lower().split() if word not in custom_stopwords]
        for text in text_data
    ]
    return tokenized_texts

def find_optimal_num_topics(texts, min_topics=2, max_topics=15, step=1, alpha_value="auto", beta_value="auto", abort_data=None):
    tokenized_texts = tokenize_texts(texts)
    dictionary = Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

    coherence_scores = []
    for num_topics in range(min_topics, max_topics + 1, step):
        if abort_data and abort_data.get("abort"):
            logging.info("[ABORT] Topic search aborted")
            return 0

        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            random_state=42,
            iterations=100,
            passes=10,
            alpha=alpha_value,
            eta=beta_value,
        )
        coherence_model = CoherenceModel(
            model=lda_model,
            texts=tokenized_texts,
            dictionary=dictionary,
            coherence="c_v",
        )
        coherence_score = coherence_model.get_coherence()
        coherence_scores.append(coherence_score)
        logging.info(f"Number of topics: {num_topics} -- Coherence: {coherence_score:.4f}")

    optimal_topics = range(min_topics, max_topics + 1, step)[
        np.argmax(coherence_scores)
    ]
    logging.info(f"Optimal number of topics: {optimal_topics}")
    return optimal_topics

def perform_lda(text_data, num_topics, alpha_value="auto", beta_value="auto", abort_data=None):
    if abort_data and abort_data.get("abort"):
        logging.info("[ABORT] LDA training not started")
        return None, None, None, None

    tokenized_texts = tokenize_texts(text_data)
    dictionary = Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

    if abort_data and abort_data.get("abort"):
        logging.info("[ABORT] LDA training aborted")
        return None, None, None, None

    lda_model = LdaModel(
        corpus=corpus,
        id2word=dictionary,
        num_topics=num_topics,
        random_state=42,
        iterations=100,
        passes=10,
        alpha=alpha_value,
        eta=beta_value,
    )
    return lda_model, dictionary, corpus, tokenized_texts

def visualize_lda(
    lda_model, dictionary, corpus, question_id, p_w=None, output_directory=None
):
    if output_directory is None:
        output_directory = os.getcwd()
    try:
        lda_vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
        output_html_path = os.path.join(
            output_directory, f"lda_visualization_question_{question_id}.html"
        )
        pyLDAvis.save_html(lda_vis, output_html_path)
        print(
            f"Visualization for question {question_id} saved as HTML: {output_html_path}"
        )
    except Exception as e:
        print(f"Error during visualization for question {question_id}: {e}")

def get_top_terms_for_topic(
    lda_model, topic_id, dictionary, p_w, lambda_val=1.0, topn=30
):
    topic_terms = lda_model.get_topic_terms(topic_id, topn=len(dictionary))
    relevance_scores = []
    for word_id, prob in topic_terms:
        pw = p_w.get(dictionary[word_id], 1e-12)
        prob = max(prob, 1e-12)
        pw = max(pw, 1e-12)
        relevance = lambda_val * prob + (1 - lambda_val) * (
            math.log(prob) - math.log(pw)
        )
        relevance_scores.append((dictionary[word_id], relevance))
    relevance_scores.sort(key=lambda x: x[1], reverse=True)
    top_terms = [word for word, score in relevance_scores[:topn]]
    return top_terms

def plot_wordclouds(
    lda_model, dictionary, lambda_val=1.0, num_words=30, p_w=None, output_directory=None
):
    if output_directory is None:
        output_directory = os.getcwd()
    if p_w is None:
        total_count = sum(dictionary.cfs.values())
        p_w = {
            dictionary[token_id]: count / total_count
            for token_id, count in dictionary.cfs.items()
        }
    num_topics = lda_model.num_topics
    for topic_idx in range(num_topics):
        top_terms = get_top_terms_for_topic(
            lda_model, topic_idx, dictionary, p_w, lambda_val=lambda_val, topn=num_words
        )
        topic_terms = dict(lda_model.get_topic_terms(topic_idx, topn=num_words))
        word_freq = {}
        for word in top_terms:
            word_id = dictionary.token2id.get(word)
            if word_id is not None:
                word_freq[word] = topic_terms.get(word_id, 1e-12)
        wordcloud = WordCloud(
            width=800, height=400, background_color="white"
        ).generate_from_frequencies(word_freq)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.title(f"Wordcloud for Topic {topic_idx + 1} (λ={lambda_val})")
        plt.axis("off")
        output_path = os.path.join(
            output_directory, f"wordcloud_topic_{topic_idx + 1}_lambda_{lambda_val}.png"
        )
        plt.savefig(output_path)
        plt.close()
        print(
            f"Wordcloud for Topic {topic_idx + 1} (λ={lambda_val}) saved: {output_path}"
        )

def create_knowledge_graph(
    lda_model,
    dictionary,
    corpus,
    sorted_topic_indices,
    num_topics,
    lambda_val=1.0,
    p_w=None,
    output_html="knowledge_graph.html",
    prob_threshold=0.001,
):
    if output_html is None:
        output_html = os.path.join(ANALYSIS_ROOT, "knowledge_graph.html")
    if p_w is None:
        total_count = sum(dictionary.cfs.values())
        p_w = {
            dictionary[token_id]: count / total_count
            for token_id, count in dictionary.cfs.items()
        }

    G = nx.Graph()

    for sorted_idx, topic_idx in enumerate(sorted_topic_indices[:num_topics]):
        topic_name = f"Topic {sorted_idx + 1}"
        G.add_node(
            topic_name, label=topic_name, color="#00B0F0", size=30, is_topic=True
        )
        top_words = get_top_terms_for_topic(
            lda_model, topic_idx, dictionary, p_w, lambda_val=lambda_val, topn=30
        )
        for word in top_words:
            word_id = dictionary.token2id[word]
            word_prob = dict(lda_model.get_topic_terms(topic_idx, topn=30)).get(
                word_id, 0.0
            )
            if word_prob < prob_threshold:
                continue
            if not G.has_node(word):
                node_size = 15 + (word_prob * 100)
                G.add_node(
                    word, label=word, color="#00CC99", size=node_size, is_topic=False
                )
                G.nodes[word]["topic"] = topic_name
            G.add_edge(topic_name, word, weight=word_prob)

    for node, data in G.nodes(data=True):
        if data.get("is_topic"):
            continue
        connections = len(list(G.neighbors(node)))
        if connections > 3:
            data["color"] = "#FF5733"
        elif connections > 2:
            data["color"] = "#FFC300"
        elif connections > 1:
            data["color"] = "#D86ECC"
        else:
            data["color"] = "#00CC99"

    pos = nx.spring_layout(G, seed=42, k=0.5, iterations=200)
    edge_traces = []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_trace = go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            line=dict(width=1, color="#262626"),
            hoverinfo="none",
            mode="lines",
            showlegend=False,
        )
        edge_traces.append(edge_trace)

    topic_x, topic_y, topic_text = [], [], []
    word_1_x, word_1_y, word_1_text, word_1_hover = [], [], [], []
    word_2_x, word_2_y, word_2_text, word_2_hover = [], [], [], []
    word_3_x, word_3_y, word_3_text, word_3_hover = [], [], [], []
    word_4_x, word_4_y, word_4_text, word_4_hover = [], [], [], []

    def make_hover_label(node_name, node_data):
        connections = len(list(G.neighbors(node_name)))
        total_weight = sum(
            [G.edges[(node_name, nbr)]["weight"] for nbr in G.neighbors(node_name)]
        )
        connected_topics = [
            nbr for nbr in G.neighbors(node_name) if nbr.startswith("Topic")
        ]
        main_topic = node_data.get("topic", "No main topic")
        other_topics = [t for t in connected_topics if t != main_topic]
        return (
            f"Label: {node_data['label']}<br>"
            f"Connections: {connections}<br>"
            f"Total weight: {total_weight:.6f}<br>"
            f"Main topic: {main_topic}<br>"
            f"Subtopics: {', '.join(other_topics) if other_topics else 'None'}"
        )

    for node, data in G.nodes(data=True):
        x, y = pos[node]
        if data.get("is_topic"):
            topic_x.append(x)
            topic_y.append(y)
            topic_text.append(data["label"])
        else:
            hover_text = make_hover_label(node, data)
            connections = len(list(G.neighbors(node)))
            if connections == 1:
                word_1_x.append(x)
                word_1_y.append(y)
                word_1_text.append(data["label"])
                word_1_hover.append(hover_text)
            elif connections == 2:
                word_2_x.append(x)
                word_2_y.append(y)
                word_2_text.append(data["label"])
                word_2_hover.append(hover_text)
            elif connections == 3:
                word_3_x.append(x)
                word_3_y.append(y)
                word_3_text.append(data["label"])
                word_3_hover.append(hover_text)
            else:
                word_4_x.append(x)
                word_4_y.append(y)
                word_4_text.append(data["label"])
                word_4_hover.append(hover_text)

    topic_trace = go.Scatter(
        x=topic_x,
        y=topic_y,
        mode="markers+text",
        text=topic_text,
        textposition="top center",
        hoverinfo="text",
        name="Topics",
        marker=dict(color="#00B0F0", size=30, line=dict(width=2, color="#262626")),
    )
    word_1_trace = go.Scatter(
        x=word_1_x,
        y=word_1_y,
        mode="markers+text",
        text=word_1_text,
        textposition="top center",
        hovertext=word_1_hover,
        hoverinfo="text",
        name="Terms (1 Connection)",
        marker=dict(color="#00CC99", size=15, line=dict(width=2, color="#262626")),
    )
    word_2_trace = go.Scatter(
        x=word_2_x,
        y=word_2_y,
        mode="markers+text",
        text=word_2_text,
        textposition="top center",
        hovertext=word_2_hover,
        hoverinfo="text",
        name="Terms (2 Connections)",
        marker=dict(color="#D86ECC", size=15, line=dict(width=2, color="#262626")),
    )
    word_3_trace = go.Scatter(
        x=word_3_x,
        y=word_3_y,
        mode="markers+text",
        text=word_3_text,
        textposition="top center",
        hovertext=word_3_hover,
        hoverinfo="text",
        name="Terms (3 Connections)",
        marker=dict(color="#FFC300", size=15, line=dict(width=2, color="#262626")),
    )
    word_4_trace = go.Scatter(
        x=word_4_x,
        y=word_4_y,
        mode="markers+text",
        text=word_4_text,
        textposition="top center",
        hovertext=word_4_hover,
        hoverinfo="text",
        name="Terms (>3 Connections)",
        marker=dict(color="#FF5733", size=15, line=dict(width=2, color="#262626")),
    )

    fig = go.Figure(
        data=edge_traces
        + [topic_trace, word_1_trace, word_2_trace, word_3_trace, word_4_trace],
        layout=go.Layout(
            title=f"Interactive knowledge graph (λ={lambda_val})",
            titlefont_size=16,
            showlegend=True,
            hovermode="closest",
            clickmode="event+select",
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            plot_bgcolor="white",
        ),
    )

    fig.write_html(output_html)
    print(f"Knowledge graph (λ={lambda_val}) saved as HTML: {output_html}")

def get_combined_prompt(top_terms_freq, top_terms_excl, question_text):
    prompt = f"""
    For the following question: '{question_text}', two sets of top terms were identified after LDA topic modeling:

    - Based on frequency (λ = 1): {', '.join(top_terms_freq)}
    - Based on exclusivity (λ = 0): {', '.join(top_terms_excl)}

    Please interpret these two sets of top terms according to the following structure:

    1) Introduction: Brief overview of the topic.
    2) Most frequent terms (λ = 1): Explain why these terms appear particularly often and what relevance they have for answering the question.
    3) Most exclusive terms (λ = 0): Explain what these especially exclusive terms indicate and how they complement or specify the topic.
    4) Conclusion: Provide a concise summary of how the two sets of terms together help answer the question.

    Please respond in clear continuous prose without markdown, bullets, asterisks, or other special formatting.
    """
    return prompt

def query_ollama(input_text, model="llama3.1p"):
    try:
        process = subprocess.Popen(
            ["ollama", "run", model],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        stdout, stderr = process.communicate(input=input_text, timeout=180)
        if process.returncode != 0:
            print(f"Error during model request: {stderr.strip()}")
            return "Interpretation not possible"
        return stdout.strip()
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error retrieving interpretation"

def save_interpretations_to_txt(interpretations, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        for interpretation in interpretations:
            top_terms_freq = interpretation.get("Top Terms (frequency, λ=1)", "")
            top_terms_excl = interpretation.get("Top Terms (exclusivity, λ=0)", "")
            top_terms = f"Frequency (λ=1): {top_terms_freq}\nExclusivity (λ=0): {top_terms_excl}"
            file.write(f"Topic {interpretation['Topic']}\n")
            file.write("=" * 30 + "\n")
            file.write(f"Terms:\n{top_terms}\n")
            file.write("\nInterpretation:\n")
            file.write(f"{interpretation['Interpretation']}\n")
            file.write("\n" + "-" * 50 + "\n")
    print(f"Interpretations saved to '{file_path}'.")

def lda_analysis_with_interpretation(file_path, question_id, num_topics, model="llama3.1p", output_directory=None, abort_data=None):
    try:

        if abort_data and abort_data.get("abort"):
            logging.info(f"[LDA] Abort detected before starting analysis for question {question_id}.")
            return

        data = pd.read_csv(file_path, encoding="utf-8")
        text_data = data[f"Question {question_id} Answers"].dropna().tolist()
        if not text_data:
            print(f"No data available for LDA analysis of question {question_id}.")
            return

        lda_model, dictionary, corpus, _ = perform_lda(
            text_data, num_topics, abort_data=abort_data
        )
        if lda_model is None:
            return

        if abort_data and abort_data.get("abort"):
            logging.info("[ABORT] Aborted after LDA training")
            return

        total_count = sum(dictionary.cfs.values())
        p_w = {
            dictionary[token_id]: count / total_count
            for token_id, count in dictionary.cfs.items()
        }

        topic_totals = np.zeros(lda_model.num_topics)
        for doc in corpus:
            doc_topics = lda_model.get_document_topics(doc, minimum_probability=0)
            for topic_id, prob in doc_topics:
                topic_totals[topic_id] += prob
        sorted_topic_indices = np.argsort(topic_totals)[::-1]

        interpretations = []
        for cluster_idx, topic_idx in enumerate(sorted_topic_indices):

            if abort_data and abort_data.get("abort"):
                logging.info(f"[LDA] Abort during interpretation for question {question_id}, topic {cluster_idx+1}.")
                break

            top_terms_freq = get_top_terms_for_topic(
                lda_model, topic_idx, dictionary, p_w, lambda_val=1.0, topn=30
            )
            top_terms_excl = get_top_terms_for_topic(
                lda_model, topic_idx, dictionary, p_w, lambda_val=0.0, topn=30
            )

            combined_prompt = get_combined_prompt(
                top_terms_freq, top_terms_excl, f"Question {question_id}"
            )

            if abort_data and abort_data.get("abort"):
                logging.info(f"[LDA] Abort before LLM interpretation for question {question_id}, topic {cluster_idx+1}.")
                break

            combined_output = query_ollama(combined_prompt, model)
            interpretation_sections = combined_output.split("\n", 1)
            interpretations.append(
                {
                    "Topic": cluster_idx + 1,
                    "Top Terms (frequency, λ=1)": ", ".join(top_terms_freq),
                    "Top Terms (exclusivity, λ=0)": ", ".join(top_terms_excl),
                    "Interpretation": "\n".join(interpretation_sections).strip(),
                }
            )

        if abort_data and abort_data.get("abort"):
            logging.info(f"[LDA] Full abort for question {question_id}, saving partial results if any.")

        if output_directory is None:
            output_directory = os.path.join(
                ANALYSIS_ROOT,
                "evaluation_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
            )
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

        if not (abort_data and abort_data.get("abort")):
            visualize_lda(
                lda_model,
                dictionary,
                corpus,
                question_id,
                p_w=p_w,
                output_directory=output_directory,
            )
            plot_wordclouds(
                lda_model,
                dictionary,
                lambda_val=1.0,
                num_words=30,
                p_w=p_w,
                output_directory=output_directory,
            )
            plot_wordclouds(
                lda_model,
                dictionary,
                lambda_val=0.0,
                num_words=30,
                p_w=p_w,
                output_directory=output_directory,
            )

            create_knowledge_graph(
                lda_model=lda_model,
                dictionary=dictionary,
                corpus=corpus,
                sorted_topic_indices=sorted_topic_indices,
                num_topics=optimal_topics,
                lambda_val=1.0,
                p_w=p_w,
                output_html=os.path.join(
                    output_directory,
                    f"knowledge_graph_question_{question_id}_lambda_1.html",
                ),
            )
            create_knowledge_graph(
                lda_model=lda_model,
                dictionary=dictionary,
                corpus=corpus,
                sorted_topic_indices=sorted_topic_indices,
                num_topics=optimal_topics,
                lambda_val=0.0,
                p_w=p_w,
                output_html=os.path.join(
                    output_directory,
                    f"knowledge_graph_question_{question_id}_lambda_0.html",
                ),
            )

        output_txt_path = os.path.join(
            output_directory, f"interpretations_question_{question_id}.txt"
        )
        save_interpretations_to_txt(interpretations, output_txt_path)

        print("LDA analysis completed. Results in folder:", output_directory)
    except Exception as e:
        print(
            f"Error in lda_analysis_with_interpretation for question {question_id}: {e}"
        )

def load_text_files(directory):
    data = []
    for filename in os.listdir(directory):
        if not filename.endswith(".txt"):
            continue
        path = os.path.join(directory, filename)
        content = None
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                with open(path, "r", encoding=encoding, errors="strict") as file:
                    content = file.read().strip()
                break
            except UnicodeDecodeError:
                continue
        if content is None:
            with open(path, "r", encoding="utf-8", errors="replace") as file:
                content = file.read().strip()
        if content:
            data.append(content)
    return data

def categorize_responses_by_question(data):
    all_sections = {}
    for text in data:
        pattern_section = (
            r"Result for section \d+:\s*((?:(?:\d+\.\s*.*?(?=\n\d+\.|\Z)))+)"
        )
        sections = re.findall(pattern_section, text, re.DOTALL)
        print(f"DEBUG: Sections found: {sections}")
        for idx, sec in enumerate(sections, start=1):
            answers = re.findall(r"\d+\.\s*(.*?)(?=\n\d+\.|\Z)", sec, re.DOTALL)
            answers = [ans.strip() for ans in answers]
            if answers:
                all_sections.setdefault(idx, []).extend(answers)
    return all_sections

def sediment_analysis(directory, analysis_id):
    global current_process
    logging.info(f"[SEDIMENT] STARTED for {directory}, analysis_id={analysis_id}")
    sediment_progress[analysis_id].update(status="running", percent=0)

    data = load_text_files(directory)
    if not data:
        logging.error("[SEDIMENT] No texts found")
        sediment_progress[analysis_id].update(
            status="error", error="No texts found"
        )
        return

    responses_by_question = categorize_responses_by_question(data)
    logging.info(f"[SEDIMENT] Answers by question: {responses_by_question}")

    filtered_responses = {
        q_id: [r for r in responses if "no answer" not in r.lower()]
        for q_id, responses in responses_by_question.items()
        if any(r.strip() for r in responses)
    }

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    evaluation_folder = os.path.join(ANALYSIS_ROOT, f"evaluation_{timestamp}")
    os.makedirs(evaluation_folder, exist_ok=True)
    logging.info(f"[SEDIMENT] Evaluation folder: {evaluation_folder}")

    questions_file = os.path.join(directory, "questions.txt")
    if os.path.exists(questions_file):
        with open(questions_file, "r", encoding="utf-8") as f:
            questions = [line.strip() for line in f if line.strip()]
    else:
        questions = []
        logging.warning("[SEDIMENT] No questions.txt found.")

    total_questions = len(filtered_responses)
    if total_questions == 0:
        logging.error("[SEDIMENT] No valid answers after filtering")
        sediment_progress[analysis_id].update(
            status="error", error="No valid answers found"
        )
        return
    sediment_progress[analysis_id].update(percent=10)
    logging.info("[SEDIMENT] CSV export completed, starting LDA analysis")

    for idx, (q_id, responses) in enumerate(filtered_responses.items(), start=1):
        if sediment_progress[analysis_id].get("abort"):
            logging.info("[ABORT] Sediment loop aborted")
            sediment_progress[analysis_id].update(status="aborted")
            with process_lock:
                current_process = None
            return

        if q_id > len(questions):
            continue

        if sediment_progress[analysis_id].get("abort"):
            sediment_progress[analysis_id].update(status="aborted")
            logging.info("[SEDIMENT] Abort detected just before LDA analysis.")
            with process_lock:
                current_process = None
            return

        question_folder = os.path.join(evaluation_folder, f"question_{q_id}")
        os.makedirs(question_folder, exist_ok=True)

        question_text = questions[q_id - 1]
        with open(
            os.path.join(question_folder, f"question_{q_id}.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(question_text)

        df = pd.DataFrame({f"Question {q_id} Answers": responses})
        csv_path = os.path.join(question_folder, f"question_{q_id}_responses.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8")
        logging.info(f"[SEDIMENT] Answers for question {q_id} saved in {csv_path}")

        percent = 10 + int((idx / total_questions) * 80)
        sediment_progress[analysis_id].update(percent=percent)
        logging.info(
            f"[SEDIMENT] Question {q_id}/{total_questions} processed, progress {percent}%"
        )

        optimal_topics = find_optimal_num_topics(
            responses, min_topics=2, max_topics=15,
            step=1, abort_data=sediment_progress[analysis_id]
        )

        logging.info(f"[SEDIMENT] Optimal topics for question {q_id}: {optimal_topics}")

        if sediment_progress[analysis_id].get("abort"):
            sediment_progress[analysis_id].update(status="aborted")
            logging.info("[SEDIMENT] Abort detected before interpretation.")
            with process_lock:
                current_process = None
            return

        lda_analysis_with_interpretation(
            csv_path,
            q_id,
            num_topics=optimal_topics,
            model="llama3.1p",
            output_directory=question_folder,
            abort_data=sediment_progress[analysis_id],
        )
        logging.info(
            f"[SEDIMENT] LDA analysis and interpretation for question {q_id} completed"
        )

    logging.info(f"[SEDIMENT] Data evaluation completed for {directory}")
    sediment_progress[analysis_id].update(status="completed", percent=100, folder=evaluation_folder)
    with process_lock:
        current_process = None

# ---------------------------
# Endpoints: Data analysis
# ---------------------------
@app.route("/list_data_directories", methods=["GET"])
def list_data_directories():
    dirs = [
        d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))
    ]
    if not dirs:
        dirs = ["No directories found"]
    return jsonify({"directories": dirs})

def sediment_analysis_background(analysis_id, data_directory):
    global current_process
    try:
        sediment_analysis(data_directory, analysis_id)
        current = sediment_progress.get(analysis_id, {})
        if current.get("status") not in ("error", "aborted", "completed"):
            sediment_progress[analysis_id].update(status="completed", percent=100)
    except Exception as e:
        logging.exception("[SEDIMENT] Unexpected error")
        sediment_progress[analysis_id].update(status="error", error=str(e))
    finally:
        with process_lock:
            current_process["sediment"] = False

@app.route("/start_sediment_analysis", methods=["POST"])
def start_sediment_analysis():
    global current_process
    data = request.get_json()
    selected_dir = data.get("data_directory")
    if not selected_dir:
        return jsonify({"error": "Please select a directory in the data folder."})

    full_path = os.path.join(DATA_ROOT, selected_dir)
    if not os.path.exists(full_path):
        return jsonify({"error": "The selected directory does not exist."})

    with process_lock:
        if current_process.get("sediment"):
            return (
                jsonify({"error": "A data evaluation process is already running."}),
                400,
            )
        current_process["sediment"] = True

    analysis_id = str(uuid.uuid4())
    sediment_progress[analysis_id] = {"status": "running", "percent": 0, "abort": False}

    thread = threading.Thread(
        target=sediment_analysis_background, args=(analysis_id, full_path)
    )
    thread.daemon = False
    thread.start()

    return jsonify({"analysis_id": analysis_id})

@app.route("/sediment_analysis_progress", methods=["GET"])
def get_sediment_analysis_progress():
    analysis_id = request.args.get("id")
    if not analysis_id or analysis_id not in sediment_progress:
        return jsonify({"error": "Invalid analysis ID."})
    return jsonify(sediment_progress[analysis_id])

# ---------------------------
# Front-end content
# ---------------------------
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>FACTS V2.5</title>
  <link rel="stylesheet" href="/styles.css">
</head>
<body>
  <div class="container">
    <h1>FACTS V2.5</h1>

    <div class="section">
      <h3>1: DATA ACQUISITION</h3>
      <label for="database-select">Select database:</label>
      <select id="database-select">
        <option value="pedocs">peDOCs</option>
        <option value="arxiv">Arxiv</option>
        <option value="eric">ERIC</option>
      </select>
      <label for="search-query">Search terms:</label>
      <input type="text" id="search-query" placeholder="e.g. machine learning">
      <label for="publication-year">Publication year:</label>
      <input type="text" id="publication-year" placeholder="e.g. 2023">
      <label for="num-papers">Number of papers:</label>
      <input type="number" id="num-papers" min="1" max="150" value="10">
      <div class="buttons">
        <button id="download-btn">Download data</button>
        <button id="abort-btn" style="display: none;">Abort</button>
      </div>
      <div id="spinner" class="spinner" style="display: none;"></div>
      <div id="progress-container" style="display: none;">
        <progress id="progress-bar" value="0" max="100"></progress>
        <p id="progress-text">0% completed</p>
      </div>
      <div id="results" class="result-container" style="display: none;"></div>
    </div>
    <hr>

    <div class="section">
      <h3>2: DATA ANALYSIS</h3>
      <form id="analyse-form">
        <label for="pdf-directory">Select PDF directory:</label>
        <select id="pdf-directory" name="pdf_directory" required>
          <option value="">Please choose a directory</option>
        </select>
        <div id="upload-area" class="upload-area">
            <p>Drag & Drop your PDFs here or click to select them</p>
            <input type="file" id="pdf-upload" accept=".pdf" multiple style="display:none;">
            <button id="upload-btn">Upload PDF files</button>
        </div>
        <label for="questions">Questions (one per line):</label>
        <textarea id="questions" name="questions" rows="5" required placeholder="e.g. Which central factors influence educational practice?"></textarea>
        <div class="buttons" style="text-align: center; margin-top: 15px;">
          <button type="submit">Start analysis</button>
          <button id="abort-analysis-btn" style="display:none;">Abort</button>
        </div>
      </form>
      <div id="analyse-progress-container" style="display: none; margin-top: 15px;">
        <progress id="analyse-progress-bar" value="0" max="100"></progress>
        <p id="analyse-progress-text">0% completed</p>
      </div>
      <div id="analyse-spinner" class="spinner" style="display: none;"></div>
      <pre id="analyse-result" class="result-container" style="display: none;"></pre> 
    </div>
    <hr>

    <div class="section" id="sediment-analysis-section">
      <h3>3: DATA EVALUATION</h3>
      <label for="data-directory">Select data directory:</label>
      <select id="data-directory" name="data_directory" required>
        <option value="">Please choose a directory</option>
      </select>
      <div class="buttons">
        <button id="start-sediment-btn">Start evaluation</button>
        <button id="abort-sediment-btn" style="display:none;">Abort</button>
      </div>
      <div class="progress-bar-container" style="display: none;">
        <div class="progress-bar"></div>
      </div>
      <pre id="sediment-result"></pre>
    </div>
  </div>
  <script src="/script.js"></script>
</body>
</html>
"""

CSS_CONTENT = """
body, input, select, textarea, button, pre {
  font-family: Arial, sans-serif;
  font-size: 16px;
  font-color: #262626;
}

button:disabled {
  opacity: 0.5;  
  cursor: not-allowed; 
}

button.running:disabled {
  opacity: 1;
  cursor: default;
}

body {
  background-color: #ffffff;
  margin: 0;
  padding: 20px;
}

.container {
  width: 90%;
  max-width: 900px;
  margin: auto;
  background: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 0 10px rgba(0,0,0,0.1);
  border: 3px solid #262626;
}

h1, h2, h3 {
  text-align: center;
  color: #333;
}

.progress-bar-container {
  width: 100%;
  height: 20px;
  background-color: #ffffff;
  border: 3px solid #262626;
  overflow: hidden;
  position: relative;
}

.progress-bar {
  height: 100%;
  background-color: #00B0F0;
  position: absolute;
  left: -100%;
  animation: progressAnimation 2s linear infinite;
}

@keyframes progressAnimation {
  0% {
    left: -100%;
    width: 100%;
  }
  50% {
    left: 0;
    width: 100%;
  }
  100% {
    left: 100%;
    width: 0;
  }
}

label {
  display: block;
  margin-top: 10px;
}

input[type="text"], input[type="number"], select {
  width: 100%;
  padding: 10px;
  margin: 10px 0;
  border: 3px solid #262626;
  border-radius: 5px;
  box-sizing: border-box;
}

input[type="text"]:focus,
input[type="number"]:focus,
select:focus,
textarea:focus {
  border-color: #262626;
  background-color: #00B0F0;  
  outline: none;   
}

textarea {
  width: 100%;
  padding: 10px;
  margin: 10px 0;
  border: 3px solid #262626;
  border-radius: 5px;
  box-sizing: border-box;
  resize: vertical;
}

.buttons {
  text-align: center;
  margin: 10px 0;
}

button {
  padding: 10px 20px;
  margin: 10px 5px;
  border-radius: 5px;
  border: 3px solid #262626;
  background-color: #ffffff;
  color: #262626;
  cursor: pointer;
  font-size: 1em;
  transition: background-color 0.3s ease;
}

button:hover {
  border: 3px solid #262626;
  background-color: #262626;
  color: #ffffff;
}

#download-btn.running {
  background-color: #00B0F0;
  font-weight: bold;
  color: #262626;
}

#analyse-form button.running,
#start-sediment-btn.running {
  background-color: #00B0F0;
  font-weight: bold;
  color: #262626;
}

.upload-area {
  border: 3px dashed #262626;  
  background-color: #ffffff;     
  padding: 20px;
  text-align: center;
  margin: 10px 0;
  cursor: pointer;
  border-radius: 8px; 
}
.upload-area.dragover {
  background-color: #9BE5FF;
  border: 3px solid #00B0F0;   
}

.result-container {
  background-color: #00B0F0;
  border: 3px solid #262626;
  padding: 15px;
  border-radius: 8px;
  margin-top: 20px;
  white-space: pre-wrap;
  text-align: left;
}

progress {
  width: 100%;
  height: 20px;
  appearance: none;
  -webkit-appearance: none;
  border: 3px solid #262626;
  border-radius: 0;
  overflow: hidden;
}

.section {
  margin-bottom: 40px;
}

.spinner {
  border: 8px solid #262626;
  border-top: 8px solid #00B0F0;
  border-radius: 50%;
  width: 50px;
  height: 50px;
  animation: spin 1s linear infinite;
  margin: 20px auto;
}

@keyframes spin {
  0% { transform: rotate(0deg); }
  100% { transform: rotate(360deg); }
}

progress::-webkit-progress-bar {
  background-color: #eee;
  border: none;
}

progress::-webkit-progress-value {
  background-color: #00B0F0;
}

progress::-moz-progress-bar {
  background-color: #00B0F0
}
"""

JS_CONTENT = """
document.addEventListener('DOMContentLoaded', function() {

  function disableAllProcessButtons() {
    const downloadBtn = document.getElementById('download-btn');
    const analysisBtn = document.querySelector('#analyse-form button[type="submit"]');
    const sedimentBtn = document.getElementById('start-sediment-btn');
    const abortAnalysisBtn = document.getElementById('abort-analysis-btn');
    const abortSedimentBtn = document.getElementById('abort-sediment-btn');

    if (downloadBtn) downloadBtn.disabled = true;
    if (analysisBtn) analysisBtn.disabled = true;
    if (sedimentBtn) sedimentBtn.disabled = true;
    if (abortAnalysisBtn) abortAnalysisBtn.disabled = true;
    if (abortSedimentBtn) abortSedimentBtn.disabled = true;
  }

  function enableAllProcessButtons() {
    const downloadBtn = document.getElementById('download-btn');
    const analysisBtn = document.querySelector('#analyse-form button[type="submit"]');
    const sedimentBtn = document.getElementById('start-sediment-btn');
    const abortAnalysisBtn = document.getElementById('abort-analysis-btn');
    const abortSedimentBtn = document.getElementById('abort-sediment-btn');

    if (downloadBtn) downloadBtn.disabled = false;
    if (analysisBtn) analysisBtn.disabled = false;
    if (sedimentBtn) sedimentBtn.disabled = false;
    if (abortAnalysisBtn) abortAnalysisBtn.disabled = false;
    if (abortSedimentBtn) abortSedimentBtn.disabled = false;
  }

  function disableOtherProcessButtons(current) {
    const downloadBtn = document.getElementById('download-btn');
    const analysisBtn = document.querySelector('#analyse-form button[type="submit"]');
    const sedimentBtn = document.getElementById('start-sediment-btn');
    const abortAnalysisBtn = document.getElementById('abort-analysis-btn');
    const abortSedimentBtn = document.getElementById('abort-sediment-btn');

    if (downloadBtn) downloadBtn.disabled = true;
    if (analysisBtn) analysisBtn.disabled = true;
    if (sedimentBtn) sedimentBtn.disabled = true;
    if (abortAnalysisBtn) abortAnalysisBtn.disabled = true;
    if (abortSedimentBtn) abortSedimentBtn.disabled = true;

    if (current === "download" && downloadBtn) {
        downloadBtn.disabled = false;
    } else if (current === "analysis" && analysisBtn) {
        analysisBtn.disabled = false;
    } else if (current === "sediment" && sedimentBtn) {
        sedimentBtn.disabled = false;
    }
  }

  function setButtonRunning(btn, runningText) {
    if (!btn) return;
    btn.classList.add('running');
    btn.disabled = true;
    btn.innerText = runningText;
  }

  function clearButtonRunning(btn, defaultText) {
    if (!btn) return;
    btn.classList.remove('running');
    btn.disabled = false;
    btn.innerText = defaultText;
    btn.style.backgroundColor = "";
  }

  function resetDownloadUI() {
    clearInterval(progressInterval);
    spinner.style.display = 'none';
    clearButtonRunning(downloadBtn, "Daten herunterladen");
    abortBtn.style.display = 'none';
    resultsDiv.style.display = 'none';
    resultsDiv.innerText = '';
    progressContainer.style.display = 'none';
    progressBar.value = 0;
    progressText.innerText = "0% abgeschlossen";
  }

  function resetAnalysisUI() {
    const submitBtn = document.querySelector('#analyse-form button[type="submit"]');
    const analyseProgressContainer = document.getElementById("analyse-progress-container");
    const analyseSpinner = document.getElementById("analyse-spinner");
    const analyseProgressBar = document.getElementById("analyse-progress-bar");
    const analyseProgressText = document.getElementById("analyse-progress-text");
    const abortAnalysisBtnLocal = document.getElementById('abort-analysis-btn');

    clearButtonRunning(submitBtn, "Analyse starten");

    analyseSpinner.style.display = "none";
    analyseProgressContainer.style.display = "none";
    analyseProgressBar.value = 0;
    analyseProgressText.innerText = "0% abgeschlossen";
    if (abortAnalysisBtnLocal) {
      abortAnalysisBtnLocal.style.display = 'none';
      abortAnalysisBtnLocal.disabled = false;
    }
  }

  const downloadBtn = document.getElementById('download-btn');
  const abortBtn = document.getElementById('abort-btn');
  const databaseSelect = document.getElementById('database-select');
  const searchQuery = document.getElementById('search-query');
  const publicationYear = document.getElementById('publication-year');
  const numPapersInput = document.getElementById('num-papers');
  const progressContainer = document.getElementById('progress-container');
  const analysisResultsContainer = document.getElementById("analyse-result");
  const sedimentResultsContainer  = document.getElementById("sediment-result");
  const progressBar = document.getElementById('progress-bar');
  const progressText = document.getElementById('progress-text');
  const resultsDiv = document.getElementById('results');
  const spinner = document.getElementById('spinner');
  const abortAnalysisBtn = document.getElementById('abort-analysis-btn');
  const abortSedimentBtn = document.getElementById('abort-sediment-btn');
  const sedimentBtn = document.getElementById('start-sediment-btn');

  analysisResultsContainer.style.display = "none";
  sedimentResultsContainer.style.display = "none";
  if (abortAnalysisBtn) abortAnalysisBtn.style.display = 'none';
  if (abortSedimentBtn) abortSedimentBtn.style.display = 'none';

  let progressInterval;
  let currentDownloadId = null;
  let currentAnalysisId = null;
  let currentSedimentId = null;
  let analysisInterval;
  let sedimentInterval;

  downloadBtn.addEventListener('click', function() {
    const db = databaseSelect.value;
    const query = searchQuery.value.trim();
    const year = publicationYear.value.trim();
    const numPapers = parseInt(numPapersInput.value);

    if (!query || !year || !numPapers) {
        alert("Please complete all fields: search query, year, and number of papers.");
        return;
    }

    disableOtherProcessButtons("download");

    setButtonRunning(downloadBtn, "Process running...");
    abortBtn.style.display = 'inline-block';
    abortBtn.disabled = false;
    spinner.style.display = 'block';
    resultsDiv.style.display = 'none';
    resultsDiv.innerText = '';
    progressContainer.style.display = 'block';
    progressBar.value = 0;
    progressText.innerText = "0% completed";

    fetch('/download_papers', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ database: db, query: query, year: year, num_papers: numPapers })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
          alert("Error: " + data.error);
          enableAllProcessButtons();
          resetDownloadUI();
          return;
        }
        currentDownloadId = data.download_id;

        progressInterval = setInterval(function() {
          fetch('/download_progress?id=' + currentDownloadId)
              .then(resp => resp.json())
              .then(progressData => {
                  progressBar.value = progressData.percent || 0;
                  progressText.innerText = (progressData.percent || 0) + "% completed";
                  if (["completed","error","aborted"].includes(progressData.status)) {
                      clearInterval(progressInterval);
                      spinner.style.display = 'none';
                      enableAllProcessButtons();

                      clearButtonRunning(downloadBtn, "Download data");

                      abortBtn.style.display = 'none';

                      if (progressData.status === "completed") {
                          resultsDiv.style.display = 'block';
                          resultsDiv.innerText = "Download complete. Files have been saved in folder " + db + ".";
                      } else if (progressData.status === "aborted") {
                          resultsDiv.style.display = 'block';
                          resultsDiv.innerText = "Download aborted.";
                      } else {
                          resultsDiv.style.display = 'block';
                          resultsDiv.innerText = "Error during download: " + (progressData.error || 'Unknown');
                      }
                      currentDownloadId = null;
                  }
              })
              .catch(error => {
                  clearInterval(progressInterval);
                  console.error("Error fetching progress:", error);
                  enableAllProcessButtons();
                  resetDownloadUI();
                  currentDownloadId = null;
              });
        }, 1000);
    })
    .catch(error => {
        console.error("Error starting download:", error);
        alert("Error starting the download.");
        enableAllProcessButtons();
        resetDownloadUI();
        currentDownloadId = null;
    });
  });

  abortBtn.addEventListener('click', function() {
    if (!currentDownloadId) {
        alert("No download process is currently running.");
        return;
    }
    abortBtn.disabled = true;
    fetch('/abort_download', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ download_id: currentDownloadId })
    })
    .then(response => response.json())
    .then(data => {
        console.log(data.message || data.error);
        clearInterval(progressInterval);
        spinner.style.display = 'none';
        progressContainer.style.display = 'none';

        clearButtonRunning(downloadBtn, "Download data");
        abortBtn.style.display = 'none';
        resultsDiv.style.display = 'block';
        resultsDiv.innerText = "Download aborted.";
        currentDownloadId = null;
        enableAllProcessButtons();
    })
    .catch(error => {
        console.error("Error aborting the download:", error);
    });
  });

  abortAnalysisBtn.addEventListener('click', function() {
    if (!currentAnalysisId) {
        alert("No analysis process is currently running.");
        return;
    }
    abortAnalysisBtn.disabled = true;
    fetch('/abort_analysis', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ analysis_id: currentAnalysisId })
    })
    .then(res => res.json())
    .then(data => {
        console.log(data.message || data.error);
        clearInterval(analysisInterval);
        analysisResultsContainer.style.display = 'block';
        analysisResultsContainer.innerText = "Analysis aborted.";
        resetAnalysisUI();
        enableAllProcessButtons();
        currentAnalysisId = null;
    })
    .catch(err => {
        console.error("Error aborting the analysis:", err);
        resetAnalysisUI();
        enableAllProcessButtons();
    });
  });

  abortSedimentBtn.addEventListener('click', function() {
    if (!currentSedimentId) {
        alert("No data evaluation process is currently running.");
        return;
    }
    abortSedimentBtn.disabled = true;
    fetch('/abort_sediment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ analysis_id: currentSedimentId })
    })
    .then(res => res.json())
    .then(data => {
        console.log(data.message || data.error);
        clearInterval(sedimentInterval);
        document.querySelector('.progress-bar-container').style.display = 'none';

        clearButtonRunning(sedimentBtn, "Start evaluation");

        sedimentResultsContainer.classList.add('result-container');
        sedimentResultsContainer.style.display = 'block';
        sedimentResultsContainer.innerText = "Data evaluation aborted.";

        enableAllProcessButtons();
        currentSedimentId = null;
        if (abortSedimentBtn) {
          abortSedimentBtn.style.display = 'none';
          abortSedimentBtn.disabled = false;
        }
    })
    .catch(err => {
        console.error("Error aborting data evaluation:", err);
        document.querySelector('.progress-bar-container').style.display = 'none';
        sedimentResultsContainer.classList.add('result-container');
        sedimentResultsContainer.style.display = 'block';
        sedimentResultsContainer.innerText = "Error aborting data evaluation.";
        enableAllProcessButtons();
        if (abortSedimentBtn) {
            abortSedimentBtn.style.display = 'none';
            abortSedimentBtn.disabled = false;
        }
    });
  });

  function updateDirectories() {
    fetch('/list_directories')
    .then(response => response.json())
    .then(data => {
        const select = document.getElementById('pdf-directory');
        const currentSelection = select.value;

        let newOptions = '<option value="">Please choose a directory</option>';
        data.directories.forEach(dir => {
          const safeDir = dir.replace(/\\\\/g, '\\\\\\\\');
          newOptions += `<option value="${safeDir}">${safeDir}</option>`;
        });

        select.innerHTML = newOptions;

        if (currentSelection && Array.from(select.options).some(opt => opt.value === currentSelection)) {
          select.value = currentSelection;
        }
    })
    .catch(error => console.error("Error loading directories:", error));
  }
  setInterval(updateDirectories, 4000);

  function updateDataDirectories() {
    fetch('/list_data_directories')
      .then(response => response.json())
      .then(data => {
        const select = document.getElementById('data-directory');
        const currentSelection = select.value;
        let newOptions = '<option value="">Please choose a directory</option>';
        data.directories.forEach(dir => {
          const safeDir = dir.replace(/\\\\/g, '\\\\\\\\');
          newOptions += `<option value="${safeDir}">${safeDir}</option>`;
        });
        select.innerHTML = newOptions;
        if (currentSelection && Array.from(select.options).some(opt => opt.value === currentSelection)) {
          select.value = currentSelection;
        }
      })
      .catch(error => console.error("Error loading data directories:", error));
  }
  setInterval(updateDataDirectories, 5000);
  updateDataDirectories();

  const uploadArea = document.getElementById('upload-area');
  const pdfUpload = document.getElementById('pdf-upload');

  uploadArea.addEventListener('click', function(){
      pdfUpload.click();
  });
  uploadArea.addEventListener('dragover', function(e){
      e.preventDefault();
      this.classList.add('dragover');
  });
  uploadArea.addEventListener('dragleave', function(e){
      e.preventDefault();
      this.classList.remove('dragover');
  });
  uploadArea.addEventListener('drop', function(e){
      e.preventDefault();
      this.classList.remove('dragover');
      let files = e.dataTransfer.files;
      uploadFiles(files);
  });
  pdfUpload.addEventListener('change', function(){
      uploadFiles(this.files);
  });
  function uploadFiles(files) {
      const formData = new FormData();
      for(let i=0; i<files.length; i++){
          formData.append('files', files[i]);
      }
      fetch('/upload_pdfs', {
          method: 'POST',
          body: formData
      })
      .then(response => response.json())
      .then(data => {
          if(data.error){
              alert("Upload-Error: " + data.error);
          } else {
              alert("Upload successful! Files were saved in folder " + data.folder + ".");
          }
      })
      .catch(error => {
          console.error("Error during upload:", error);
          alert("Fehler beim Upload.");
      });
  }

  sedimentBtn.addEventListener('click', function() {
    disableOtherProcessButtons("sediment");

    const dataDir = document.getElementById('data-directory').value;
    if (!dataDir) {
      alert("Please select a data directory.");
      enableAllProcessButtons();
      return;
    }

    setButtonRunning(sedimentBtn, "Process running...");
    document.querySelector('.progress-bar-container').style.display = 'block';
    sedimentResultsContainer.style.display = 'none';

    fetch('/start_sediment_analysis', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data_directory: dataDir })
    })
    .then(response => response.json())
    .then(data => {
      if (data.error) {
        alert("Error: " + data.error);
        document.querySelector('.progress-bar-container').style.display = 'none';
        enableAllProcessButtons();
        clearButtonRunning(sedimentBtn, "Start evaluation");
        return;
      }
      currentSedimentId = data.analysis_id;
      if (abortSedimentBtn) {
        abortSedimentBtn.style.display = 'inline-block';
        abortSedimentBtn.disabled = false;
      }

      sedimentInterval = setInterval(function() {
        fetch('/sediment_analysis_progress?id=' + currentSedimentId)
          .then(resp => resp.json())
          .then(progressData => {
            if (["completed", "error"].includes(progressData.status)) {
              clearInterval(sedimentInterval);
              document.querySelector('.progress-bar-container').style.display = 'none';
              enableAllProcessButtons();
              clearButtonRunning(sedimentBtn, "Start evaluation");

              if (progressData.status === "completed") {
                  const folderName = progressData.folder || document.getElementById('data-directory').value;
                  sedimentResultsContainer.style.display = 'block';
                  sedimentResultsContainer.innerText =
                    `Data evaluation completed. Results were saved in folder "${folderName}".`;
              } else {
                  sedimentResultsContainer.style.display = 'block';
                  sedimentResultsContainer.innerText =
                    `Error during data evaluation: ${progressData.error || 'Unknown'}`;
              }
              if (abortSedimentBtn) abortSedimentBtn.style.display = 'none';
              currentSedimentId = null;
            }
          })
          .catch(error => {
            clearInterval(sedimentInterval);
            console.error("Error fetching data evaluation progress:", error);
            enableAllProcessButtons();
            document.querySelector('.progress-bar-container').style.display = 'none';
            clearButtonRunning(sedimentBtn, "Start evaluation");
          });
      }, 1000);
    })
    .catch(error => {
      console.error("Error starting data evaluation", error);
      enableAllProcessButtons();
      document.querySelector('.progress-bar-container').style.display = 'none';
      clearButtonRunning(sedimentBtn, "Start evaluation");
    });
  });

  const analyseForm = document.getElementById("analyse-form");
  analyseForm.addEventListener("submit", function(e) {
    e.preventDefault();

    disableOtherProcessButtons("analysis");

    const submitBtn = document.querySelector('#analyse-form button[type="submit"]');
    const analyseProgressContainer = document.getElementById("analyse-progress-container");
    const analyseProgressBar = document.getElementById("analyse-progress-bar");
    const analyseProgressText = document.getElementById("analyse-progress-text");
    const analyseSpinner = document.getElementById("analyse-spinner");

    analyseProgressContainer.style.display = "block";
    analyseSpinner.style.display = "block";
    analyseProgressBar.value = 0;
    analyseProgressText.innerText = "0% completed";
    analysisResultsContainer.style.display = 'none';

    setButtonRunning(submitBtn, "Process running…");

    const data = {
      pdf_directory: document.getElementById("pdf-directory").value,
      questions: document.getElementById("questions").value
    };

    fetch('/start_analysis', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
      if (result.error) {
          alert(result.error);
          enableAllProcessButtons();
          resetAnalysisUI();
          return;
      }
      currentAnalysisId = result.analysis_id;
      if (abortAnalysisBtn) {
        abortAnalysisBtn.style.display = 'inline-block';
        abortAnalysisBtn.disabled = false;
      }

      analysisInterval = setInterval(function() {
        fetch('/analysis_progress?id=' + currentAnalysisId)
        .then(resp => resp.json())
        .then(progressData => {
          analyseProgressBar.value = progressData.percent || 0;
          analyseProgressText.innerText = (progressData.percent || 0) + "% completed";
          if (["completed","error","aborted"].includes(progressData.status)) {
              clearInterval(analysisInterval);
              analyseSpinner.style.display = "none";
              enableAllProcessButtons();

              analysisResultsContainer.style.display = "block";

              if (progressData.status === "completed") {
                const folderName = progressData.folder 
                                    || document.getElementById("pdf-directory").value;
                analysisResultsContainer.innerText =
                    `Data analysis completed. Results were saved in folder "${folderName}".`;
              } else {
                analysisResultsContainer.innerText =
                    `Error during data analysis: ${progressData.error || "Unknown"}`;
              }
              resetAnalysisUI();
              currentAnalysisId = null;
          }
        })
        .catch(error => {
          clearInterval(analysisInterval);
          console.error("Error fetching analysis progress:", error);
          enableAllProcessButtons();
          resetAnalysisUI();
        });
      }, 1000);
    })
    .catch(error => {
      console.error("Error starting the analysis:", error);
      alert("Error starting the analysis.");
      enableAllProcessButtons();
      resetAnalysisUI();
    });
  });
});
"""

# ---------------------------
# Flask endpoints
# ---------------------------
@app.route("/")
def index():
    return Response(HTML_CONTENT, mimetype="text/html")

@app.route("/styles.css")
def styles():
    return Response(CSS_CONTENT, mimetype="text/css")

@app.route("/script.js")
def script():
    return Response(JS_CONTENT, mimetype="application/javascript")

@app.route("/download_papers", methods=["POST"])
def download_papers():
    global current_process
    data = request.get_json()
    database = data.get("database")
    query = data.get("query")
    year = data.get("year")
    num_papers = data.get("num_papers")

    if not database or not query or not year or not num_papers:
        return jsonify(
            {
                "error": "Please provide all parameters (database, search query, year, number)."
            }
        )

    with process_lock:
        if current_process.get("download"):
            return (
                jsonify({"error": "A download process is already running."}),
                400,
            )
        current_process["download"] = True

    download_id = str(uuid.uuid4())
    download_progress[download_id] = {"status": "starting", "percent": 0, "abort": False}

    thread = threading.Thread(
        target=download_papers_background,
        args=(download_id, database, query, year, num_papers),
    )
    thread.daemon = False
    thread.start()

    return jsonify({"download_id": download_id})

@app.route("/download_progress", methods=["GET"])
def get_download_progress():
    download_id = request.args.get("id")
    if not download_id or download_id not in download_progress:
        return jsonify({"error": "Invalid download ID."})
    return jsonify(download_progress[download_id])

@app.route("/abort_download", methods=["POST"])
def abort_download():
    data = request.get_json()
    download_id = data.get("download_id")
    if download_id in download_progress:
        download_progress[download_id]["abort"] = True
        logging.info(f"[ABORT_DOWNLOAD] Abort flag set for download ID {download_id}.")
        return jsonify({"message": "Abort of data acquisition requested"})
    return jsonify({"error": "Invalid download ID"}), 400

@app.route("/abort_analysis", methods=["POST"])
def abort_analysis():
    data = request.get_json()
    analysis_id = data.get("analysis_id")
    if analysis_id in analysis_progress:
        analysis_progress[analysis_id]["abort"] = True
        logging.info(f"[ABORT_ANALYSIS] Abort flag set for {analysis_id}")
        return jsonify({"message": "Abort of data analysis requested"})
    return jsonify({"error": "Invalid analysis ID"}), 400

@app.route("/abort_sediment", methods=["POST"])
def abort_sediment():
    data = request.get_json()
    analysis_id = data.get("analysis_id")
    if analysis_id in sediment_progress:
        sediment_progress[analysis_id]["abort"] = True
        logging.info(f"[ABORT_SEDIMENT] Abort flag set for sediment ID {analysis_id}.")
        with process_lock:
            if current_process.get("sediment"):
                current_process["sediment"] = False
        return jsonify({"message": "Abort of data evaluation requested"})
    return jsonify({"error": "Invalid analysis ID"}), 400

@app.route("/list_directories", methods=["GET"])
def list_directories():
    dirs = [
        d
        for d in os.listdir(DOWNLOAD_ROOT)
        if os.path.isdir(os.path.join(DOWNLOAD_ROOT, d))
    ]
    if not dirs:
        dirs = ["."]
    return jsonify({"directories": dirs})

@app.route("/start_analysis", methods=["POST"])
def start_analysis():
    global current_process
    data = request.get_json()
    pdf_directory = data.get("pdf_directory")
    questions = data.get("questions")

    logging.info(f"[START_ANALYSIS] Request received: pdf_directory={pdf_directory}, questions={(questions or '')[:100]}")

    if not pdf_directory or not questions:
        return jsonify({"error": "Please provide both a PDF directory and questions."})

    with process_lock:
        if current_process.get("analysis"):
            return (jsonify({"error": "An analysis process is already running."}), 400)
        current_process["analysis"] = True

    question_lines = [line.strip() for line in questions.splitlines() if line.strip()]
    expected_count = len(question_lines)

    if expected_count == 1:
        context_query = (
            "Answer the following question solely based on the provided excerpt. "
            "If the excerpt does not contain sufficient information OR no literal quote is possible, output exactly 'NO ANSWER!'. "
            "Then output 'Citation:' with a short literal quote (max 20 words) from the excerpt.\n"
            f"Question 1: {question_lines[0]}"
        )
    else:
        context_query = (
            "Answer the following questions solely based on the provided excerpt. "
            "For each question: exactly ONE sentence OR exactly 'NO ANSWER!'. "
            "If no literal quote is possible, output 'NO ANSWER!'. "
            "After each answer: 'Citation:' with a short literal quote (max 20 words) from the excerpt.\n"
        )
        for idx, q in enumerate(question_lines, start=1):
            context_query += f"Question {idx}: {q}\n"

    analysis_id = str(uuid.uuid4())
    analysis_progress[analysis_id] = {"status": "starting", "percent": 0}

    pdf_dir_name = os.path.basename(pdf_directory)
    current_date = datetime.now().strftime("%Y-%m-%d")
    analysis_output_folder = os.path.join(DATA_ROOT, f"{pdf_dir_name}_{current_date}")
    os.makedirs(analysis_output_folder, exist_ok=True)

    questions_file = os.path.join(analysis_output_folder, "questions.txt")
    with open(questions_file, "w", encoding="utf-8") as f:
        for q in question_lines:
            f.write(q + "\n")

    thread = threading.Thread(
        target=run_analysis,
        args=(
            analysis_id,
            os.path.join(DOWNLOAD_ROOT, pdf_directory),
            context_query,
            expected_count,
            question_lines,  
        ),
    )
    thread.daemon = False
    thread.start()
    return jsonify({"analysis_id": analysis_id})

@app.route("/analysis_progress", methods=["GET"])
def get_analysis_progress():
    analysis_id = request.args.get("id")
    if not analysis_id or analysis_id not in analysis_progress:
        return jsonify({"error": "Invalid analysis ID."})
    return jsonify(analysis_progress[analysis_id])

# ---------------------------
# Start application
# ---------------------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)

