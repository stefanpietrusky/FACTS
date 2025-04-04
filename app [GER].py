"""
title: FACTS V2 [FILTERING AND ANALYSIS OF CONTENT IN TEXTUAL SOURCES]
author: stefanpietrusky
author_url1: https://downchurch.studio/
author_url2: https://urlz.fr/uj1I [CAEDHET/HCDH Heidelberg University]
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
from datetime import datetime

import fitz  

from flask import Flask, request, jsonify, Response

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.firefox.service import Service
from selenium.webdriver.firefox.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from bs4 import BeautifulSoup

import pandas as pd
import nltk
from nltk.corpus import stopwords
from gensim.models import LdaModel, CoherenceModel
from gensim.corpora import Dictionary
from wordcloud import WordCloud
import pyLDAvis
import pyLDAvis.gensim_models
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import matplotlib
import matplotlib.pyplot as plt
import math

matplotlib.use('Agg')

# ---------------------------
# Logging & Konfiguration
# ---------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app = Flask(__name__)
start_time = datetime.now()

# Globales Dictionary 
download_progress = {}
analysis_progress = {}
sediment_progress = {}

# Basisordner für Downloads (Datenbeschaffung)
DOWNLOAD_ROOT = os.path.join(os.getcwd(), "downloads")
if not os.path.exists(DOWNLOAD_ROOT):
    os.makedirs(DOWNLOAD_ROOT)

# Basisordner für Analyseergebnisse (Datenanalyse)
DATA_ROOT = os.path.join(os.getcwd(), "data")
if not os.path.exists(DATA_ROOT):
    os.makedirs(DATA_ROOT)

# Basisordner für Analyseergebnisse (Datenanalyse)
ANALYSIS_ROOT = os.path.join(os.getcwd(), "analysis")
if not os.path.exists(ANALYSIS_ROOT):
    os.makedirs(ANALYSIS_ROOT)    

# ---------------------------
# Prozessexklusivität
# ---------------------------
current_process = None
process_lock = threading.Lock()

# ---------------------------
# Konstanten 
# ---------------------------
GECKO_DRIVER_PATH = r'\geckodriver.exe'
PDF_BASE_URL = "https://files.eric.ed.gov/fulltext/"

# ---------------------------
# Funktionen: Datenbeschaffung (Code 1)
# ---------------------------
def generate_apa_citation(metadata, database):
    authors = metadata.get("authors", [])
    if isinstance(authors, list):
        author_str = ", ".join(authors) if authors else "Unbekannter Autor"
    else:
        author_str = authors or "Unbekannter Autor"
    year = metadata.get("year", "o.J.")
    title = metadata.get("title", "Kein Titel")
    citation = f"{author_str} ({year}). {title}. {database}. Abgerufen von {metadata.get('url', 'n/a')}"
    return citation

def download_pdf_eric(paper_id, download_folder, session=None, file_name=None):
    pdf_url = f"{PDF_BASE_URL}{paper_id}.pdf"
    pdf_filename = os.path.join(download_folder, file_name) if file_name else os.path.join(download_folder, f"{paper_id}.pdf")
    session = session or requests.Session()
    
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/85.0.4183.121 Safari/537.36")
    }
    
    try:
        logging.info(f"⬇ Lade PDF herunter: {pdf_url}")
        response = session.get(pdf_url, timeout=30, headers=headers)
        response.raise_for_status()  
        with open(pdf_filename, "wb") as f:
            f.write(response.content)
        logging.info(f"PDF gespeichert: {pdf_filename}")
        return pdf_url
    except Exception as e:
        logging.error(f"PDF Download Fehler: {e}")
        return None

def extract_paper_data_eric(paper_div, base_url):
    paper_id = paper_div.get("id", "")
    paper_id = re.sub(r"^r_", "", paper_id)
    if not paper_id:
        logging.warning("Keine gültige Paper-ID gefunden, überspringe.")
        return None

    title_elem = paper_div.find('div', class_='r_t').find('a')
    title = title_elem.text.strip() if title_elem else "Unbekannter Titel"
    paper_url = base_url + title_elem['href'] if title_elem else None

    metadata_elem = paper_div.find('div', class_='r_a')
    metadata_text = metadata_elem.text.strip() if metadata_elem else "Unbekannte Quelle"
    if ',' in metadata_text:
        author_journal, pub_year = metadata_text.rsplit(',', 1)
        pub_year = pub_year.strip()
    else:
        author_journal, pub_year = metadata_text, "o.J."

    return {
        "paper_id": paper_id,
        "title": title,
        "paper_url": paper_url,
        "author_journal": author_journal,
        "pub_year": pub_year
    }

def download_eric_selenium(query, target_year, num_papers, download_folder, progress_data):
    options = Options()
    options.headless = True
    options.add_argument('--headless')
    options.set_preference("permissions.default.image", 2)
    service = Service(executable_path=GECKO_DRIVER_PATH)
    driver = webdriver.Firefox(service=service, options=options)
    session = requests.Session()
    
    try:
        driver.get("https://eric.ed.gov/")
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "s")))
        
        if progress_data.get("abort"):
            progress_data["status"] = "aborted"
            driver.quit()
            return

        try:
            cookie_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Accept')]"))
            )
            cookie_button.click()
        except TimeoutException:
            logging.info("Kein Cookie-Pop-up gefunden.")

        search_box = driver.find_element(By.ID, "s")
        search_box.clear()
        search_box.send_keys(query)
        
        for name, desc in [("pr", "Peer-Review"), ("ft", "Full-Text verfügbar")]:
            if progress_data.get("abort"):
                progress_data["status"] = "aborted"
                driver.quit()
                return
            try:
                checkbox = driver.find_element(By.NAME, name)
                if not checkbox.is_selected():
                    checkbox.click()
            except NoSuchElementException:
                logging.warning(f" {desc} Checkbox nicht gefunden!")
        
        search_box.send_keys(Keys.RETURN)
        WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "r_i")))
        
        downloaded_papers = 0
        os.makedirs(download_folder, exist_ok=True)
        references_file = os.path.join(download_folder, "quellenangaben.txt")
        
        base_url = "https://eric.ed.gov/"
        eric_index = 1  
        
        with open(references_file, 'a', encoding='utf-8') as ref_file:
            while downloaded_papers < num_papers:
                if progress_data.get("abort"):
                    progress_data["status"] = "aborted"
                    driver.quit()
                    return

                soup = BeautifulSoup(driver.page_source, 'html.parser')
                paper_divs = soup.find_all('div', class_='r_i')
                if not paper_divs:
                    logging.error("Keine weiteren Papers gefunden!")
                    break

                for paper in paper_divs:
                    if progress_data.get("abort"):
                        progress_data["status"] = "aborted"
                        driver.quit()
                        return

                    if downloaded_papers >= num_papers:
                        break

                    data = extract_paper_data_eric(paper, base_url)
                    if not data:
                        continue
                    if data["pub_year"] != str(target_year):
                        logging.info(f" {data['title']} wird übersprungen (Jahr {data['pub_year']}, erwartet: {target_year})")
                        continue

                    pdf_url = download_pdf_eric(
                        data["paper_id"],
                        download_folder,
                        session,
                        file_name=f"eric_{eric_index}.pdf"
                    )
                    metadata = {
                        "title": data["title"],
                        "authors": data["author_journal"],
                        "year": data["pub_year"],
                        "url": pdf_url
                    }
                    citation = generate_apa_citation(metadata, "ERIC")
                    ref_file.write(f"{eric_index}. {citation}\n\n")
                    
                    downloaded_papers += 1
                    progress_data["completed"] = downloaded_papers
                    progress_data["percent"] = int((downloaded_papers / num_papers) * 100)
                    eric_index += 1

                if progress_data.get("abort"):
                    progress_data["status"] = "aborted"
                    driver.quit()
                    return

                try:
                    next_page_link = WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable((By.XPATH, "//a[contains(text(), 'Next Page')]"))
                    )
                    next_page_link.click()
                    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "r_i")))
                except TimeoutException:
                    logging.error("Keine weitere Seite gefunden.")
                    break

        logging.info("ERIC-Scraping abgeschlossen.")
        progress_data["status"] = "completed"
    
    except Exception as e:
        progress_data["status"] = "error"
        progress_data["error"] = str(e)
    finally:
        driver.quit()

def download_pedocs(query, year, num_papers, download_folder, progress_data):
    gecko_driver_path = GECKO_DRIVER_PATH  
    options = Options()
    options.headless = True
    options.add_argument('--headless')
    options.set_preference("permissions.default.image", 2)  

    service = Service(executable_path=gecko_driver_path)
    driver = webdriver.Firefox(service=service, options=options)

    url = "https://www.pedocs.de"
    driver.get(url)
    
    try:
        search_box = WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.ID, 'volltextsuche'))
        )
        search_box.clear()
        search_box.send_keys(query)
        search_box.send_keys(Keys.RETURN)
        
        WebDriverWait(driver, 10).until(
            lambda d: d.find_elements(By.XPATH, "//a[contains(@href, 'frontdoor.php')]")
        )

        if not os.path.exists(download_folder):
            os.makedirs(download_folder)
        references_file = os.path.join(download_folder, "quellenangaben.txt")

        paper_index = 1  
        total_downloaded = 0

        while total_downloaded < num_papers:
            if progress_data.get("abort"):
                progress_data["status"] = "aborted"
                driver.quit()
                return

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            paper_links = [link['href'] for link in soup.find_all('a', href=True) if 'frontdoor.php' in link['href']]
            base_url = "https://www.pedocs.de/"

            for paper_link in paper_links:
                if progress_data.get("abort"):
                    progress_data["status"] = "aborted"
                    driver.quit()
                    return

                if total_downloaded >= num_papers:
                    break

                paper_url = base_url + paper_link
                logging.info(f"Visiting: {paper_url}")
                paper_response = requests.get(paper_url)
                paper_soup = BeautifulSoup(paper_response.content, 'html.parser')

                year_element = paper_soup.find('td', itemprop='datePublished')
                if year_element:
                    publication_year = year_element.text.strip()
                    if publication_year == str(year):
                        title_elem = paper_soup.find('h1') or paper_soup.find('title')
                        title = title_elem.text.strip() if title_elem else "Kein Titel"

                        pdf_link = paper_soup.find('a', class_='a5-book-list-item-fulltext')
                        if pdf_link:
                            pdf_url = pdf_link['href']
                            if pdf_url.startswith('//'):
                                pdf_url = 'https:' + pdf_url
                            pdf_filename = os.path.join(download_folder, f"pedocs_{paper_index}.pdf")
                            try:
                                r = requests.get(pdf_url, stream=True, timeout=30)
                                r.raise_for_status()
                                with open(pdf_filename, "wb") as f:
                                    for chunk in r.iter_content(chunk_size=8192):
                                        f.write(chunk)
                            except Exception as e:
                                logging.error(f"PDF Download Fehler: {e}")
                                continue

                            reference_row = paper_soup.find('th', scope="row", string="Quellenangabe")
                            if reference_row:
                                reference = reference_row.find_next('td').text.strip()
                                with open(references_file, 'a', encoding='utf-8') as f:
                                    f.write(f"{paper_index}. {reference}\n\n")
                                logging.info(f"Quellenangabe für Paper {paper_index} gespeichert.")
                            else:
                                logging.info(f"Keine Quellenangabe gefunden für Paper {paper_index}")

                            total_downloaded += 1
                            progress_data["completed"] = total_downloaded
                            progress_data["percent"] = int((total_downloaded / num_papers) * 100)
                            paper_index += 1
                        else:
                            logging.info(f"Kein PDF-Download-Link gefunden für {paper_url}")
                    else:
                        logging.info(f"Artikel aus dem Jahr {publication_year} wird übersprungen.")
                else:
                    logging.info(f"Kein Erscheinungsjahr gefunden für {paper_url}")

            if total_downloaded >= num_papers:
                break

            if progress_data.get("abort"):
                progress_data["status"] = "aborted"
                driver.quit()
                return

            try:
                next_button = WebDriverWait(driver, 10).until(
                    EC.element_to_be_clickable((By.XPATH, "//a[@class='svg-wrapper a5-svg-hover' and (@aria-label='Weiter' or @title='Weiter')]"))
                )
                next_button.click()
                WebDriverWait(driver, 10).until(
                    lambda d: d.find_elements(By.XPATH, "//a[contains(@href, 'frontdoor.php')]")
                )
            except Exception:
                logging.info("Keine weiteren Seiten gefunden, Download abgeschlossen.")
                break

        progress_data["status"] = "completed"

    except Exception as e:
        progress_data["status"] = "error"
        progress_data["error"] = str(e)
    finally:
        driver.quit()

def download_arxiv(query, year, num_papers, download_folder, progress_data):
    base_url = "http://export.arxiv.org/api/query?"
    search_query = f"all:{query} AND submittedDate:[{year}01010000 TO {year}12312359]"
    params = {
        "search_query": search_query,
        "start": 0,
        "max_results": num_papers,
        "sortBy": "submittedDate",
        "sortOrder": "descending"
    }
    query_url = base_url + "&".join([f"{key}={value}" for key, value in params.items()])
    
    if progress_data.get("abort"):
        progress_data["status"] = "aborted"
        return

    try:
        response = requests.get(query_url, timeout=30)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        entries = root.findall("atom:entry", ns)

        total = min(num_papers, len(entries))
        progress_data["total"] = total
        progress_data["completed"] = 0
        citations = []
        paper_index = 1

        if progress_data.get("abort"):
            progress_data["status"] = "aborted"
            return

        for idx, entry in enumerate(entries[:total], start=1):
            if progress_data.get("abort"):
                progress_data["status"] = "aborted"
                return

            title_elem = entry.find("atom:title", ns)
            title = title_elem.text.strip() if title_elem is not None else "Kein Titel"
            published_elem = entry.find("atom:published", ns)
            published = published_elem.text.strip() if published_elem is not None else "0000-00-00T00:00:00Z"
            published_year = published[:4]
            authors = [author.find("atom:name", ns).text for author in entry.findall("atom:author", ns)]
            
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
                    with open(pdf_filename, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                except Exception as e:
                    logging.error(f"PDF Download Fehler: {e}")

            metadata = {
                "title": title,
                "authors": authors,
                "year": published_year,
                "url": pdf_url if pdf_url else entry.find("atom:id", ns).text
            }
            citation = generate_apa_citation(metadata, "arxiv")
            citations.append(citation)

            progress_data["completed"] = idx
            progress_data["percent"] = int((idx / total) * 100)
            paper_index += 1

        progress_data["status"] = "completed"

        if progress_data.get("abort"):
            progress_data["status"] = "aborted"
            return

        citations_file = os.path.join(download_folder, "quellenangaben.txt")
        with open(citations_file, "a", encoding='utf-8') as f:
            for i, citation in enumerate(citations, start=1):
                f.write(f"{i}. {citation}\n\n")

    except Exception as e:
        progress_data["status"] = "error"
        progress_data["error"] = str(e)

def download_papers_background(download_id, database, query, year, num_papers):
    global current_process
    progress_data = download_progress.get(download_id, {})
    progress_data["status"] = "running"
    progress_data["abort"] = False
    download_progress[download_id] = progress_data

    db_folder = os.path.join(DOWNLOAD_ROOT, f"{database}_{re.sub(r'[^A-Za-z0-9]+', '_', query)}_{year}")

    try:
        if database == "pedocs":
            download_pedocs(query, year, num_papers, db_folder, progress_data)
        elif database == "arxiv":
            download_arxiv(query, year, num_papers, db_folder, progress_data)
        elif database == "eric":
            download_eric_selenium(query, year, num_papers, db_folder, progress_data)
        else:
            progress_data["status"] = "error"
            progress_data["error"] = "Unbekannte Datenbank."
    except Exception as e:
        progress_data["status"] = "error"
        progress_data["error"] = str(e)
    finally:
        with process_lock:
            current_process = None

# ---------------------------
# Funktionen: Datenanalyse (Code 2)
# ---------------------------
@app.route('/upload_pdfs', methods=['POST'])
def upload_pdfs():
    if 'files' not in request.files:
        return jsonify({"error": "Keine Dateien gefunden."}), 400
    files = request.files.getlist('files')
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    upload_folder = os.path.join(DOWNLOAD_ROOT, f"upload_{timestamp}")
    os.makedirs(upload_folder, exist_ok=True)
    saved_files = []
    for file in files:
        if file.filename.endswith('.pdf'):
            file_path = os.path.join(upload_folder, file.filename)
            file.save(file_path)
            saved_files.append(file.filename)
    return jsonify({"message": "Dateien hochgeladen", "folder": os.path.basename(upload_folder), "files": saved_files})

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    page_count = doc.page_count  
    for page_num in range(page_count):
        page = doc.load_page(page_num)
        text += page.get_text()
    doc.close()  
    return text, page_count

def clean_text(text):
    text = re.sub(r'\nPage \d+\n|\n\d+\n', '', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\d+\.', '', text)
    return text

def save_text_to_file(text, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)

def analysis_background(analysis_id, pdf_directory, context_query, expected_count):
    progress_data = analysis_progress[analysis_id]
    progress_data["status"] = "running"
    progress_data["percent"] = 0
    for i in range(1, 101):
        time.sleep(0.1) 
        progress_data["percent"] = i
    progress_data["status"] = "completed"

def query_llm_via_cli(input_text):
    try:
        process = subprocess.Popen(
            ["ollama", "run", "llama3.1p"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='ignore',
            bufsize=1
        )
        stdout, stderr = process.communicate(input=f"{input_text}\n", timeout=180)
        if process.returncode != 0:
            logging.error(f"LLM-Fehler: {stderr.strip()}")
            return ""
        response = re.sub(r'\x1b\[.*?m', '', stdout)
        logging.info(f"LLM-Rohantwort: {response.strip()}")
        return response.strip()
    except subprocess.TimeoutExpired:
        process.kill()
        return "Timeout for the model request"
    except Exception as e:
        return f"An unexpected error has occurred: {str(e)}"

def validate_llm_response(response, expected_count):
    response = re.sub(r'(?i)^Hier\s+sind\s+die\s+Antworten\s+auf\s+die\s+Fragen:\s*[\n\r]*', '', response)
    response = re.sub(r'(?im)^Frage\s+\d+:\s*', '', response)
    
    if not response.strip():
        return "\n".join(f"{i+1}. KEINE ANTWORT!" for i in range(expected_count))
    
    if expected_count == 1:
        answer = response.strip()
        if len(answer.split()) < 3:
            return "1. KEINE ANTWORT!"
        else:
            return "1. " + answer  
    else:
        lines = [re.sub(r'^\d+\.\s*', '', line.strip()) for line in response.splitlines() if line.strip()]
        invalid_keywords = [
            "keine antwort", "keine antwert", "nein", "irrelevant", "nicht angegeben", 
            "keine relevanz", "keine daten", "nicht explizit erwähnt", "ich kann keine", "ich kann nur"
        ]
        validated_answers = []
        for i in range(expected_count):
            if i < len(lines):
                answer = lines[i]
                if any(keyword in answer.lower() for keyword in invalid_keywords) or len(answer.split()) <= 2:
                    validated_answers.append("KEINE ANTWORT!")
                else:
                    validated_answers.append(answer)
            else:
                validated_answers.append("KEINE ANTWORT!")
        return "\n".join(f"{i+1}. {ans}" for i, ans in enumerate(validated_answers))

def split_text_by_sentences(text, chunk_size=4000):
    sentences = re.split(r'(?<=[.!?]) +', text) 
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

def analyze_long_text_in_chunks_and_save(cleaned_text, context_query, output_file, expected_count):
    system_message = ("System: Du bist ein intelligenter Assistent. Bitte antworte ausschließlich "
                      "prägnant und korrekt. Verwende keine Einleitungen, Meta-Antworten, "
                      "Nummerierungen oder Wiederholungen der Frage in deiner Antwort.")
    full_prompt_header = f"{system_message}\n\n{context_query}"
    
    chunks = split_text_by_sentences(cleaned_text, chunk_size=4000)
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                result_text = "\n\nErgebnis für Abschnitt {}:\n{}".format(
                    i+1, "\n".join(f"{j+1}. KEINE ANTWORT!" for j in range(expected_count))
                )
                f.write(result_text)
                continue

            logging.info(f"Fortschritt: Abschnitt {i+1}/{len(chunks)} ({(i+1)/len(chunks)*100:.2f}%)")
            prompt = f"{full_prompt_header}\n\nTextabschnitt {i+1}:\n{chunk}"
            logging.info(f"Sende Abschnitt {i+1}/{len(chunks)} zur Analyse...")
            analysis_result = query_llm_via_cli(prompt)
            validated_result = validate_llm_response(analysis_result, expected_count)
            result_text = f"\n\nErgebnis für Abschnitt {i+1}:\n{validated_result}\n"
            f.write(result_text)
    return output_file

def run_analysis(analysis_id, pdf_directory, context_query, expected_count):
    global current_process
    try:
        pdf_files = [f for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
        total_files = len(pdf_files)
        
        pdf_dir_name = os.path.basename(pdf_directory)
        current_date = datetime.now().strftime("%Y-%m-%d")
        analysis_output_folder = os.path.join(DATA_ROOT, f"{pdf_dir_name}_{current_date}")
        if not os.path.exists(analysis_output_folder):
            os.makedirs(analysis_output_folder)
        
        results_summary = []
        for idx, filename in enumerate(pdf_files):
            pdf_path = os.path.join(pdf_directory, filename)
            logging.info(f"Verarbeite {filename} ...")
            
            extracted_text, page_count = extract_text_from_pdf(pdf_path)
            cleaned = clean_text(extracted_text)
            
            analysis_output_path = os.path.join(analysis_output_folder, f"analyseergebnis_paper{idx+1}.txt")
            analyze_long_text_in_chunks_and_save(cleaned, context_query, analysis_output_path, expected_count)
            
            results_summary.append(f"Analyse für {filename} abgeschlossen (Ergebnis in {analysis_output_path}).")
            
            progress = int(((idx + 1) / total_files) * 100)
            analysis_progress[analysis_id]["percent"] = progress
            logging.info(f"Analyse-Fortschritt: {progress}%")
        
        analysis_progress[analysis_id]["status"] = "completed"
        analysis_progress[analysis_id]["result"] = "\n".join(results_summary)
    finally:
        with process_lock:
            current_process = None

# ---------------------------
# Funktionen: Datenauswertung (Code 3)
# ---------------------------
def get_stopwords():
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download('stopwords')
    return nltk.corpus.stopwords.words('german')

german_stopwords = get_stopwords()
custom_stopwords = set(german_stopwords).union({
    'sowie', 'geht', 'gut', 'eignet', 'ganzes', 'enge', 'ganzes.', 'dafür', 'eng',
    'etwa', 'deren', 'wären', 'se', 'an.', 'per', 'kommt', '(wie', 'werden', 'drei',
    'haben', 'u.a.'
})

def tokenize_texts(text_data):
    tokenized_texts = [
        [word for word in text.lower().split() if word not in custom_stopwords]
        for text in text_data
    ]
    return tokenized_texts

def find_optimal_num_topics(texts, min_topics=2, max_topics=15, step=1, alpha_value='auto', beta_value='auto'):
    tokenized_texts = tokenize_texts(texts)
    dictionary = Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]

    coherence_scores = []
    for num_topics in range(min_topics, max_topics + 1, step):
        lda_model = LdaModel(
            corpus=corpus, 
            id2word=dictionary, 
            num_topics=num_topics, 
            random_state=42, 
            iterations=100,
            passes=10,
            alpha=alpha_value,  
            eta=beta_value  
        )
        coherence_model = CoherenceModel(model=lda_model, texts=tokenized_texts, dictionary=dictionary, coherence='c_v')
        coherence_score = coherence_model.get_coherence()
        coherence_scores.append(coherence_score)
        print(f"Anzahl Topics: {num_topics} -- Coherence: {coherence_score:.4f}")

    optimal_topics = range(min_topics, max_topics + 1, step)[np.argmax(coherence_scores)]
    print(f"Optimale Anzahl an Topics: {optimal_topics}")
    return optimal_topics

def perform_lda(text_data, num_topics, alpha_value='auto', beta_value='auto'):
    tokenized_texts = tokenize_texts(text_data)
    dictionary = Dictionary(tokenized_texts)
    corpus = [dictionary.doc2bow(text) for text in tokenized_texts]
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42,
                         iterations=100, passes=10, alpha=alpha_value, eta=beta_value)
    return lda_model, dictionary, corpus, tokenized_texts

def visualize_lda(lda_model, dictionary, corpus, question_id, p_w=None, output_directory=None):
    if output_directory is None:
        output_directory = os.getcwd()
    try:
        lda_vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
        output_html_path = os.path.join(output_directory, f'lda_visualization_question_{question_id}.html')
        pyLDAvis.save_html(lda_vis, output_html_path)
        print(f"Visualisierung für Frage {question_id} als HTML gespeichert: {output_html_path}")
    except Exception as e:
        print(f"Fehler bei der Visualisierung für Frage {question_id}: {e}")

def get_top_terms_for_topic(lda_model, topic_id, dictionary, p_w, lambda_val=1.0, topn=30):
    topic_terms = lda_model.get_topic_terms(topic_id, topn=len(dictionary))
    relevance_scores = []
    for word_id, prob in topic_terms:
        pw = p_w.get(dictionary[word_id], 1e-12)
        prob = max(prob, 1e-12)
        pw = max(pw, 1e-12)
        relevance = lambda_val * prob + (1 - lambda_val) * (math.log(prob) - math.log(pw))
        relevance_scores.append((dictionary[word_id], relevance))
    relevance_scores.sort(key=lambda x: x[1], reverse=True)
    top_terms = [word for word, score in relevance_scores[:topn]]
    return top_terms

def plot_wordclouds(lda_model, dictionary, lambda_val=1.0, num_words=30, p_w=None, output_directory=None):
    if output_directory is None:
        output_directory = os.getcwd()
    if p_w is None:
        total_count = sum(dictionary.cfs.values())
        p_w = {dictionary[token_id]: count / total_count for token_id, count in dictionary.cfs.items()}
    num_topics = lda_model.num_topics
    for topic_idx in range(num_topics):
        top_terms = get_top_terms_for_topic(lda_model, topic_idx, dictionary, p_w, lambda_val=lambda_val, topn=num_words)
        topic_terms = dict(lda_model.get_topic_terms(topic_idx, topn=num_words))
        word_freq = {}
        for word in top_terms:
            word_id = dictionary.token2id.get(word)
            if word_id is not None:
                word_freq[word] = topic_terms.get(word_id, 1e-12)
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freq)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.title(f"Wordcloud für Topic {topic_idx + 1} (λ={lambda_val})")
        plt.axis('off')
        output_path = os.path.join(output_directory, f"wordcloud_topic_{topic_idx + 1}_lambda_{lambda_val}.png")
        plt.savefig(output_path)
        plt.close()
        print(f"Wordcloud für Topic {topic_idx + 1} (λ={lambda_val}) gespeichert: {output_path}")

def create_knowledge_graph(lda_model, dictionary, corpus, sorted_topic_indices, num_topics,
                           lambda_val=1.0, p_w=None, output_html="knowledge_graph.html",
                           prob_threshold=0.001):
    if output_html is None:
        output_html = os.path.join(ANALYSIS_ROOT, "knowledge_graph.html")
    if p_w is None:
        total_count = sum(dictionary.cfs.values())
        p_w = {dictionary[token_id]: count / total_count for token_id, count in dictionary.cfs.items()}
    
    G = nx.Graph()

    for sorted_idx, topic_idx in enumerate(sorted_topic_indices[:num_topics]):
        topic_name = f"Topic {sorted_idx + 1}"
        G.add_node(topic_name, label=topic_name, color="#00B0F0", size=30, is_topic=True)
        top_words = get_top_terms_for_topic(lda_model, topic_idx, dictionary, p_w, lambda_val=lambda_val, topn=30)
        for word in top_words:
            word_id = dictionary.token2id[word]
            word_prob = dict(lda_model.get_topic_terms(topic_idx, topn=30)).get(word_id, 0.0)
            if word_prob < prob_threshold:
                continue
            if not G.has_node(word):
                node_size = 15 + (word_prob * 100)
                G.add_node(word, label=word, color="#00CC99", size=node_size, is_topic=False)
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
            showlegend=False
        )
        edge_traces.append(edge_trace)

    topic_x, topic_y, topic_text = [], [], []
    word_1_x, word_1_y, word_1_text, word_1_hover = [], [], [], []
    word_2_x, word_2_y, word_2_text, word_2_hover = [], [], [], []
    word_3_x, word_3_y, word_3_text, word_3_hover = [], [], [], []
    word_4_x, word_4_y, word_4_text, word_4_hover = [], [], [], []

    def make_hover_label(node_name, node_data):
        connections = len(list(G.neighbors(node_name)))
        total_weight = sum([G.edges[(node_name, nbr)]["weight"] for nbr in G.neighbors(node_name)])
        connected_topics = [nbr for nbr in G.neighbors(node_name) if nbr.startswith("Topic")]
        main_topic = node_data.get("topic", "Kein Haupttopic")
        other_topics = [t for t in connected_topics if t != main_topic]
        return (
            f"Label: {node_data['label']}<br>"
            f"Verbindungen: {connections}<br>"
            f"Gesamtgewichtung: {total_weight:.6f}<br>"
            f"Haupttopic: {main_topic}<br>"
            f"Nebentopics: {', '.join(other_topics) if other_topics else 'Keine'}"
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
                word_1_x.append(x); word_1_y.append(y); word_1_text.append(data["label"]); word_1_hover.append(hover_text)
            elif connections == 2:
                word_2_x.append(x); word_2_y.append(y); word_2_text.append(data["label"]); word_2_hover.append(hover_text)
            elif connections == 3:
                word_3_x.append(x); word_3_y.append(y); word_3_text.append(data["label"]); word_3_hover.append(hover_text)
            else:
                word_4_x.append(x); word_4_y.append(y); word_4_text.append(data["label"]); word_4_hover.append(hover_text)

    topic_trace = go.Scatter(
        x=topic_x, y=topic_y, mode="markers+text", text=topic_text,
        textposition="top center", hoverinfo="text", name="Topics",
        marker=dict(color="#00B0F0", size=30, line=dict(width=2, color="#262626"))
    )
    word_1_trace = go.Scatter(
        x=word_1_x, y=word_1_y, mode="markers+text", text=word_1_text,
        textposition="top center", hovertext=word_1_hover, hoverinfo="text",
        name="Begriffe (1 Verbindung)", marker=dict(color="#00CC99", size=15, line=dict(width=2, color="#262626"))
    )
    word_2_trace = go.Scatter(
        x=word_2_x, y=word_2_y, mode="markers+text", text=word_2_text,
        textposition="top center", hovertext=word_2_hover, hoverinfo="text",
        name="Begriffe (2 Verbindungen)", marker=dict(color="#D86ECC", size=15, line=dict(width=2, color="#262626"))
    )
    word_3_trace = go.Scatter(
        x=word_3_x, y=word_3_y, mode="markers+text", text=word_3_text,
        textposition="top center", hovertext=word_3_hover, hoverinfo="text",
        name="Begriffe (3 Verbindungen)", marker=dict(color="#FFC300", size=15, line=dict(width=2, color="#262626"))
    )
    word_4_trace = go.Scatter(
        x=word_4_x, y=word_4_y, mode="markers+text", text=word_4_text,
        textposition="top center", hovertext=word_4_hover, hoverinfo="text",
        name="Begriffe (>3 Verbindungen)", marker=dict(color="#FF5733", size=15, line=dict(width=2, color="#262626"))
    )

    fig = go.Figure(
        data=edge_traces + [topic_trace, word_1_trace, word_2_trace, word_3_trace, word_4_trace],
        layout=go.Layout(
            title=f"Interaktiver Wissensgraph (λ={lambda_val})",
            titlefont_size=16, showlegend=True, hovermode="closest", clickmode="event+select",
            margin=dict(b=0, l=0, r=0, t=40), xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False), plot_bgcolor="white"
        )
    )

    fig.write_html(output_html)
    print(f"Wissensgraph (λ={lambda_val}) als HTML gespeichert: {output_html}")
    
def get_combined_prompt(top_terms_freq, top_terms_excl, question_text):
    prompt = f"""
Für folgende Frage: '{question_text}' wurden nach der LDA-Themenmodellierung zwei Sätze von Top-Begriffen ermittelt:

- Basierend auf Häufigkeit (λ = 1): {', '.join(top_terms_freq)}
- Basierend auf Exklusivität (λ = 0): {', '.join(top_terms_excl)}

Bitte interpretiere diese beiden Sätze von Top-Begriffen nach folgendem Schema:

1) Einleitung: Kurze Einführung in das Thema.
2) Häufigste Begriffe (λ = 1): Erläutere, warum diese Begriffe besonders oft auftreten und welche Relevanz sie für die Beantwortung der Frage haben.
3) Exklusivste Begriffe (λ = 0): Erläutere, was diese besonders exklusiven Begriffe aussagen und wie sie das Thema ergänzen oder spezifizieren.
4) Fazit: Ziehe ein zusammenfassendes Resümee, wie die beiden Begriffssätze gemeinsam die Frage beantworten helfen.

Bitte gib deine Antwort in klarem Fließtext ohne Markdown, Sterne, Aufzählungszeichen oder andere Sonderformatierungen zurück.
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
            errors="replace"
        )
        stdout, stderr = process.communicate(input=input_text, timeout=180)
        if process.returncode != 0:
            print(f"Fehler bei der Modellanfrage: {stderr.strip()}")
            return "Interpretation nicht möglich"
        return stdout.strip()
    except Exception as e:
        print(f"Ein Fehler ist aufgetreten: {e}")
        return "Fehler beim Abrufen der Interpretation"

def save_interpretations_to_txt(interpretations, file_path):
    with open(file_path, "w", encoding="utf-8") as file:
        for interpretation in interpretations:
            top_terms_freq = interpretation.get("Top Terms (Häufigkeit, λ=1)", "")
            top_terms_excl = interpretation.get("Top Terms (Exklusivität, λ=0)", "")
            top_terms = f"Häufigkeit (λ=1): {top_terms_freq}\nExklusivität (λ=0): {top_terms_excl}"
            file.write(f"Thema {interpretation['Topic']}\n")
            file.write("=" * 30 + "\n")
            file.write(f"Begriffe:\n{top_terms}\n")
            file.write("\nInterpretation:\n")
            file.write(f"{interpretation['Interpretation']}\n")
            file.write("\n" + "-" * 50 + "\n")
    print(f"Interpretationen in '{file_path}' gespeichert.")

def lda_analysis_with_interpretation(file_path, question_id, num_topics, model="llama3.1p", output_directory=None):
    try:
        data = pd.read_csv(file_path, encoding='utf-8')
        text_data = data[f'Frage {question_id} Antworten'].dropna().tolist()
        if not text_data:
            print(f"Keine Daten für die LDA-Analyse von Frage {question_id}.")
            return

        optimal_topics = num_topics
        lda_model, dictionary, corpus, tokenized_texts = perform_lda(text_data, optimal_topics)
        total_count = sum(dictionary.cfs.values())
        p_w = {dictionary[token_id]: count / total_count for token_id, count in dictionary.cfs.items()}

        topic_totals = np.zeros(lda_model.num_topics)
        for doc in corpus:
            doc_topics = lda_model.get_document_topics(doc, minimum_probability=0)
            for topic_id, prob in doc_topics:
                topic_totals[topic_id] += prob
        sorted_topic_indices = np.argsort(topic_totals)[::-1]

        interpretations = []
        for cluster_idx, topic_idx in enumerate(sorted_topic_indices):
            top_terms_freq = get_top_terms_for_topic(lda_model, topic_idx, dictionary, p_w, lambda_val=1.0, topn=30)
            top_terms_excl = get_top_terms_for_topic(lda_model, topic_idx, dictionary, p_w, lambda_val=0.0, topn=30)

            combined_prompt = get_combined_prompt(top_terms_freq, top_terms_excl, f"Frage {question_id}")
            combined_output = query_ollama(combined_prompt, model)
            interpretation_sections = combined_output.split("\n", 1)
            interpretations.append({
                "Topic": cluster_idx + 1,
                "Top Terms (Häufigkeit, λ=1)": ', '.join(top_terms_freq),
                "Top Terms (Exklusivität, λ=0)": ', '.join(top_terms_excl),
                "Interpretation": "\n".join(interpretation_sections).strip()
            })

        if output_directory is None:
            output_directory = os.path.join(ANALYSIS_ROOT, "evaluation_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
            if not os.path.exists(output_directory):
                os.makedirs(output_directory)

        visualize_lda(lda_model, dictionary, corpus, question_id, p_w=p_w, output_directory=output_directory)
        plot_wordclouds(lda_model, dictionary, lambda_val=1.0, num_words=30, p_w=p_w, output_directory=output_directory)
        plot_wordclouds(lda_model, dictionary, lambda_val=0.0, num_words=30, p_w=p_w, output_directory=output_directory)

        create_knowledge_graph(
            lda_model=lda_model,
            dictionary=dictionary,
            corpus=corpus,
            sorted_topic_indices=sorted_topic_indices,
            num_topics=optimal_topics,
            lambda_val=1.0,
            p_w=p_w,
            output_html=os.path.join(output_directory, f"knowledge_graph_question_{question_id}_lambda_1.html")
        )
        create_knowledge_graph(
            lda_model=lda_model,
            dictionary=dictionary,
            corpus=corpus,
            sorted_topic_indices=sorted_topic_indices,
            num_topics=optimal_topics,
            lambda_val=0.0,
            p_w=p_w,
            output_html=os.path.join(output_directory, f"knowledge_graph_question_{question_id}_lambda_0.html")
        )

        output_txt_path = os.path.join(output_directory, f"interpretations_question_{question_id}.txt")
        save_interpretations_to_txt(interpretations, output_txt_path)

        print("LDA-Analyse abgeschlossen. Ergebnisse im Ordner:", output_directory)
    except Exception as e:
        print(f"Fehler in lda_analysis_with_interpretation für Frage {question_id}: {e}")

def load_text_files(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            with open(os.path.join(directory, filename), 'r', encoding='utf-8') as file:
                content = file.read().strip()
                if content:
                    data.append(content)
    return data

def categorize_responses_by_question(data):
    all_sections = {}
    for text in data:
        pattern_section = r'Ergebnis für Abschnitt \d+:\s*((?:(?:\d+\.\s*.*?(?=\n\d+\.|\Z)))+)'
        sections = re.findall(pattern_section, text, re.DOTALL)
        print(f"DEBUG: Sections gefunden: {sections}")
        for idx, sec in enumerate(sections, start=1):
            answers = re.findall(r'\d+\.\s*(.*?)(?=\n\d+\.|\Z)', sec, re.DOTALL)
            answers = [ans.strip() for ans in answers]
            if answers:
                all_sections.setdefault(idx, []).extend(answers)
    return all_sections

def sediment_analysis(directory):
    data = load_text_files(directory)
    if not data:
        print("Keine Texte gefunden")
        return

    def categorize_responses_by_question(data):
        all_sections = {}
        for text in data:
            pattern_section = r'Ergebnis für Abschnitt \d+:\s*((?:(?:\d+\.\s*.*?(?=\n\d+\.|\Z)))+)'
            sections = re.findall(pattern_section, text, re.DOTALL)
            print(f"DEBUG: Sections gefunden: {sections}")
            for idx, sec in enumerate(sections, start=1):
                answers = re.findall(r'\d+\.\s*(.*?)(?=\n\d+\.|\Z)', sec, re.DOTALL)
                answers = [ans.strip() for ans in answers if ans.strip()]
                if answers:
                    all_sections.setdefault(idx, []).extend(answers)
        return all_sections

    responses_by_question = categorize_responses_by_question(data)
    print("DEBUG: Antworten nach Frage:", responses_by_question)

    filtered_responses = {}
    for q_id, responses in responses_by_question.items():
        filtered = [r for r in responses if "keine antwort" not in r.lower()]
        if filtered:
            filtered_responses[q_id] = filtered

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    evaluation_folder = os.path.join(ANALYSIS_ROOT, "evaluation_" + timestamp)
    if not os.path.exists(evaluation_folder):
        os.makedirs(evaluation_folder)
    print("Evaluation-Ordner:", evaluation_folder)

    questions_file = os.path.join(directory, "questions.txt")
    if os.path.exists(questions_file):
        with open(questions_file, "r", encoding="utf-8") as f:
            questions = [line.strip() for line in f if line.strip()]
    else:
        questions = []
        print("Keine questions.txt gefunden.")

    for q_id, responses in filtered_responses.items():
        if q_id > len(questions):
            continue
        question_folder = os.path.join(evaluation_folder, f"question_{q_id}")
        if not os.path.exists(question_folder):
            os.makedirs(question_folder)
        question_text = questions[q_id - 1]
        question_text_file = os.path.join(question_folder, f"question_{q_id}.txt")
        with open(question_text_file, "w", encoding="utf-8") as f:
            f.write(question_text)

        df = pd.DataFrame({f"Frage {q_id} Antworten": responses})
        csv_path = os.path.join(question_folder, f"question_{q_id}_responses.csv")
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"Antworten für Frage {q_id} gespeichert in {csv_path}")
        
        optimal_topics = find_optimal_num_topics(responses, min_topics=2, max_topics=15, step=1)
        lda_analysis_with_interpretation(csv_path, q_id, num_topics=optimal_topics, model="llama3.1p", output_directory=question_folder)

    print("Datenauswertung abgeschlossen. Evaluation-Ergebnisse befinden sich im Ordner:", evaluation_folder)

# ---------------------------
# Endpunkte: Datenauswertung
# ---------------------------
@app.route('/list_data_directories', methods=['GET'])
def list_data_directories():
    dirs = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
    if not dirs:
        dirs = ["Keine Verzeichnisse gefunden"]
    return jsonify({"directories": dirs})

def sediment_analysis_background(analysis_id, data_directory):
    global current_process
    try:
        sediment_analysis(data_directory)
        sediment_progress[analysis_id]["status"] = "completed"
        sediment_progress[analysis_id]["percent"] = 100
    except Exception as e:
        sediment_progress[analysis_id]["status"] = "error"
        sediment_progress[analysis_id]["error"] = str(e)
    finally:
        with process_lock:
            current_process = None

@app.route('/start_sediment_analysis', methods=['POST'])
def start_sediment_analysis():
    global current_process
    data = request.get_json()
    selected_dir = data.get("data_directory")
    if not selected_dir:
        return jsonify({"error": "Bitte ein Verzeichnis im data-Ordner auswählen."})
    
    full_path = os.path.join(DATA_ROOT, selected_dir)
    if not os.path.exists(full_path):
        return jsonify({"error": "Das gewählte Verzeichnis existiert nicht."})
    
    with process_lock:
        if current_process is not None:
            return jsonify({"error": f"Ein Prozess läuft bereits: {current_process}"}), 400
        current_process = "sediment"
    
    analysis_id = str(uuid.uuid4())
    sediment_progress[analysis_id] = {"status": "running", "percent": 0}
    
    thread = threading.Thread(target=sediment_analysis_background, args=(analysis_id, full_path))
    thread.daemon = False
    thread.start()
    
    return jsonify({"analysis_id": analysis_id})

@app.route('/sediment_analysis_progress', methods=['GET'])
def get_sediment_analysis_progress():
    analysis_id = request.args.get("id")
    if not analysis_id or analysis_id not in sediment_progress:
        return jsonify({"error": "Ungültige Analysis-ID."})
    return jsonify(sediment_progress[analysis_id])
   
# ---------------------------
# Frontend-Inhalte
# ---------------------------
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="de">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>FACTS V2 - Datenbeschaffung und Analyse</title>
  <link rel="stylesheet" href="/styles.css">
</head>
<body>
  <div class="container">
    <h1>FACTS V2</h1>

    <div class="section">
      <h3>1: DATENBESCHAFFUNG</h3>
      <label for="database-select">Datenbank auswählen:</label>
      <select id="database-select">
        <option value="pedocs">peDOCs</option>
        <option value="arxiv">Arxiv</option>
        <option value="eric">ERIC</option>
      </select>
      <label for="search-query">Suchbegriffe:</label>
      <input type="text" id="search-query" placeholder="z.B. maschinelles Lernen">
      <label for="publication-year">Veröffentlichungsjahr:</label>
      <input type="text" id="publication-year" placeholder="z.B. 2023">
      <label for="num-papers">Anzahl der Paper:</label>
      <input type="number" id="num-papers" min="1" max="150" value="10">
      <div class="buttons">
        <button id="download-btn">Daten herunterladen</button>
        <button id="abort-btn" style="display: none;">Abbrechen</button>
      </div>
      <div id="spinner" class="spinner" style="display: none;"></div>
      <div id="progress-container" style="display: none;">
        <progress id="progress-bar" value="0" max="100"></progress>
        <p id="progress-text">0% abgeschlossen</p>
      </div>
      <div id="results" class="result-container" style="display: none;"></div>
    </div>
    <hr>

    <div class="section">
      <h3>2: DATENANALYSE</h3>
      <form id="analyse-form">
        <label for="pdf-directory">PDF-Verzeichnis auswählen:</label>
        <select id="pdf-directory" name="pdf_directory" required>
          <option value="">Bitte Verzeichnis wählen</option>
        </select>
        <div id="upload-area" class="upload-area">
            <p>Drag & Drop Ihre PDFs hier oder klicken Sie, um sie auszuwählen</p>
            <input type="file" id="pdf-upload" accept=".pdf" multiple style="display:none;">
            <button id="upload-btn">PDF-Dateien hochladen</button>
        </div>
        <label for="questions">Fragen (eine pro Zeile):</label>
        <textarea id="questions" name="questions" rows="5" required placeholder="z.B. Welche zentralen Faktoren beeinflussen schulische Praxis?"></textarea>
        <div style="text-align: center; margin-top: 15px;">
          <button type="submit">Analyse starten</button>
        </div>
      </form>
      <!-- Fortschrittsanzeige für Analyse -->
      <div id="analyse-progress-container" style="display: none; margin-top: 15px;">
        <progress id="analyse-progress-bar" value="0" max="100"></progress>
        <p id="analyse-progress-text">0% abgeschlossen</p>
      </div>
      <div id="analyse-spinner" class="spinner" style="display: none;"></div>
      <pre id="analyse-result"></pre>
    </div>
    <hr>

    <div class="section" id="sediment-analysis-section">
      <h3>3: DATENAUSWERTUNG</h3>
      <label for="data-directory">Data-Verzeichnis auswählen:</label>
      <select id="data-directory" name="data_directory" required>
        <option value="">Bitte Verzeichnis wählen</option>
      </select>
      <div class="buttons">
        <button id="start-sediment-btn">Auswertung starten</button>
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
}

button:disabled {
  opacity: 0.5;  
  cursor: not-allowed; 
}

body {
  background-color: #f4f4f4;
  margin: 0;
  padding: 20px;
}

.container {
  width: 90%;
  max-width: 800px;
  margin: auto;
  background: white;
  padding: 20px;
  border-radius: 8px;
  box-shadow: 0 0 10px rgba(0,0,0,0.1);
}

h1, h2, h3 {
  text-align: center;
  color: #333;
}

.progress-bar-container {
  width: 100%;
  height: 20px;
  background-color: #eee;
  border: 2px solid #3498db;
  overflow: hidden;
  position: relative;
}

.progress-bar {
  height: 100%;
  background-color: #3498db;
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
  border: 2px solid #3498db;
  border-radius: 5px;
  box-sizing: border-box;
}

textarea {
  width: 100%;
  padding: 10px;
  margin: 10px 0;
  border: 2px solid #3498db;
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
  border: none;
  border-radius: 5px;
  background-color: #3498db;
  color: white;
  cursor: pointer;
  font-size: 1em;
  transition: background-color 0.3s ease;
}

button:hover {
  background-color: #054b7a;
}

.upload-area {
  border: 2px dashed #3498db;  
  background-color: #fff;     
  padding: 20px;
  text-align: center;
  margin: 10px 0;
  cursor: pointer;
  border-radius: 8px; 
}
.upload-area.dragover {
  background-color: #e0f7fa;   
}

.result-container {
  background-color: #e8f4fb;
  border: 2px solid #3498db;
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
  border: 2px solid #3498db;
  border-radius: 0;
  overflow: hidden;
}

.section {
  margin-bottom: 40px;
}

.spinner {
  border: 8px solid #f3f3f3;
  border-top: 8px solid #3498db;
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
  background-color: #3498db;
}

progress::-moz-progress-bar {
  background-color: #3498db;
}
"""

JS_CONTENT = """
document.addEventListener('DOMContentLoaded', function() {

  // Hilfsfunktionen zum (De)aktivieren aller Prozess-Buttons
  function disableAllProcessButtons() {
    const downloadBtn = document.getElementById('download-btn');
    const analysisBtn = document.querySelector('#analyse-form button[type="submit"]');
    const sedimentBtn = document.getElementById('start-sediment-btn');
    
    if (downloadBtn) downloadBtn.disabled = true;
    if (analysisBtn) analysisBtn.disabled = true;
    if (sedimentBtn) sedimentBtn.disabled = true;
  }

  function enableAllProcessButtons() {
    const downloadBtn = document.getElementById('download-btn');
    const analysisBtn = document.querySelector('#analyse-form button[type="submit"]');
    const sedimentBtn = document.getElementById('start-sediment-btn');

    if (downloadBtn) downloadBtn.disabled = false;
    if (analysisBtn) analysisBtn.disabled = false;
    if (sedimentBtn) sedimentBtn.disabled = false;
  }

  function disableOtherProcessButtons(current) {
    const downloadBtn = document.getElementById('download-btn');
    const analysisBtn = document.querySelector('#analyse-form button[type="submit"]');
    const sedimentBtn = document.getElementById('start-sediment-btn');

    if (downloadBtn) downloadBtn.disabled = true;
    if (analysisBtn) analysisBtn.disabled = true;
    if (sedimentBtn) sedimentBtn.disabled = true;

    if (current === "download" && downloadBtn) {
        downloadBtn.disabled = false;
    } else if (current === "analysis" && analysisBtn) {
        analysisBtn.disabled = false;
    } else if (current === "sediment" && sedimentBtn) {
        sedimentBtn.disabled = false;
    }
  }

  const downloadBtn = document.getElementById('download-btn');
  const abortBtn = document.getElementById('abort-btn');
  const databaseSelect = document.getElementById('database-select');
  const searchQuery = document.getElementById('search-query');
  const publicationYear = document.getElementById('publication-year');
  const numPapersInput = document.getElementById('num-papers');
  const progressContainer = document.getElementById('progress-container');
  const progressBar = document.getElementById('progress-bar');
  const progressText = document.getElementById('progress-text');
  const resultsDiv = document.getElementById('results');
  const spinner = document.getElementById('spinner');

  let progressInterval;
  let currentDownloadId = null;

  // Download-Prozess
  downloadBtn.addEventListener('click', function() {
    const db = databaseSelect.value;
    const query = searchQuery.value.trim();
    const year = publicationYear.value.trim();
    const numPapers = parseInt(numPapersInput.value);

    if (!query || !year || !numPapers) {
      alert("Bitte alle Felder (Suchbegriffe, Jahr, Anzahl) ausfüllen.");
      return;
    }

     disableOtherProcessButtons("download"); 

    downloadBtn.innerText = "Prozess läuft…";
    downloadBtn.style.backgroundColor = "#054b7a";
    abortBtn.style.display = 'inline-block';
    abortBtn.disabled = false;
    spinner.style.display = 'block';
    resultsDiv.style.display = 'none';
    resultsDiv.innerText = '';
    progressContainer.style.display = 'block';
    progressBar.value = 0;
    progressText.innerText = "0% abgeschlossen";

    fetch('/download_papers', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ database: db, query: query, year: year, num_papers: numPapers })
    })
    .then(response => response.json())
    .then(data => {
      if (data.error) {
        alert("Fehler: " + data.error);
        enableAllProcessButtons();
        resetUI();
        return;
      }
      currentDownloadId = data.download_id;

      progressInterval = setInterval(function() {
        fetch('/download_progress?id=' + currentDownloadId)
          .then(resp => resp.json())
          .then(progressData => {
            progressBar.value = progressData.percent || 0;
            progressText.innerText = (progressData.percent || 0) + "% abgeschlossen";
            if (["completed","error","aborted"].includes(progressData.status)) {
              clearInterval(progressInterval);
              spinner.style.display = 'none';
              enableAllProcessButtons();

              downloadBtn.style.backgroundColor = "";
              downloadBtn.innerText = "Daten herunterladen";
              abortBtn.style.display = 'none';

              if (progressData.status === "completed") {
                resultsDiv.style.display = 'block';
                resultsDiv.innerText = "Download abgeschlossen. Dateien wurden im Ordner " + db + " gespeichert.";
              } else if (progressData.status === "aborted") {
                resultsDiv.style.display = 'block';
                resultsDiv.innerText = "Download abgebrochen.";
              } else {
                resultsDiv.style.display = 'block';
                resultsDiv.innerText = "Fehler beim Download: " + progressData.error;
              }
            }
          })
          .catch(error => {
            clearInterval(progressInterval);
            console.error("Fehler beim Abrufen des Fortschritts:", error);
            enableAllProcessButtons();
            resetUI();
          });
      }, 1000);
    })
    .catch(error => {
      console.error("Fehler beim Starten des Downloads:", error);
      alert("Fehler beim Starten des Downloads.");
      enableAllProcessButtons();
      resetUI();
    });
  });

  // Abbrechen
  abortBtn.addEventListener('click', function() {
    if (!currentDownloadId) {
      alert("Es läuft derzeit kein Download-Prozess.");
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
    })
    .catch(error => {
      console.error("Fehler beim Abbrechen des Downloads:", error);
    });
  });

  function resetUI() {
    clearInterval(progressInterval);
    spinner.style.display = 'none';
    downloadBtn.innerText = "Daten herunterladen";
    downloadBtn.style.backgroundColor = "#3498db";
    downloadBtn.disabled = false;
    abortBtn.style.display = 'none';
    resultsDiv.style.display = 'none';
    progressContainer.style.display = 'none';
  }

  // Verzeichnisse aktualisieren
  function updateDirectories() {
    fetch('/list_directories')
      .then(response => response.json())
      .then(data => {
        const select = document.getElementById('pdf-directory');
        const currentSelection = select.value;
        
        let newOptions = '<option value="">Bitte Verzeichnis wählen</option>';
        data.directories.forEach(dir => {
          // Hier doppelt escapen, damit im Browser /\\/g, '\\\\' ankommt
          const safeDir = dir.replace(/\\\\/g, '\\\\\\\\');
          newOptions += `<option value="${safeDir}">${safeDir}</option>`;
        });
        
        select.innerHTML = newOptions;
        
        if (currentSelection && Array.from(select.options).some(opt => opt.value === currentSelection)) {
          select.value = currentSelection;
        }
      })
      .catch(error => console.error("Fehler beim Laden der Verzeichnisse:", error));
  }
  setInterval(updateDirectories, 4000);

  function updateDataDirectories() {
    fetch('/list_data_directories')
      .then(response => response.json())
      .then(data => {
        const select = document.getElementById('data-directory');
        const currentSelection = select.value;
        let newOptions = '<option value="">Bitte Verzeichnis wählen</option>';
        data.directories.forEach(dir => {
          const safeDir = dir.replace(/\\\\/g, '\\\\\\\\');
          newOptions += `<option value="${safeDir}">${safeDir}</option>`;
        });
        select.innerHTML = newOptions;
        if (currentSelection && Array.from(select.options).some(opt => opt.value === currentSelection)) {
          select.value = currentSelection;
        }
      })
      .catch(error => console.error("Fehler beim Laden der Data-Verzeichnisse:", error));
  }
  setInterval(updateDataDirectories, 5000);
  updateDataDirectories();

  // Drag-and-Drop und Upload für PDFs
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
              alert("Upload-Fehler: " + data.error);
          } else {
              alert("Upload erfolgreich! Dateien wurden in Ordner " + data.folder + " gespeichert.");
          }
      })
      .catch(error => {
          console.error("Fehler beim Upload:", error);
          alert("Fehler beim Upload.");
      });
  }

  // Start der Sedimentanalyse
  const sedimentBtn = document.getElementById('start-sediment-btn');
  sedimentBtn.addEventListener('click', function() {
    disableOtherProcessButtons("sediment");

    const dataDir = document.getElementById('data-directory').value;
    if (!dataDir) {
      alert("Bitte wähle ein Data-Verzeichnis aus.");
      enableAllProcessButtons();
      return;
    }
    sedimentBtn.disabled = true;
    document.querySelector('.progress-bar-container').style.display = 'block';
    
    fetch('/start_sediment_analysis', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ data_directory: dataDir })
    })
    .then(response => response.json())
    .then(data => {
      if (data.error) {
        alert("Fehler: " + data.error);
        document.querySelector('.progress-bar-container').style.display = 'none';
        enableAllProcessButtons();
        return;
      }
      const analysisId = data.analysis_id;
      const sedimentInterval = setInterval(function() {
        fetch('/sediment_analysis_progress?id=' + analysisId)
          .then(resp => resp.json())
          .then(progressData => {          
            if (["completed", "error"].includes(progressData.status)) {
              clearInterval(sedimentInterval);
              document.querySelector('.progress-bar-container').style.display = 'none';
              enableAllProcessButtons();

              if (progressData.status === "completed") {
                document.getElementById('sediment-result').innerText = "Sedimentanalyse abgeschlossen.";
              } else {
                document.getElementById('sediment-result').innerText = "Fehler bei der Analyse: " + progressData.error;
              }
            }
          })
          .catch(error => {
            clearInterval(sedimentInterval);
            console.error("Fehler beim Abrufen des Sedimentanalyse-Fortschritts:", error);
            enableAllProcessButtons();
            document.querySelector('.progress-bar-container').style.display = 'none';
          });
      }, 1000);
    })
    .catch(error => {
      console.error("Fehler beim Starten der Sedimentanalyse:", error);
      enableAllProcessButtons();
      document.querySelector('.progress-bar-container').style.display = 'none';
    });
  });

  // Analyseformular-Verarbeitung
  const analyseForm = document.getElementById("analyse-form");
  analyseForm.addEventListener("submit", function(e) {
    e.preventDefault();
    
    disableOtherProcessButtons("analysis");

    const analyseProgressContainer = document.getElementById("analyse-progress-container");
    const analyseProgressBar = document.getElementById("analyse-progress-bar");
    const analyseProgressText = document.getElementById("analyse-progress-text");
    const analyseSpinner = document.getElementById("analyse-spinner");
    
    analyseProgressContainer.style.display = "block";
    analyseSpinner.style.display = "block";
    analyseProgressBar.value = 0;
    analyseProgressText.innerText = "0% abgeschlossen";
    
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
        return;
      }
      const analysisId = result.analysis_id;
      const analysisInterval = setInterval(function() {
        fetch('/analysis_progress?id=' + analysisId)
        .then(resp => resp.json())
        .then(progressData => {
          analyseProgressBar.value = progressData.percent || 0;
          analyseProgressText.innerText = (progressData.percent || 0) + "% abgeschlossen";
          if (["completed", "error", "aborted"].includes(progressData.status)) {
            clearInterval(analysisInterval);
            analyseSpinner.style.display = "none";
            enableAllProcessButtons();

            if (progressData.status === "completed") {
              alert("Analyse abgeschlossen.");
            }
          }
        })
        .catch(error => {
          clearInterval(analysisInterval);
          console.error("Fehler beim Abrufen des Analysefortschritts:", error);
          enableAllProcessButtons();
        });
      }, 1000);
    })
    .catch(error => {
      console.error("Fehler beim Starten der Analyse:", error);
      alert("Fehler beim Starten der Analyse.");
      enableAllProcessButtons();
    });
  });
});
"""

# ---------------------------
# Flask-Endpunkte
# ---------------------------
@app.route('/')
def index():
    return Response(HTML_CONTENT, mimetype='text/html')

@app.route('/styles.css')
def styles():
    return Response(CSS_CONTENT, mimetype='text/css')

@app.route('/script.js')
def script():
    return Response(JS_CONTENT, mimetype='application/javascript')

@app.route('/download_papers', methods=['POST'])
def download_papers():
    global current_process
    data = request.get_json()
    database = data.get("database")
    query = data.get("query")
    year = data.get("year")
    num_papers = data.get("num_papers")

    if not database or not query or not year or not num_papers:
        return jsonify({"error": "Bitte alle Parameter (Datenbank, Suchbegriffe, Jahr, Anzahl) angeben."})

    with process_lock:
        if current_process is not None:
            return jsonify({"error": f"Ein Prozess läuft bereits: {current_process}"}), 400
        current_process = "download"

    download_id = str(uuid.uuid4())
    download_progress[download_id] = {"status": "starting", "percent": 0}

    thread = threading.Thread(target=download_papers_background, args=(download_id, database, query, year, num_papers))
    thread.daemon = False
    thread.start()

    return jsonify({"download_id": download_id})

@app.route('/download_progress', methods=['GET'])
def get_download_progress():
    download_id = request.args.get("id")
    if not download_id or download_id not in download_progress:
        return jsonify({"error": "Ungültige Download-ID."})
    return jsonify(download_progress[download_id])

@app.route('/abort_download', methods=['POST'])
def abort_download():
    data = request.get_json()
    download_id = data.get("download_id")
    if download_id in download_progress:
        download_progress[download_id]["abort"] = True
        return jsonify({"message": "Abbruch angefordert"})
    return jsonify({"error": "Ungültige Download-ID"}), 400

@app.route('/list_directories', methods=['GET'])
def list_directories():
    dirs = [d for d in os.listdir(DOWNLOAD_ROOT) if os.path.isdir(os.path.join(DOWNLOAD_ROOT, d))]
    if not dirs:
        dirs = ["."]
    return jsonify({"directories": dirs})

@app.route('/start_analysis', methods=['POST'])
def start_analysis():
    global current_process
    data = request.get_json()
    pdf_directory = data.get("pdf_directory")
    questions = data.get("questions")
    
    if not pdf_directory or not questions:
        return jsonify({"error": "Bitte sowohl ein PDF-Verzeichnis als auch Fragen angeben."})
    
    with process_lock:
        if current_process is not None:
            return jsonify({"error": f"Ein Prozess läuft bereits: {current_process}"}), 400
        current_process = "analysis"
    
    question_lines = [line.strip() for line in questions.splitlines() if line.strip()]
    expected_count = len(question_lines)
    
    if expected_count == 1:
        context_query = f"""Bitte antworte auf die folgende Frage ausschließlich mit einem prägnanten, korrekten Satz. Verwende weder zusätzliche Einleitungen noch Nummerierungen oder Wiederholungen der Frage.
Frage: {question_lines[0]}
"""
    else:
        context_query = "Bitte antworte prägnant auf die folgenden Fragen. Für jede Frage soll ausschließlich ein Satz als Antwort ausgegeben werden – ohne Einleitungen, Nummerierungen oder Wiederholungen der Frage:\n"
        for idx, q in enumerate(question_lines, start=1):
            context_query += f"Frage {idx}: {q}\n"
    
    analysis_id = str(uuid.uuid4())
    analysis_progress[analysis_id] = {"status": "starting", "percent": 0}
    
    pdf_dir_name = os.path.basename(pdf_directory)
    current_date = datetime.now().strftime("%Y-%m-%d")
    analysis_output_folder = os.path.join(DATA_ROOT, f"{pdf_dir_name}_{current_date}")
    if not os.path.exists(analysis_output_folder):
        os.makedirs(analysis_output_folder)
    
    questions_file = os.path.join(analysis_output_folder, "questions.txt")
    with open(questions_file, "w", encoding="utf-8") as f:
        for q in question_lines:
            f.write(q + "\n")
    
    thread = threading.Thread(
        target=run_analysis,
        args=(analysis_id, os.path.join(DOWNLOAD_ROOT, pdf_directory), context_query, expected_count)
    )
    thread.daemon = False
    thread.start()
    return jsonify({"analysis_id": analysis_id})

@app.route('/analysis_progress', methods=['GET'])
def get_analysis_progress():
    analysis_id = request.args.get("id")
    if not analysis_id or analysis_id not in analysis_progress:
        return jsonify({"error": "Ungültige Analysis-ID."})
    return jsonify(analysis_progress[analysis_id])    

# ---------------------------
# Anwendung starten
# ---------------------------
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
