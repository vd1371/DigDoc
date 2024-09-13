import os
import re
import datetime

import json
import requests
import time
import random
from urllib.parse import urljoin

import ebooklib
from ebooklib import epub
import mobi
import docx
import nbformat

from bs4 import BeautifulSoup

from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters.base import Document

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


from dotenv import load_dotenv
load_dotenv()

from .utils import get_context, save_response


BASE_DIREC = "projects"

class DigDoc:

    def __init__(self, **params):
        self.model = params.get("model", "gpt-4o")
        self.vectorstore = None

        self.target_docs = params.get("target_docs", [])
        self.reindex = params.get("reindex", True)

        self.project_name = params.get("project_name", "default")
        self.docs_directory = params.get("docs_directory", "web")
        self.file_name = os.path.join(BASE_DIREC, self.project_name, "ChatHistory.html")
        os.makedirs(BASE_DIREC, exist_ok=True)
        self.project_path = os.path.join(BASE_DIREC, self.project_name)
        os.makedirs(self.project_path, exist_ok=True)

        self.look_back_window = 3
        self.chat_history = []

        self.max_depth = 1
        self.max_pages = 1
        self.load_from_cache = False

    def answer(self, query):
        # Create a custom prompt template
        prompt_template = """
        Your role is a research assistant at a university that helps with reviewing documents.
        You have been asked to find information based on the provided context and question.
        Use the following pieces of context to answer the question.\n"""

        prompt_template = "[HISTORY STARTS] Here is the chat history:\n"
        prompt_template += f"{get_context(self.chat_history, self.look_back_window)}\n"
        prompt_template += "[HISTORY ENDS]"

        prompt_template += """
        Context:
        {context}

        Question: {question}
        Answer: """

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Create a retrieval-based question-answering chain with GPT-4
        qa_chain = RetrievalQA.from_chain_type(
            llm=get_llm(self.model),
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),  # Retrieve top 4 documents
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

        result = qa_chain.invoke({
            "query": query,
        })

        # Ensure necessary keys are present in the result
        if 'result' not in result or 'source_documents' not in result:
            raise ValueError("The response from qa_chain is missing expected keys.")

        response, source_docs = result["result"], result["source_documents"]

        self.chat_history.append({
            "query": query,
            "response": response
        })

        save_response(self.chat_history, self.file_name, self.project_name)

        return result["result"], result["source_documents"]

    def dig(self):
        """
        :param target_docs: list of strings to filter the PDFs to vectorize
        :param reindex: whether to reindex the documents
        """

        if len(self.target_docs) == 0:
            raise ValueError("You need to pass at least one document to read.")

        # This is the case when we have already vectorized the documents
        if os.path.exists(self.project_path) and not self.reindex:
            x = FAISS.load_local(
                self.project_path,
                embeddings=OpenAIEmbeddings(),
                allow_dangerous_deserialization=True
            )
            self.vectorstore = x

        # Iterate over all PDFs in the directory recursively in all folders
        if self.docs_directory == "web":
            raw_documents = self.get_content_of_website()
        else:
            raw_documents = self.get_documents_raw_text()

        # Split text into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(raw_documents)

        # Create embeddings
        embeddings = OpenAIEmbeddings()

        # Create vector store
        vectorstore = FAISS.from_documents(texts, embeddings)
        vectorstore.save_local(self.project_path)

        self.vectorstore = vectorstore

    def set_scrapping_params(self, max_depth, max_pages, load_from_cache):
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.load_from_cache = load_from_cache

    def get_documents_raw_text(self):
        raw_documents = []
        n_files = 0

        for root, dirs, files in os.walk(self.docs_directory):
            for filename in files:

                if isinstance(self.target_docs, list) and \
                        not any(doc.lower() in filename.lower() for doc in self.target_docs):
                    continue

                print (filename, "is being processed")

                if filename.endswith('.pdf'):
                    pdf_path = os.path.join(root, filename)
                    loader = PyPDFLoader(pdf_path)
                    raw_documents.extend(loader.load())
                    n_files += 1

                elif filename.endswith('.txt'):
                    with open(os.path.join(root, filename), 'r') as f:
                        file_content = f.read()
                    raw_documents.append(Document(file_content))
                    n_files += 1

                elif filename.endswith('.epub'):
                    book = epub.read_epub(os.path.join(root, filename))
                    for item in book.get_items():
                        if item.get_type() == ebooklib.ITEM_DOCUMENT:
                            content = item.get_content()
                            if isinstance(content, bytes):
                                content = content.decode('utf-8')

                            text = BeautifulSoup(content, 'html.parser').get_text(strip=True)
                            if len(text) < 100:
                                continue

                            raw_documents.append(Document(content))
                            n_files += 1

                # Add cpp and c files
                elif filename.split('.')[-1] in ['cpp', 'c', 'h']:
                    with open(os.path.join(root, filename), 'r') as f:
                        file_content = f.read()
                    raw_documents.append(Document(file_content))
                    n_files += 1

                elif filename.endswith(".mobi"):
                    book = mobi.Mobi(os.path.join(root, filename))
                    content = book.get_text()
                    if content:
                        raw_documents.append(Document(content))
                        n_files += 1

                elif filename.endswith(".docx") or filename.endswith(".doc"):
                    doc = docx.Document(os.path.join(root, filename))
                    content = ""
                    for para in doc.paragraphs:
                        content += para.text + "\n"
                    raw_documents.append(Document(content))
                    n_files += 1

                elif filename.endswith(".py"):
                    with open(os.path.join(root, filename), 'r') as f:
                        file_content = f.read()
                    raw_documents.append(Document(file_content))
                    n_files += 1

                elif filename.endswith(".ipynb"):
                    with open(os.path.join(root, filename), 'r') as f:
                        file_content = f.read()
                    nb = nbformat.reads(file_content, as_version=4)
                    for cell in nb.cells:
                        if cell.cell_type == 'code':
                            raw_documents.append(Document(cell.source))
                    n_files += 1

                else:
                    pass

        if n_files == 0:
            raise ValueError("No Files found.")

        return raw_documents
    
    def get_content_of_website(self):

        raw_documents = []
        scraper = WebScraper(
            max_pages=self.max_pages,
            max_depth=self.max_depth,
            load_from_cache=self.load_from_cache
        )

        for url in self.target_docs:

            contents = scraper.scrape_website(url)
            for content in contents:

                new_doc = Document(content[1])
                new_doc.metadata['source'] = content[0]
                raw_documents.append(new_doc)

        return raw_documents



# ---------------------------------------------------------------------------- #
#                       Utils for Digestor                                    #
# ---------------------------------------------------------------------------- #

def generate_markdown_output(question, answer, source_docs):
    # Remove the HISTORY STARTS and HISTORY ENDS and anything in between from
    # the question and answer
    question = remove_between_tags(question, "[HISTORY STARTS]", "[HISTORY ENDS]")

    markdown = f"""

# {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Answer
--------------------------------------------------------------------------------
{answer}
--------------------------------------------------------------------------------

## Question
{question}

## Source Documents
"""
    for i, doc in enumerate(source_docs):
        source = doc.metadata.get('source', 'Unknown')
        page = doc.metadata.get('page', 'Unknown')
        content = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content

        markdown += f"""
### Document {i + 1}
- **Source**: {source}
- **Page**: {page}

#### Content Preview {content}

---
"""
    return markdown


def save_markdown(markdown, project_name, append=False):
    file_path = os.path.join("../projects", project_name, "ChatHistory.md")

    # Ensure the file exists, create it if it doesn't
    if not os.path.exists(file_path):
        open(file_path, 'w', encoding="utf-8").close()

    existing_content = ""

    # Read existing content
    if append:
        try:
            with open(file_path, 'r', encoding="utf-8") as file:
                existing_content = file.read()
        except UnicodeDecodeError:
            # If utf-8 fails, try with a different encoding
            with open(file_path, 'r', encoding="iso-8859-1") as file:
                existing_content = file.read()

    # Write new content
    with open(file_path, 'w', encoding="utf-8") as file:
        file.write(markdown + existing_content)

    print(f"Markdown saved to {file_path}")


def remove_between_tags(text, start_tag, end_tag):
    pattern = f'{re.escape(start_tag)}.*?{re.escape(end_tag)}'
    return re.sub(pattern, '', text, flags=re.DOTALL)


def get_llm(model):
    if model == "gpt-4o":
        return ChatOpenAI(model_name="gpt-4o")

    elif model == "anthropic":
        return ChatAnthropic(model_name="claude-3-5-sonnet-20240620")

    else:
        raise ValueError(f"Model {model} not found.")


class WebScraper:
    def __init__(self, **params):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.session = requests.Session()
        self.max_pages = params.get("max_pages", 10)
        self.max_depth = params.get("max_depth", 1)
        self.load_from_cache = params.get("load_from_cache", False)

        self.cache_direc = "scrapping_cache.json"
        if not os.path.exists(self.cache_direc):
            with open(self.cache_direc, "w") as f:
                json.dump({}, f)
        
        with open(self.cache_direc, "r") as f:
            self.cache = json.load(f)

    def add_to_cache(self, url, content, page_urls):
        self.cache[url] = {
            "content": content,
            "page_urls": page_urls
        }
        with open(self.cache_direc, "w") as f:
            json.dump(self.cache, f)

    def get_page(self, url):
        try:
            response = self.session.get(url, headers=self.headers)
            response.raise_for_status()
            return response.text
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None

    def parse_content(self, html):
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text

    def scrape_website(self, start_url, max_pages=10, max_depth=1):
        
        visited = set()
        assert max_depth > 0, "max_depth must be greater than 0, at least 1"

        to_visit = {i: [] for i in range(max_depth)}
        to_visit[0] = [start_url]

        scraped_content = []
        for depth in to_visit:
            for url in to_visit[depth]:

                if url in visited or len(visited) >= max_pages:
                    continue

                if url in self.cache and self.load_from_cache:
                    print (f"Using cached content for {url}")
                    scraped_content.append((url, self.cache[url]["content"]))
                    visited.add(url)

                    # Add new urls to visit
                    for new_url in self.cache[url]["page_urls"]:
                        if new_url not in visited and depth + 1 < max_depth:
                            to_visit[depth + 1].append(new_url)

                    continue

                print(f"Scraping: {url}")
                html = self.get_page(url)
                if html:
                    content = self.parse_content(html)
                    scraped_content.append((url, content))
                    visited.add(url)

                    # Find more links
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    new_urls = []
                    for link in soup.find_all('a', href=True):
                        new_url = urljoin(url, link['href'])

                        if "javascript:void(0)" in new_url:
                            continue
                        new_urls.append(new_url)
                        
                        if new_url not in visited and depth + 1 < max_depth:
                            to_visit[depth + 1].append(new_url)

                    self.add_to_cache(url, content, new_urls)

                # Add a random delay between requests
                time.sleep(random.uniform(1, 3))

        return scraped_content