import os
import re
import json
import datetime

import mobi

# from langchain.text_splitter import CharacterTextSplitter
# from langchain_community.vectorstores import FAISS
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters.base import Document
# from langchain_openai import OpenAIEmbeddings
# from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
# from langchain.chains import RetrievalQA
# from langchain.prompts import PromptTemplate

from llama_index.core import VectorStoreIndex
from llama_index.core import load_index_from_storage
from llama_index.core import PromptTemplate
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core.tools import QueryEngineTool

from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document
from llama_index.readers.file import (
            DocxReader,
            EpubReader,
            HWPReader,
            ImageReader,
            IPYNBReader,
            MarkdownReader,
            MboxReader,
            PandasCSVReader,
            PandasExcelReader,
            PDFReader,
            PptxReader,
            VideoAudioReader,
        )
from llama_index.core.selectors import (
            PydanticMultiSelector,
            PydanticSingleSelector,
        )



from .utils import get_context, save_response
from .WebScraper import WebScraper

from dotenv import load_dotenv
load_dotenv()

BASE_DIREC = "projects"

class DigDoc:

    def __init__(self, **params):
        self.model = params.get("model", "gpt-4o")
        self.vectorstore = None

        self.target = params.get("target", [])
        self.force_reindex = params.get("force_reindex", True)

        self.project_name = params.get("project_name", "default")
        self.directory = params.get("directory", "web")
        self.file_name = os.path.join(BASE_DIREC, self.project_name, "ChatHistory.html")
        os.makedirs(BASE_DIREC, exist_ok=True)
        self.project_path = os.path.join(BASE_DIREC, self.project_name)
        os.makedirs(self.project_path, exist_ok=True)

        self.look_back_window = 3
        self.chat_history = []
        self.index = None

        self.max_depth = 1
        self.max_pages = 1
        self.load_from_cache = False

    # def answer(self, query):
    #
    #     if self.directory == "google":
    #         self.dig(query)
    #
    #     # Create a custom prompt template
    #     prompt_template = """
    #     Your role is a research assistant at a university that helps with reviewing documents.
    #     You have been asked to find information based on the provided context and question.
    #     Say you don't know if there are not enough related information\n"""
    #
    #     prompt_template = "[HISTORY STARTS] Here is the chat history:\n"
    #     prompt_template += f"{get_context(self.chat_history, self.look_back_window)}\n"
    #     prompt_template += "[HISTORY ENDS]"
    #
    #     prompt_template += """
    #     Context:
    #     {context}
    #
    #     Question: {question}
    #     Answer: """
    #
    #     PROMPT = PromptTemplate(
    #         template=prompt_template,
    #         input_variables=["context", "question"]
    #     )
    #
    #     # Create a retrieval-based question-answering chain with GPT-4
    #     qa_chain = RetrievalQA.from_chain_type(
    #         llm=get_llm(self.model),
    #         chain_type="stuff",
    #         retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5}),  # Retrieve top 4 documents
    #         return_source_documents=True,
    #         chain_type_kwargs={"prompt": PROMPT}
    #     )
    #
    #     result = qa_chain.invoke({
    #         "query": query,
    #     })
    #
    #     # Ensure necessary keys are present in the result
    #     if 'result' not in result or 'source_documents' not in result:
    #         raise ValueError("The response from qa_chain is missing expected keys.")
    #
    #     response, source_docs = result["result"], result["source_documents"]
    #
    #     self.chat_history.append({
    #         "query": query,
    #         "response": response
    #     })
    #
    #     save_response(self.chat_history, self.file_name, self.project_name)
    #
    #     return result["result"], result["source_documents"]
    #
    # def dig(self, query=None):
    #     if not query and self.directory=="google":
    #         """
    #         The reason is that we need to have a way to pass the query to the dig method
    #         The Jupyter architecture cannot be changed, so I will dig when the answer method is called with the query
    #         """
    #         return
    #
    #     if self.directory == "google":
    #         assert len(self.target) == 1, "You need to pass a single website, or 'all' to the target if directory is google"
    #         assert self.max_depth == 1, "Max depth should be 1 for google search"
    #         assert self.max_pages <= 10, "Max pages should be max 10 for google search"
    #
    #     if len(self.target) == 0:
    #         raise ValueError("You need to pass at least one document to read.")
    #
    #     # This is the case when we have already vectorized the documents
    #     if os.path.exists(self.project_path) and not self.reindex:
    #         x = FAISS.load_local(
    #             self.project_path,
    #             embeddings=OpenAIEmbeddings(),
    #             allow_dangerous_deserialization=True
    #         )
    #         self.vectorstore = x
    #
    #     # Iterate over all PDFs in the directory recursively in all folders
    #     if self.directory == "web":
    #         raw_documents = self.get_content_of_website()
    #
    #     elif self.directory == "google":
    #         raw_documents = self.get_content_from_google_search(query)
    #
    #     else:
    #         raw_documents = self.get_documents_raw_text()
    #
    #     # Split text into chunks
    #     text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    #     texts = text_splitter.split_documents(raw_documents)
    #
    #     # Create embeddings
    #     embeddings = OpenAIEmbeddings()
    #
    #     # Create vector store
    #     vectorstore = FAISS.from_documents(texts, embeddings)
    #     vectorstore.save_local(self.project_path)
    #
    #     self.vectorstore = vectorstore

    def answer(self, query):
        if self.directory == "google":
            self.dig(query)

        if not self.index:
            self.load_or_create_index()

        prompt_template = """
        Your role is a research assistant at a university that helps with reviewing documents.
        You have been asked to find information based on the provided context and question.
        Say you don't know if there are not enough related information

        [HISTORY STARTS] Here is the chat history:
        {history}
        [HISTORY ENDS]

        Question: {question}
        Answer: """

        history = get_context(self.chat_history, self.look_back_window)
        full_prompt = prompt_template.format(history=history, question=query)

        vector_query_engine = self.index.as_query_engine(similarity_top_k=50)
        vector_tool = QueryEngineTool.from_defaults(
            query_engine=vector_query_engine,
            description=(
                "Useful for retrieving specific context from the Document."
            ),
        )

        query_engine = RouterQueryEngine(
            selector=PydanticSingleSelector.from_defaults(),
            query_engine_tools=[
                vector_tool,
            ],
            verbose=True,
        )

        response = query_engine.query(full_prompt)

        self.chat_history.append({
            "query": query,
            "response": str(response)
        })

        save_response(self.chat_history, self.file_name, self.project_name)

        return str(response), response.source_nodes

    def dig(self, query=None):
        if not query and self.directory == "google":
            return

        if self.directory == "google":
            assert len(
                self.target) == 1, "You need to pass a single website, or 'all' to the target if directory is google"
            assert self.max_depth == 1, "Max depth should be 1 for google search"
            assert self.max_pages <= 10, "Max pages should be max 10 for google search"

        if len(self.target) == 0:
            raise ValueError("You need to pass at least one document to read.")

        self.load_or_create_index(query=query)

    def load_or_create_index(self, query=None):

        if os.path.exists(self.project_path) and not self.force_reindex:
            storage_context = StorageContext.from_defaults(persist_dir=self.project_path)
            self.index = load_index_from_storage(storage_context)
        else:
            if self.directory == "web":
                documents = self.get_content_of_website()
            elif self.directory == "google":
                documents = self.get_content_from_google_search(query)
            else:
                documents = self.get_documents_raw_text()

            Settings.llm = OpenAI(model="gpt-4o")
            Settings.embed_model = OpenAIEmbedding()
            Settings.node_parser = SimpleNodeParser.from_defaults(chunk_size=1000, chunk_overlap=20)
            Settings.sentence_splitter = SentenceSplitter.from_defaults()

            storage_context = StorageContext.from_defaults()
            self.index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
            )
            self.index.storage_context.persist(persist_dir=self.project_path)

    def load_filtered_documents(self):

        documents = []
        for root, dirs, files in os.walk(self.directory):
            for fl_nm in files:
                filename = fl_nm.lower()

                if isinstance(self.target, list) and \
                        not any(doc.lower() in filename for doc in self.target):
                    continue

                print (filename, "is being processed")
                content = self.read_file_content(os.path.join(root, filename))
                if content:
                    doc = {"content": content, "metadata": {"source": os.path.join(root, filename)}}
                    documents.append(doc)

        return documents

    @staticmethod
    def read_file_content(file_path):
        _, file_extension = os.path.splitext(file_path)

        if file_extension.lower() in ['.cpp', '.h', '.c']:
            with open(file_path, 'r') as file:
                return file.read()

        elif file_extension.lower() in ['.mobi']:
            book = mobi.Mobi(file_path)
            content = book.get_text()
            return content

        # Add more file type handlers here

        else:
            print(f"Unsupported file type: {file_extension}")
            return None

    def set_scrapping_params(self, max_depth, max_pages, load_from_cache):
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.load_from_cache = load_from_cache

    def get_documents_raw_text(self):
        raw_documents = []
        n_files = 0

        for root, dirs, files in os.walk(self.directory):
            for filename in files:

                if isinstance(self.target, list) and \
                        not any(doc.lower() in filename.lower() for doc in self.target):
                    continue

                print (filename.lower(), "is being processed")

                if filename.lower().endswith('.pdf'):
                    reader = PDFReader()
                    raw_documents.extend(reader.load_data(os.path.join(root, filename)))
                    n_files += 1

                elif filename.lower().endswith('.txt'):
                    with open(os.path.join(root, filename), 'r') as f:
                        file_content = f.read()
                    raw_documents.append(Document(file_content, metadata={"source": os.path.join(root, filename)}))
                    n_files += 1

                elif filename.lower().endswith('.epub'):
                    reader = EpubReader()
                    raw_documents.extend(reader.load_data(os.path.join(root, filename)))
                    n_files += 1

                elif filename.lower().split('.')[-1] in ['cpp', 'c', 'h']:
                    with open(os.path.join(root, filename), 'r') as f:
                        file_content = f.read()
                    raw_documents.append(Document(file_content, metadata={"source": os.path.join(root, filename)}))
                    n_files += 1

                elif filename.lower().endswith(".mobi"):
                    book = mobi.Mobi(os.path.join(root, filename))
                    content = book.get_text()
                    if content:
                        raw_documents.append(Document(content, metadata={"source": os.path.join(root, filename)}))
                        n_files += 1

                elif filename.lower().split('.')[-1] in ['docx', 'doc']:
                    reader = DocxReader()
                    raw_documents.extend(reader.load_data(os.path.join(root, filename)))
                    n_files += 1

                elif filename.lower().endswith(".py"):
                    with open(os.path.join(root, filename), 'r') as f:
                        file_content = f.read()
                    raw_documents.append(Document(file_content, metadata={"source": os.path.join(root, filename)}))
                    n_files += 1

                elif filename.lower().endswith(".ipynb"):
                    reader = IPYNBReader()
                    raw_documents.extend(reader.load_data(os.path.join(root, filename)))
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
            load_from_cache=self.load_from_cache,
            project_path=self.project_path
        )

        for url in self.target:

            contents = scraper.scrape_website(url)
            for content in contents:

                new_doc = Document(content[1], metadata={"source": content[0]})
                raw_documents.append(new_doc)

        return raw_documents

    def get_content_from_google_search(self, query):

        raw_documents = []
        scraper = WebScraper(
            max_pages=self.max_pages,
            max_depth=self.max_depth,
            load_from_cache=self.load_from_cache,
            project_path=self.project_path
        )

        contents = scraper.scrape_google_search(self.target[0], query)
        for url, content in contents:

            new_doc = Document(content, metadata={"source": url})
            raw_documents.append(new_doc)

        return raw_documents


def remove_between_tags(text, start_tag, end_tag):
    pattern = f'{re.escape(start_tag)}.*?{re.escape(end_tag)}'
    return re.sub(pattern, '', text, flags=re.DOTALL)