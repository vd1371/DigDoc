import os
import pickle
import numpy as np


from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from rank_bm25 import BM25Okapi
from sklearn.metrics.pairwise import cosine_similarity

from utils.utils import get_context, save_response
from utils.DocumentReader import DocumentReader

from dotenv import load_dotenv
load_dotenv()

BASE_DIREC = "projects"

class DigDoc:

    def __init__(self, **params):
        self.model = params.get("model", "gpt-4o")
        self.vectorstore = None

        self.target = params.get("target", [])
        self.reindex = params.get("reindex", True)

        self.retriever = params["retriever"]
        assert self.retriever in ['FAISS', 'bm25'], "Retriever should be either FAISS or bm25"

        # Handle the directories for saving the files
        self.project_name = params.get("project_name", "default")
        self.directory = params.get("directory", "web")
        self.file_name = os.path.join(BASE_DIREC, self.project_name, "ChatHistory.html")
        os.makedirs(BASE_DIREC, exist_ok=True)
        self.project_path = os.path.join(BASE_DIREC, self.project_name)
        os.makedirs(self.project_path, exist_ok=True)

        self.look_back_window = 3
        self.chat_history = []

        self.k = 5 # The number of documents to retrieve
        self.document_reader = DocumentReader(
            project_path = self.project_path,
            **params
            )


    def index(self, query=None):
        if query is None and self.directory=="google":
            """
            The reason is that we need to have a way to pass the query to the dig method
            The Jupyter architecture cannot be changed, so I will dig when the answer method is called with the query
            """
            return

        if self.retriever == "FAISS":
            index_path = os.path.join(self.project_path, 'index.faiss')
            if os.path.exists(index_path) and not self.reindex:
                x = FAISS.load_local(
                    self.project_path,
                    embeddings=OpenAIEmbeddings(),
                    allow_dangerous_deserialization=True
                )
                self.vectorstore = x

        elif self.retriever == "bm25":
            index_path = os.path.join(self.project_path, "tokenized_texts.pkl")
            if os.path.exists(index_path) and not self.reindex:
                with open(index_path, 'rb') as f:
                    tokenized_texts = pickle.load(f)
                self.bm25 = BM25Okapi(tokenized_texts)
        
        # Read the documents
        raw_documents = self.document_reader.read(query)

        # Split text into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
        texts = text_splitter.split_documents(raw_documents)
        
        if self.retriever == "FAISS":
            vectorstore = FAISS.from_documents(texts, OpenAIEmbeddings())
            self.vectorstore = vectorstore
            if self.reindex:
                vectorstore.save_local(self.project_path)

        elif self.retriever == "bm25":

            self.original_texts = [doc.page_content for doc in texts]
            tokenized_texts = [doc.page_content.split() for doc in texts]  # Tokenize the documents for BM25
            self.bm25 = BM25Okapi(tokenized_texts)

            # Save the tokenized_texts for future use as pkl file
            if self.reindex:
                with open(index_path, 'wb') as f:
                    pickle.dump(tokenized_texts, f)

    
    def answer(self, query):

        if self.directory == "google":
            self.index(query)

        prompt_template = get_prompt_template(self.chat_history, self.look_back_window)        

        if self.retriever == "FAISS":
            PROMPT = PromptTemplate(
                template=prompt_template,
                input_variables=["context", "question"]
            )

            # Create a retrieval-based question-answering chain with GPT-4
            qa_chain = RetrievalQA.from_chain_type(
                llm=get_llm(self.model),
                chain_type="stuff",
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": self.k}),  # Retrieve top 4 documents
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

        elif self.retriever == "bm25":
        
            # Retrieve Top-N candidates using BM25
            query_terms = query.split()
            bm25_indices = [self.original_texts.index(doc) for doc in self.bm25.get_top_n(query_terms, self.original_texts, n=100)]
            bm25_candidates = [self.original_texts[idx] for idx in bm25_indices]

            # Use your embeddings model to compute the embeddings
            embeddings_model = OpenAIEmbeddings()  # Change this to your specific embeddings model if different

            # Compute embeddings for the query
            query_embedding = embeddings_model.embed_query(query)
            query_embedding = np.array(query_embedding).reshape(1, -1)

            # Compute embeddings for BM25 candidates
            candidate_embeddings = embeddings_model.embed_documents(bm25_candidates)
            candidate_embeddings = np.array(candidate_embeddings)

            # Calculate cosine similarity
            similarities = cosine_similarity(query_embedding, candidate_embeddings)[0]

            # Select top K documents based on cosine similarity
            top_k_indices = np.argsort(similarities)[-self.k:][::-1]
            top_k_docs = [bm25_candidates[i] for i in top_k_indices]

            # Compose final context based on top K documents
            context = ' '.join(top_k_docs)

            llm = get_llm(self.model)
            response = llm.invoke(f"{prompt_template.format(context=context, question=query)}").content


        self.chat_history.append({
            "query": query,
            "response": response
        })

        save_response(self.chat_history, self.file_name, self.project_name)

        return response



# ---------------------------------------------------------------------------- #
#                       Utils for Digestor                                    #
# ---------------------------------------------------------------------------- #


def get_llm(model):
    if model == "gpt-4o":
        return ChatOpenAI(model_name="gpt-4o")

    elif model == "anthropic":
        return ChatAnthropic(model_name="claude-3-5-sonnet-20240620")

    else:
        raise ValueError(f"Model {model} not found.")


def get_prompt_template(chat_history, look_back_window):

    # Create a custom prompt template
    prompt_template = """
    Your role is a research assistant at a university that helps with reviewing documents.
    You have been asked to find information based on the provided context and question.
    Say you don't know if there are not enough related information\n"""

    prompt_template = "[HISTORY STARTS] Here is the chat history:\n"
    prompt_template += f"{get_context(chat_history, look_back_window)}\n"
    prompt_template += "[HISTORY ENDS]"

    prompt_template += "Context: {context}\n"
    prompt_template += "Question: {question}\n"
    prompt_template += "Answer: "

    return prompt_template

