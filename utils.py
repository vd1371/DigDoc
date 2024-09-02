import os

from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.environ["API_KEY"]
llm = ChatOpenAI(model_name="gpt-4o")

# Function to ask questions
def ask_question(question, vectorstore):
    # Create a custom prompt template
    prompt_template = """
    Use the following pieces of context to answer the question.

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
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),  # Retrieve top 4 documents
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    result = qa_chain.invoke({
        "query": question,
    })

    # Ensure necessary keys are present in the result
    if 'result' not in result or 'source_documents' not in result:
        raise ValueError("The response from qa_chain is missing expected keys.")

    answer, source_docs = result["result"], result["source_documents"]
    markdown = generate_markdown_output(question, answer, source_docs)
    save_markdown(markdown, "result.md")

    return result["result"], result["source_documents"]


def convert_chat_history(chat_history, n_turns=2):
    """
    :param chat_history: list of dicts with keys 'question', 'answer'
    :return: string with chat history formatted as a conversation
    """

    if len(chat_history) == 0:
        return  ""

    chat_history = chat_history[-n_turns:]
    chat = "[HISTORY] Here is the chat history:\n"
    for i, chat_his in enumerate(chat_history):
        chat += f"User: {chat_his['question']}\n"
        chat += f"AI: {chat_his['answer']}\n"
    return chat


def get_vectorized_of_pdfs(directory, project_name, reindex):

    base_direc = "projects"
    os.makedirs(base_direc, exist_ok=True)
    project_path = os.path.join(base_direc, project_name)

    if os.path.exists(project_path) and not reindex:
        x = FAISS.load_local(project_path)
        return x.as_retriever()

    raw_documents = []

    # Iterate over all PDFs in the directory recursively in all folders
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(root, filename)
                loader = PyPDFLoader(pdf_path)
                raw_documents.extend(loader.load())

    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(raw_documents)

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Create vector store
    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(project_path)

    return vectorstore


def generate_markdown_output(question, answer, source_docs):
    markdown = f"""# Question and Answer

## Question
{question}

## Answer
{answer}

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


def save_markdown(markdown, filename="output.md"):
    with open(filename, "w", encoding="utf-8") as f:
        f.write(markdown)
    print(f"Markdown saved to {filename}")