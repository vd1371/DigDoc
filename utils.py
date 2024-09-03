import os
import re

from openai import OpenAI
import anthropic
from anthropic.types import TextBlock

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader


from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

from langchain_anthropic import ChatAnthropic

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv
load_dotenv()

# Function to ask questions
def ask_question(question, vectorstore, model):
    # Create a custom prompt template
    prompt_template = """
    Your role is a research assistant at a university that helps with
    literature reviews. You have been asked to find information based on the provided
    context and question.

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
        llm=get_llm(model),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),  # Retrieve top 4 documents
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
    save_markdown(markdown, "LiteratureReview.md")

    return result["result"], result["source_documents"]


def convert_chat_history(chat_history, n_turns=2):
    """
    :param chat_history: list of dicts with keys 'question', 'answer'
    :return: string with chat history formatted as a conversation
    """

    if len(chat_history) == 0:
        return  ""

    chat_history = chat_history[-n_turns:]
    chat = "[HISTORY STARTS] Here is the chat history:\n"
    for i, chat_his in enumerate(chat_history):
        chat += f"User: {chat_his['question']}\n"
        chat += f"AI: {chat_his['answer']}\n"

    chat += "[HISTORY ENDS]"
    
    return chat


def get_vectorized_of_pdfs(
        directory,
        target_docs = "all",
        project_name = "literature_review",
        reindex = False,
    ):

    """
    :param directory: directory containing PDFs
    :param target_doc: list of strings to filter the PDFs to vectorize
    :param project_name: name of the project
    :param reindex: whether to reindex the documents

    :return: vectorstore containing the vectorized documents
    """

    base_direc = "projects"
    os.makedirs(base_direc, exist_ok=True)
    project_path = os.path.join(base_direc, project_name)


    # This is the case when we have already vectorized the documents
    if os.path.exists(project_path) and not reindex and target_docs == "all":
        x = FAISS.load_local(
            project_path,
            embeddings=OpenAIEmbeddings(),
            allow_dangerous_deserialization=True
            )
        return x

    
    # Iterate over all PDFs in the directory recursively in all folders
    raw_documents = get_documents_raw_text(directory, target_docs)

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

    # Remove the HISTORY STARTS and HISTORY ENDS and anything in between from
    # the question and answer
    question = remove_between_tags(question, "[HISTORY STARTS]", "[HISTORY ENDS]")

    markdown = f"""
    
## Question
{question}

## Answer
--------------------------------------------------------------------------------
{answer}
--------------------------------------------------------------------------------

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


def save_markdown(markdown, filename="output.md", append=False):

    # Ensure the file exists, create it if it doesn't
    if not os.path.exists(filename):
        open(filename, 'w', encoding="utf-8").close()

    existing_content = ""

    # Read existing content
    if append:
        try:
            with open(filename, 'r', encoding="utf-8") as file:
                existing_content = file.read()
        except UnicodeDecodeError:
            # If utf-8 fails, try with a different encoding
            with open(filename, 'r', encoding="iso-8859-1") as file:
                existing_content = file.read()

    # Write new content
    with open(filename, 'w', encoding="utf-8") as file:
        file.write(markdown + existing_content)

    print(f"Markdown saved to {filename}")


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
    
def get_documents_raw_text(directory, target_doc):

    raw_documents = []
    n_pdfs = 0
    for root, dirs, files in os.walk(directory):
        for filename in files:
            
            if target_doc != "all" and \
                not any(doc in filename for doc in target_doc):
                continue

            if filename.endswith('.pdf'):
                
                print (filename, "is being processed", end="\r")

                pdf_path = os.path.join(root, filename)
                loader = PyPDFLoader(pdf_path)
                raw_documents.extend(loader.load())

                n_pdfs += 1
    
    if n_pdfs == 0:
        raise ValueError("No PDFs found.")
    
    return raw_documents


    







# ---------------------------------------------------------------------------- #
#                       Utils for HER model                                    #
# ---------------------------------------------------------------------------- #

def generate_response(query, context, role, model = "gpt-4o"):

    if model == "gpt-4o":
        client = OpenAI(
            api_key=os.getenv("API_KEY"),
        )
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": role},
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
            ]
        )
        return response.choices[0].message.content
    
    elif model == "anthropic":
        client = anthropic.Anthropic(
            api_key=os.getenv("ANTHROPIC_API_KEY"),
        )
        message = client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=4096,
            messages=[
                {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
            ]
        )

        # Extract text from TextBlock
        if isinstance(message.content[0], TextBlock):
            return message.content[0].text
        else:
            return str(message.content[0])
        

def get_context_for_her():

    # Open a markdown file and write the response
    file_name = "Her.md"
    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            f.write("")
        context = ""

    with open(file_name, "r") as f:
        context = f.read()

    return context

def save_response_to_her(response, file_name="Her.md"):
    with open(file_name, "a") as f:
        f.write(response + "\n\n")

    print(f"Response saved to {file_name}")