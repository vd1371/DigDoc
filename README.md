# DigDoc

## Introduction
DigDoc is my personal Python library designed to streamline the process of document ingestion and analysis. It supports various document formats including PDF, EPUB, MOBI, DOCX, and Jupyter Notebooks (.ipynb). The library leverages powerful models like GPT-4 and tools from LangChain for text processing, embedding, and retrieval.

## How to Use
0. **Setup**: Add the API_KEY for OpenAI and Anthropic in the `.env` file. Create one if there's no `.env` file in the root directory.
    ```bash
    OPENAI_API_KEY=YOUR_API_KEY
    ANTHROPIC_API_KEY=YOUR_API_KEY
    ```

1. **Initialization**: Start by initializing the DigDoc class with the desired document directory and model.
    ```python
    from utils.digdoc_utils import DigDoc

    digdoc = DigDoc(
        docs_directory="/path/to/your/documents",
        model="gpt-4o",
        project_name="YourProjectName"
    )
    ```

2. **Reading Documents**: Load the documents you want to process by specifying the file types.
    ```python
    digdoc.read_docs(
        target_docs=[
            ".py", ".ipynb", ".pdf", ".epub", ".mobi", ".docx"
        ],
        reindex=False
    )
    ```

3. **Querying**: Use the `digdoc.answer` method to ask questions or retrieve information from the loaded documents.
    ```python
    query = "Your query here"
    answer, source_docs = digdoc.answer(query, refresh_history=True)
    print(f"Answer: {answer}")
    ```


## License

This project is licensed under the MIT License. See the [LICENSE](https://github.com/vd1371/LiteratureReviewRAG/blob/main/LICENSE) file for details.