import re
import webbrowser
import markdown
from bs4 import BeautifulSoup
from markdown.extensions import Extension
from markdown.preprocessors import Preprocessor


def get_context(chat_history, w):
    """
    :param chat_history: List of chat history with Her, a dict with keys "query" and "response"
    :return: String with all the chat history
    """

    context = ""
    for i in range(max(0, len(chat_history) - w), len(chat_history)):
        context += f"User: {chat_history[i]['query']}\nHer: {chat_history[i]['response']}\n\n"

    return context


def save_response(chat_history, file_name, title="Talking with HER"):
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Her</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/themes/prism.min.css">
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                padding: 20px;
                background-color: #f9f9f9;
            }
            .container {
                max-width: 50vw;
                margin: 40px auto;
                padding: 20px;
                background-color: #fff;
                border: 1px solid #ddd;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
            }
            .chat {
                display: flex;
                flex-direction: column;
                gap: 10px;
            }
            .message {
                max-width: 70%;
                padding: 10px;
                border-radius: 10px;
                position: relative;
                font-size: 16px;
                margin: 5px !important;
            }
            .query {
                background-color: #dcf8c6;
                align-self: flex-end;
            }
            .response {
                background-color: #ececec;
                align-self: flex-start;
            }
            .message::after {
                content: '';
                position: absolute;
                width: 0;
                height: 0;
            }
            .query::after {
                border-left: 10px solid transparent;
                border-right: 10px solid #dcf8c6;
                border-top: 10px solid #dcf8c6;
                bottom: -10px;
                right: 10px;
            }
            .response::after {
                border-left: 10px solid #ececec;
                border-right: 10px solid transparent;
                border-top: 10px solid #ececec;
                bottom: -10px;
                left: 10px;
            }
            pre[class*="language-"] {
                position: relative;
                border-radius: 5px;
                padding: 15px;
                margin: 20px 0;
                overflow: auto;
            }
            code[class*="language-"] {
                font-family: Consolas, Monaco, 'Andale Mono', 'Ubuntu Mono', monospace;
                font-size: 14px;
            }
            .copy-button {
                position: absolute;
                top: 10px;
                right: 10px;
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 5px 10px;
                cursor: pointer;
                border-radius: 5px;
                font-size: 12px;
            }
        </style>
    </head>
    <body>
        <div class="container">
    """

    html_content += f"""
    <h1>{title}</h1>
    """

    for result in chat_history:
        query = result["query"]
        query = clean_response_for_code(query)


        response = result["response"]
        response = clean_response_for_code(response)

        html_content += f'''
        <div class="chat">
            <div class="message query">{query}</div>
            <div class="message response">{response}</div>
        </div>
        '''

    html_content += """
        </div>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/components/prism-core.min.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/plugins/autoloader/prism-autoloader.min.js"></script>
        <script>
            document.addEventListener('DOMContentLoaded', (event) => {
                document.querySelectorAll('pre').forEach((block) => {
                    const button = document.createElement('button');
                    button.innerText = 'Copy';
                    button.className = 'copy-button';
                    block.appendChild(button);
    
                    button.addEventListener('click', () => {
                        const code = block.querySelector('code').innerText;
                        navigator.clipboard.writeText(code).then(() => {
                            button.innerText = 'Copied';
                            setTimeout(() => {
                                button.innerText = 'Copy';
                            }, 2000);
                        });
                    });
                });
            });
        </script>
    </body>
    </html>
    """

    with open(file_name, "w") as file:
        file.write(html_content)

    print (f"Chat history saved to {file_name}")

    # open the html file using Chrome
    webbrowser.open(file_name)


def clean_response_for_code(response):

    to_return = ""
    for lang, res in split_code_and_text(response):

        # First, we'll preserve the numbering by replacing it with HTML
        res = re.sub(r'^(\d+)\.\s+\*\*(.+?)\*\*:', r'<h3>\1. \2</h3>', res, flags=re.MULTILINE)
        res = re.sub(r'^(\d+)\.\s+\*\*(.+?):\*\*', r'<h3>\1. \2</h3>', res, flags=re.MULTILINE)

        # Convert markdown to HTML
        res = markdown.markdown(res, extensions=['tables', 'fenced_code', 'pymdownx.superfences'])
        soup = BeautifulSoup(res, 'html.parser')

        if lang in ['sh', 'bash']:
            # Put in a block and return
            to_return += f'<pre class="language-{lang}"><code class="language-{lang}">{res}</code></pre>'
            continue

        for pre in soup.find_all('pre'):
            code = pre.find('code')

            if code:

                if lang == 'plaintext':
                    continue

                # Add or update the language class
                pre['class'] = [cls for cls in pre.get('class', []) if not cls.startswith('language-')] + [
                    'language-' + lang]
                code['class'] = [cls for cls in code.get('class', []) if not cls.startswith('language-')] + [
                    'language-' + lang]

                print(f"Detected language: {lang}")

        to_return += str(soup)

    return to_return


def guess_language(code):

    # Simple language detection based on common patterns
    if re.search(r'\bdef\b.*:|class.*:', code):
        return 'python'
    elif re.search(r'(var|let|const).*=|function.*{', code):
        return 'javascript'
    elif re.search(r'#include|int main\(', code):
        return 'cpp'
    elif re.search(r'public\s+class\s+\w+|public\s+static\s+void\s+main', code):
        return 'java'
    elif re.search(r'\bfn\b|\blet\b.*=', code):
        return 'rust'
    elif re.search(r'\bpackage\b|\bfunc\s+\w+\(\)', code):
        return 'go'
    elif re.search(r'using\s+System;|public\s+class\s+\w+', code):
        return 'csharp'
    elif re.search(r'<\?php|\bfunction\s+\w+\(', code):
        return 'php'
    elif re.search(r'\bdef\b\s+\w+|class\s+\w+', code):
        return 'ruby'
    # Add more language detection patterns as needed
    return 'plaintext'


def split_code_and_text(response):

    holder = []
    while '```' in response:

        start_index = response.find('```')
        end_index = response.find('```', start_index + 3)

        holder.append(("plaintext", response[:start_index]))

        lang = response[start_index + 3:].split('\n')[0]

        if lang in ['sh', 'bash']:
            code = response[start_index + 3 + len(lang) + 1: end_index]
        else:
            code = response[start_index: end_index+3]

        holder.append((lang, code))

        response = response[end_index + 3:]

    holder.append(("plaintext", response))

    return holder