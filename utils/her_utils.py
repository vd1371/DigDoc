import os

from openai import OpenAI
import anthropic
from anthropic.types import TextBlock

from dotenv import load_dotenv

from .utils import get_context, save_response

load_dotenv()

class Her:

    def __init__(self, model="gpt-4o"):
        self.model = model
        self.file_name = "TalkWithHer.html"
        self.chat_history = []
        self.look_back_window = 3

    def ask(self, query, role=None):
        if role is None:
            role = "You are a general assistant helping with various tasks."

        context = get_context(self.chat_history, self.look_back_window)

        if self.model == "gpt-4o":
            client = OpenAI(
                api_key=os.getenv("API_KEY"),
            )
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": role},
                    {"role": "user",
                     "content": f"Context: {context}\n\nQuestion: {query}"}
                ]
            )
            response = response.choices[0].message.content

        elif self.model == "anthropic":
            client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY"),
            )
            message = client.messages.create(
                model="claude-3-5-sonnet-20240620",
                max_tokens=4096,
                messages=[
                    {"role": "user",
                     "content": f"Context: {context}\n\nQuestion: {query}"}
                ]
            )

            # Extract text from TextBlock
            if isinstance(message.content[0], TextBlock):
                response = message.content[0].text
            else:
                response = str(message.content[0])

        self.chat_history.append({
            "query": query,
            "response": response,
        })

        save_response(self.chat_history, self.file_name)

        return response
