import torch


class Helper:

    @staticmethod
    def text_formatter(text: str) -> str:
        return text.replace("\n", " ").strip()

    @staticmethod
    def set_device(device: str, from_class: str, print_status: bool = True) -> str:
        device_used = 'cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu'
        if device == device_used:
            print(f"[INFO] {device} is available, using {device_used} as the {from_class} device")
        else:
            print(f"[INFO] {device} is not available, using {device_used} as the {from_class} device")
        return device_used

    @staticmethod
    def chat_template(role: str, content: str) -> list:
        return [{
            'role': role,
            'content': content
        }]

    @staticmethod
    def base_prompt(query: str, context: str) -> str:
        return (
            """Your name is Simple RAG.
Your job is to answer user questions based on some given context.
Here is a user question that you must answer.
User query: {query}
Prioritize answering in paragraph form.
Do not include the analysis process, just give the final answer.
Ensure your answers are as detailed as possible.
If the question asked is not in context, then answer that you don't know with an apology.
Now use the following context items to address the user's query:
{context}
Answer:
"""
        ).format(query=query, context=context)