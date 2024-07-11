import torch


class Helper:

    @staticmethod
    def text_formatter(text: str) -> str:
        return text.replace("\n", " ").strip()

    @staticmethod
    def set_device(device: str) -> str:
        return 'cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu'
