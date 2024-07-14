import torch
import pandas as pd


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
