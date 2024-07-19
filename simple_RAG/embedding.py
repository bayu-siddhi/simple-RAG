import os
import torch
import numpy as np
import pandas as pd
from simple_RAG.helper import Helper
from simple_RAG.file_not_found_exception import FileNotFoundException


class Embedding:

    def __init__(self, device: str, embedding: list[dict] | torch.Tensor | None = None) -> None:
        self.device = Helper.set_device(device, self.__class__.__name__)
        self.embedding: torch.Tensor | None = None
        self.df_embedding: pd.DataFrame | None = None
        if embedding is not None:
            self.add_new(embedding)

    def add_new(self, embedding: list[dict] | torch.Tensor) -> None:
        if isinstance(embedding, list):
            self.__to_dataframe(embedding)
            self.__to_torch_tensor()
        elif isinstance(embedding, torch.Tensor):
            self.embedding = embedding

    def load_from_csv(self, csv_path: str) -> None:
        if not os.path.exists(csv_path):
            raise FileNotFoundException(f"File {csv_path} doesn't exist.")
        self.__to_dataframe(csv_path)
        self.__to_torch_tensor()

    def save_to_csv(self, filename: str) -> None:
        if filename.endswith('.csv'):
            self.df_embedding.to_csv(filename, index=False)
        else:
            self.df_embedding.to_csv(f"{filename}.csv", index=False)

    def __to_dataframe(self, embedding_or_csv_path: list[dict] | str) -> None:
        if isinstance(embedding_or_csv_path, list):
            self.df_embedding = pd.DataFrame(embedding_or_csv_path)
        elif isinstance(embedding_or_csv_path, str):
            self.df_embedding = pd.read_csv(embedding_or_csv_path)
            self.df_embedding['embedding'] = self.df_embedding['embedding'].apply(
                lambda x: np.fromstring(x.strip('[]'), sep=' ')
            )

    def __to_torch_tensor(self) -> None:
        """ Convert embeddings from numpy array to torch tensor and send to device """
        self.embedding = torch.tensor(
            np.array(self.df_embedding['embedding'].tolist()),
            dtype=torch.float32
        ).to(self.device)
