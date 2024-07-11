from simple_RAG.helper import Helper
from simple_RAG.embedding import Embedding
from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(self, model_name: str) -> None:
        self.device = Helper.set_device('cuda')
        self.model_name = model_name
        # Raise OSError or ValueError
        self.model = SentenceTransformer(
            model_name_or_path=self.model_name,
            device=self.device
        )

    def encode(self, filtered_chunks: list[dict], device: str = None) -> Embedding:
        self.device = self.device if device is None else Helper.set_device(device)
        self.model.to(self.device)
        for item in filtered_chunks:
            item['embedding'] = self.model.encode(item['chunk'])
            # item['embedding'] = self.model.encode(item['chunk'], convert_to_tensor=True)
        embedding: Embedding = Embedding()
        embedding.add_new(filtered_chunks)
        return embedding
