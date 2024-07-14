from simple_RAG.helper import Helper
from simple_RAG.embedding import Embedding
from sentence_transformers import SentenceTransformer


class EmbeddingModel:

    def __init__(self, model_name: str, device: str) -> None:
        self.device = Helper.set_device(device, self.__class__.__name__)
        self.model_name = model_name
        # Raise OSError or ValueError
        self.model = SentenceTransformer(
            model_name_or_path=self.model_name,
            device=self.device
        )

    def encode(self, filtered_chunks_or_query: list[dict] | str) -> Embedding:
        if isinstance(filtered_chunks_or_query, list):
            for item in filtered_chunks_or_query:
                item['embedding'] = self.model.encode(item['chunk'])
        elif isinstance(filtered_chunks_or_query, str):
            filtered_chunks_or_query = self.model.encode(filtered_chunks_or_query, convert_to_tensor=True)
        embedding: Embedding = Embedding(self.device, filtered_chunks_or_query)
        return embedding
