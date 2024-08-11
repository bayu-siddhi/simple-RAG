import pandas as pd
from simple_RAG.llm_model import LLM
from simple_RAG.helper import Helper
from simple_RAG.source import Source
from simple_RAG.retrieval import Retrieval
from simple_RAG.embedding import Embedding
from simple_RAG.query_config import QueryConfig
from simple_RAG.embedding_model import EmbeddingModel


class RAGPipeline:
    def __init__(
            self,
            device: str,
            llm_model_name_or_path: str,
            embedding_model_name_or_path: str,
            use_quantization_config: bool
    ) -> None:

        self.device: str = Helper.set_device(device, self.__class__.__name__)
        self.llm_model_name_or_path: str = llm_model_name_or_path
        self.embedding_model_name_or_path: str | None = None
        self.use_quantization_config: bool = use_quantization_config
        self.embeddings: Embedding | None = None
        self.llm = LLM(
            self.llm_model_name_or_path,
            self.use_quantization_config,
            self.device
        )
        self.embedding_model = EmbeddingModel(
            model_name=embedding_model_name_or_path,
            device=self.device
        )

    def new_embeddings(self, pdf_path: str):
        source = Source(
            pdf_path=pdf_path,
            min_token_length_per_chunk=30
        )
        self.embeddings = self.embedding_model.encode(
            filtered_chunks_or_query=source.filtered_chunks
        )
        self.embeddings.save_to_csv(f"{pdf_path[:-4]}_{self.device}_embeddings.csv")

    def load_embeddings(self, csv_path: str):
        self.embeddings = Embedding(device=self.device)
        self.embeddings.load_from_csv(csv_path=csv_path)

    def query(
            self,
            query: str,
            query_config: QueryConfig
    ) -> str:

        if query_config.use_context:
            query_embedding = self.embedding_model.encode(query)
            scores, indices, df_context = Retrieval.retrieve_context(
                query_embedding.embedding,
                self.embeddings.embedding,
                self.embeddings.df_embedding,
                query_config.top_k_sentence_chunks
            )
        else:
            df_context = None

        response = self.llm.generate_response(
            role=query_config.role,
            query=query,
            use_context=query_config.use_context,
            df_context=df_context,
            temperature=query_config.temperature,
            max_new_tokens=query_config.max_new_tokens
        )

        return response
