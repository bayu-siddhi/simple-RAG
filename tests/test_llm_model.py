import unittest
from simple_RAG import LLM
from simple_RAG import Retrieval
from simple_RAG import Embedding
from simple_RAG import EmbeddingModel


class LLMTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.device = 'cuda'
        self.use_quantization_config = True
        self.llm_model_success = './google/gemma-2b-it'
        self.llm_model_failed = './google/not-found'
        self.embedding_csv = 'data/test_embeddings.csv'
        self.embedding_model_name = 'all-mpnet-base-v2'

        self.role = 'user'
        self.top_k_chunks = 2
        self.temperature = 0.7
        self.max_new_tokens = 128
        self.query = 'What is Retrieval Augmented Generation?'
        self.format_response_text = True

    def test_connect_to_llm_model_success(self) -> None:
        """Test connect to LLM model using quantization config and cuda (success)"""
        llm = LLM(self.llm_model_success, self.use_quantization_config, self.device)
        self.assertEqual(llm.device, self.device)
        self.assertEqual(llm.model_name_or_path, self.llm_model_success)
        self.assertEqual(llm.use_quantization_config, self.use_quantization_config)
        self.assertIsNotNone(llm.model)
        self.assertIsNotNone(llm.tokenizer)

    def test_connect_to_llm_model_failed(self) -> None:
        """Test connect to not found LLM model (failed)"""
        with self.assertRaises(OSError):
            llm = LLM(self.llm_model_failed, self.use_quantization_config, self.device)

    def test_generate_response_without_context(self) -> None:
        """Test generate response without context using LLM model on maximum of 128 new tokens"""
        llm = LLM(self.llm_model_success, self.use_quantization_config, self.device)
        response = llm.generate_response(
            role=self.role,
            query=self.query,
            use_context=False,
            df_context=None,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            format_response_text=self.format_response_text
        )

        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_generate_response_with_context(self) -> None:
        """Test generate response with context using LLM model on maximum of 128 new tokens"""
        llm = LLM(self.llm_model_success, self.use_quantization_config, self.device)

        embeddings = Embedding(self.device)
        embeddings.load_from_csv(self.embedding_csv)

        embedding_model = EmbeddingModel(self.embedding_model_name, self.device)
        query_embedding = embedding_model.encode(self.query)

        scores, indices, df_context = Retrieval.retrieve_context(
            query_embedding.embedding,
            embeddings.embedding,
            embeddings.df_embedding,
            self.top_k_chunks
        )

        response = llm.generate_response(
            role=self.role,
            query=self.query,
            use_context=True,
            df_context=df_context,
            temperature=self.temperature,
            max_new_tokens=self.max_new_tokens,
            format_response_text=self.format_response_text
        )

        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)


if __name__ == '__main__':
    unittest.main()
