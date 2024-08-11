import unittest
from simple_RAG import LLM
from simple_RAG import Retrieval
from simple_RAG import Embedding
from simple_RAG import EmbeddingModel
from tests.variable import Variable


class LLMTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.variable = Variable()

    def test_connect_to_cpu_llm_model_success(self) -> None:
        """Test connect to LLM model using cpu (success)"""
        llm = LLM(self.variable.llm_model_success, self.variable.use_quantization_config_false, self.variable.device_cpu)
        self.assertEqual(llm.device, self.variable.device_cpu)
        self.assertEqual(llm.model_name_or_path, self.variable.llm_model_success)
        self.assertEqual(llm.use_quantization_config, self.variable.use_quantization_config_false)
        self.assertIsNotNone(llm.model)
        self.assertIsNotNone(llm.tokenizer)

    def test_connect_to_cuda_llm_model_with_quantization_config_success(self) -> None:
        """Test connect to LLM model with quantization config and cuda (success)"""
        llm = LLM(self.variable.llm_model_success, self.variable.use_quantization_config_true, self.variable.device_cuda)
        self.assertEqual(llm.device, self.variable.device_cuda)
        self.assertEqual(llm.model_name_or_path, self.variable.llm_model_success)
        self.assertEqual(llm.use_quantization_config, self.variable.use_quantization_config_true)
        self.assertIsNotNone(llm.model)
        self.assertIsNotNone(llm.tokenizer)

    def test_connect_to_cuda_llm_model_without_quantization_config_success(self) -> None:
        """Test connect to LLM model without quantization config and cuda (success)"""
        llm = LLM(self.variable.llm_model_success, self.variable.use_quantization_config_false, self.variable.device_cuda)
        self.assertEqual(llm.device, self.variable.device_cuda)
        self.assertEqual(llm.model_name_or_path, self.variable.llm_model_success)
        self.assertEqual(llm.use_quantization_config, self.variable.use_quantization_config_false)
        self.assertIsNotNone(llm.model)
        self.assertIsNotNone(llm.tokenizer)

    def test_connect_to_llm_model_failed(self) -> None:
        """Test connect to not found LLM model (failed)"""
        with self.assertRaises(OSError):
            llm = LLM(self.variable.llm_model_failed, self.variable.use_quantization_config_false, self.variable.device_cpu)

    def test_generate_response_without_context_using_cpu(self) -> None:
        """Test generate response without context using cpu LLM model on maximum of 128 new tokens"""
        llm = LLM(self.variable.llm_model_success, self.variable.use_quantization_config_false, self.variable.device_cpu)
        response = llm.generate_response(
            role=self.variable.role,
            query=self.variable.query,
            use_context=False,
            df_context=None,
            temperature=self.variable.temperature,
            max_new_tokens=self.variable.max_new_tokens
        )

        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_generate_response_without_context_using_cuda(self) -> None:
        """
        Test generate response without context using cuda LLM model with quantization config
        on maximum of 128 new tokens
        """
        llm = LLM(self.variable.llm_model_success, self.variable.use_quantization_config_true, self.variable.device_cuda)
        response = llm.generate_response(
            role=self.variable.role,
            query=self.variable.query,
            use_context=False,
            df_context=None,
            temperature=self.variable.temperature,
            max_new_tokens=self.variable.max_new_tokens
        )

        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_generate_response_with_context_using_cpu(self) -> None:
        """Test generate response with context using cpu LLM model on maximum of 128 new tokens"""
        llm = LLM(self.variable.llm_model_success, self.variable.use_quantization_config_false, self.variable.device_cpu)

        embeddings = Embedding(self.variable.device_cpu)
        embeddings.load_from_csv(self.variable.embedding_csv_cpu_success)

        embedding_model = EmbeddingModel(self.variable.embedding_model_success, self.variable.device_cpu)
        query_embedding = embedding_model.encode(self.variable.query)

        scores, indices, df_context = Retrieval.retrieve_context(
            query_embedding.embedding,
            embeddings.embedding,
            embeddings.df_embedding,
            self.variable.n_chunks
        )

        response = llm.generate_response(
            role=self.variable.role,
            query=self.variable.query,
            use_context=True,
            df_context=df_context,
            temperature=self.variable.temperature,
            max_new_tokens=self.variable.max_new_tokens
        )

        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_generate_response_with_context_using_cuda(self) -> None:
        """
        Test generate response with context using cuda LLM model with quantization config
        on maximum of 128 new tokens
        """
        llm = LLM(self.variable.llm_model_success, self.variable.use_quantization_config_true, self.variable.device_cuda)

        embeddings = Embedding(self.variable.device_cuda)
        embeddings.load_from_csv(self.variable.embedding_csv_cuda_success)

        embedding_model = EmbeddingModel(self.variable.embedding_model_success, self.variable.device_cuda)
        query_embedding = embedding_model.encode(self.variable.query)

        scores, indices, df_context = Retrieval.retrieve_context(
            query_embedding.embedding,
            embeddings.embedding,
            embeddings.df_embedding,
            self.variable.n_chunks
        )

        response = llm.generate_response(
            role=self.variable.role,
            query=self.variable.query,
            use_context=True,
            df_context=df_context,
            temperature=self.variable.temperature,
            max_new_tokens=self.variable.max_new_tokens
        )

        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)


if __name__ == '__main__':
    unittest.main()
