import unittest
import pandas as pd
from simple_RAG import QueryConfig
from simple_RAG import RAGPipeline
from tests.variable import Variable


class RAGPipelineTestCase(unittest.TestCase):

    def setUp(self):
        self.variable = Variable()

    def test_create_cpu_rag_pipeline_success(self):
        rag = RAGPipeline(
            self.variable.device_cpu,
            self.variable.llm_model_success,
            self.variable.embedding_model_success,
            self.variable.use_quantization_config_false
        )

        self.assertEqual(rag.device, self.variable.device_cpu)
        self.assertEqual(rag.llm_model_name_or_path, self.variable.llm_model_success)
        self.assertEqual(rag.use_quantization_config, self.variable.use_quantization_config_false)

        self.assertEqual(rag.llm.device, self.variable.device_cpu)
        self.assertEqual(rag.llm.model_name_or_path, self.variable.llm_model_success)
        self.assertEqual(rag.llm.use_quantization_config, self.variable.use_quantization_config_false)
        self.assertIsNotNone(rag.llm.model)
        self.assertIsNotNone(rag.llm.tokenizer)

    def test_create_cuda_rag_pipeline_without_quantization_config_success(self):
        rag = RAGPipeline(
            self.variable.device_cuda,
            self.variable.llm_model_success,
            self.variable.embedding_model_success,
            self.variable.use_quantization_config_false
        )

        self.assertEqual(rag.device, self.variable.device_cuda)
        self.assertEqual(rag.llm_model_name_or_path, self.variable.llm_model_success)
        self.assertEqual(rag.use_quantization_config, self.variable.use_quantization_config_false)

        self.assertEqual(rag.llm.device, self.variable.device_cuda)
        self.assertEqual(rag.llm.model_name_or_path, self.variable.llm_model_success)
        self.assertEqual(rag.llm.use_quantization_config, self.variable.use_quantization_config_false)
        self.assertIsNotNone(rag.llm.model)
        self.assertIsNotNone(rag.llm.tokenizer)

    def test_create_cuda_rag_pipeline_with_quantization_config_success(self):
        rag = RAGPipeline(
            self.variable.device_cuda,
            self.variable.llm_model_success,
            self.variable.embedding_model_success,
            self.variable.use_quantization_config_true
        )

        self.assertEqual(rag.device, self.variable.device_cuda)
        self.assertEqual(rag.llm_model_name_or_path, self.variable.llm_model_success)
        self.assertEqual(rag.use_quantization_config, self.variable.use_quantization_config_true)

        self.assertEqual(rag.llm.device, self.variable.device_cuda)
        self.assertEqual(rag.llm.model_name_or_path, self.variable.llm_model_success)
        self.assertEqual(rag.llm.use_quantization_config, self.variable.use_quantization_config_true)
        self.assertIsNotNone(rag.llm.model)
        self.assertIsNotNone(rag.llm.tokenizer)

    def test_create_rag_pipeline_llm_model_not_found(self):
        with self.assertRaises(OSError):
            rag = RAGPipeline(
                self.variable.device_cpu,
                self.variable.llm_model_failed,
                self.variable.embedding_model_success,
                self.variable.use_quantization_config_false
            )

    def test_create_rag_pipeline_embedding_model_not_found(self):
        with self.assertRaises(OSError):
            rag = RAGPipeline(
                self.variable.device_cpu,
                self.variable.llm_model_success,
                self.variable.embedding_model_failed,
                self.variable.use_quantization_config_false
            )

    def test_create_new_cpu_embeddings_success(self):
        rag = RAGPipeline(
            self.variable.device_cuda,
            self.variable.llm_model_success,
            self.variable.embedding_model_success,
            self.variable.use_quantization_config_false
        )

        rag.new_embeddings(self.variable.pdf_path_success)

        df = pd.read_csv(f"{self.variable.pdf_path_success[:-4]}_{self.variable.device_cpu}_embeddings.csv", sep=',')
        self.assertListEqual(list(df.columns), ['page_index', 'chunk', 'num_chunk_token', 'embedding'])
        self.assertEqual(len(df), len(rag.embeddings.embedding))

    def test_create_new_cuda_embeddings_success(self):
        rag = RAGPipeline(
            self.variable.device_cuda,
            self.variable.llm_model_success,
            self.variable.embedding_model_success,
            self.variable.use_quantization_config_true
        )

        rag.new_embeddings(self.variable.pdf_path_success)

        df = pd.read_csv(f"{self.variable.pdf_path_success[:-4]}_{self.variable.device_cuda}_embeddings.csv", sep=',')
        self.assertListEqual(list(df.columns), ['page_index', 'chunk', 'num_chunk_token', 'embedding'])
        self.assertEqual(len(df), len(rag.embeddings.embedding))

    def test_create_new_embeddings_pdf_not_found(self):
        rag = RAGPipeline(
            self.variable.device_cpu,
            self.variable.llm_model_success,
            self.variable.embedding_model_success,
            self.variable.use_quantization_config_false
        )

        with self.assertRaises(FileNotFoundError):
            rag.new_embeddings(self.variable.pdf_path_failed)

    def test_load_cpu_embeddings_success(self):
        rag = RAGPipeline(
            self.variable.device_cpu,
            self.variable.llm_model_success,
            self.variable.embedding_model_success,
            self.variable.use_quantization_config_false
        )

        rag.load_embeddings(self.variable.embedding_csv_cpu_success)

        df = pd.read_csv(f"{self.variable.pdf_path_success[:-4]}_{self.variable.device_cpu}_embeddings.csv", sep=',')
        self.assertListEqual(list(df.columns), ['page_index', 'chunk', 'num_chunk_token', 'embedding'])
        self.assertEqual(len(df), len(rag.embeddings.embedding))

    def test_load_cuda_embeddings_success(self):
        rag = RAGPipeline(
            self.variable.device_cuda,
            self.variable.llm_model_success,
            self.variable.embedding_model_success,
            self.variable.use_quantization_config_true
        )

        rag.load_embeddings(self.variable.embedding_csv_cuda_success)

        df = pd.read_csv(f"{self.variable.pdf_path_success[:-4]}_{self.variable.device_cuda}_embeddings.csv", sep=',')
        self.assertListEqual(list(df.columns), ['page_index', 'chunk', 'num_chunk_token', 'embedding'])
        self.assertEqual(len(df), len(rag.embeddings.embedding))

    def test_load_embeddings_not_found(self):
        rag = RAGPipeline(
            self.variable.device_cpu,
            self.variable.llm_model_success,
            self.variable.embedding_model_success,
            self.variable.use_quantization_config_false
        )

        with self.assertRaises(FileNotFoundError):
            rag.load_embeddings(self.variable.embedding_csv_failed)

    def test_ask_cpu_rag_pipeline_without_context_success(self):
        rag = RAGPipeline(
            self.variable.device_cpu,
            self.variable.llm_model_success,
            self.variable.embedding_model_success,
            self.variable.use_quantization_config_false
        )

        query_config = QueryConfig(
            self.variable.role,
            self.variable.temperature,
            self.variable.max_new_tokens,
            self.variable.use_context_false,
            self.variable.n_chunks
        )

        response, context = rag.query(
            self.variable.query,
            query_config
        )

        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        self.assertIsNone(context)

    def test_ask_cuda_rag_pipeline_using_quantization_config_without_context_success(self):
        rag = RAGPipeline(
            self.variable.device_cpu,
            self.variable.llm_model_success,
            self.variable.embedding_model_success,
            self.variable.use_quantization_config_true
        )

        query_config = QueryConfig(
            self.variable.role,
            self.variable.temperature,
            self.variable.max_new_tokens,
            self.variable.use_context_false,
            self.variable.n_chunks
        )

        response, context = rag.query(
            self.variable.query,
            query_config
        )

        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        self.assertIsNone(context)

    def test_ask_cpu_rag_pipeline_with_context_success(self):
        rag = RAGPipeline(
            self.variable.device_cpu,
            self.variable.llm_model_success,
            self.variable.embedding_model_success,
            self.variable.use_quantization_config_false
        )

        rag.load_embeddings(self.variable.embedding_csv_cpu_success)

        query_config = QueryConfig(
            self.variable.role,
            self.variable.temperature,
            self.variable.max_new_tokens,
            self.variable.use_context_true,
            self.variable.n_chunks
        )

        response, context = rag.query(
            self.variable.query,
            query_config
        )

        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        self.assertIsInstance(context, pd.DataFrame)
        self.assertEqual(len(context), self.variable.n_chunks)

    def test_ask_cuda_rag_pipeline_using_quantization_config_with_context_success(self):
        rag = RAGPipeline(
            self.variable.device_cuda,
            self.variable.llm_model_success,
            self.variable.embedding_model_success,
            self.variable.use_quantization_config_true
        )

        rag.load_embeddings(self.variable.embedding_csv_cuda_success)

        query_config = QueryConfig(
            self.variable.role,
            self.variable.temperature,
            self.variable.max_new_tokens,
            self.variable.use_context_true,
            self.variable.n_chunks
        )

        response, context = rag.query(
            self.variable.query,
            query_config
        )

        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        self.assertIsInstance(context, pd.DataFrame)
        self.assertEqual(len(context), self.variable.n_chunks)


if __name__ == '__main__':
    unittest.main()
