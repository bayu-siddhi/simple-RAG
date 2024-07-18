import unittest
import pandas as pd
from simple_RAG import QueryConfig
from simple_RAG import RAGPipeline
from simple_RAG import FileNotFoundException


class RAGPipelineTestCase(unittest.TestCase):

    def setUp(self):
        self.device = 'cuda'
        self.llm_model_name_or_path = './google/gemma-2b-it'
        self.llm_model_name_or_path_failed = './google/not-found'
        self.embedding_model_name_or_path = 'all-mpnet-base-v2'
        self.embedding_model_name_or_path_failed = 'not-found'
        self.use_quantization_config = True
        self.pdf_path = './data/test.pdf'
        self.pdf_path_failed = './data/not_found.pdf'
        self.embeddings_path = './data/test_embeddings.csv'
        self.embeddings_path_failed = './data/not_found_embeddings.csv'

        self.role = 'user'
        self.query = 'what is Retrieval Augmented Generation?'
        self.temperature = 0.7
        self.max_new_tokens = 256
        self.use_context_true = True
        self.use_context_false = False
        self.top_k_sentence_chunks = 2
        self.format_answer_text = True

    def test_create_rag_pipeline_success(self):
        rag = RAGPipeline(
            self.device,
            self.llm_model_name_or_path,
            self.embedding_model_name_or_path,
            self.use_quantization_config
        )

        self.assertEqual(rag.device, self.device)
        self.assertEqual(rag.llm_model_name_or_path, self.llm_model_name_or_path)
        self.assertEqual(rag.use_quantization_config, self.use_quantization_config)

        self.assertEqual(rag.llm.device, self.device)
        self.assertEqual(rag.llm.model_name_or_path, self.llm_model_name_or_path)
        self.assertEqual(rag.llm.use_quantization_config, self.use_quantization_config)
        self.assertIsNotNone(rag.llm.model)
        self.assertIsNotNone(rag.llm.tokenizer)

    def test_create_rag_pipeline_llm_model_not_found(self):
        with self.assertRaises(OSError):
            rag = RAGPipeline(
                self.device,
                self.llm_model_name_or_path_failed,
                self.embedding_model_name_or_path,
                self.use_quantization_config
            )

    def test_create_rag_pipeline_embedding_model_not_found(self):
        with self.assertRaises(OSError):
            rag = RAGPipeline(
                self.device,
                self.llm_model_name_or_path,
                self.embedding_model_name_or_path_failed,
                self.use_quantization_config
            )

    def test_create_new_embeddings_for_rag_pipeline_success(self):
        rag = RAGPipeline(
            self.device,
            self.llm_model_name_or_path,
            self.embedding_model_name_or_path,
            self.use_quantization_config
        )

        rag.new_embeddings(self.pdf_path)

        df = pd.read_csv(f"{self.pdf_path[:-4]}_embeddings.csv", sep=',')
        self.assertListEqual(list(df.columns), ['page_index', 'chunk', 'num_chunk_token', 'embedding'])
        self.assertEqual(len(df), len(rag.embeddings.embedding))

    def test_create_new_embeddings_for_rag_pipeline_pdf_not_found(self):
        rag = RAGPipeline(
            self.device,
            self.llm_model_name_or_path,
            self.embedding_model_name_or_path,
            self.use_quantization_config
        )

        with self.assertRaises(FileNotFoundException):
            rag.new_embeddings(self.pdf_path_failed)

    def test_create_load_embeddings_for_rag_pipeline_success(self):
        rag = RAGPipeline(
            self.device,
            self.llm_model_name_or_path,
            self.embedding_model_name_or_path,
            self.use_quantization_config
        )

        rag.load_embeddings(self.embeddings_path)

        df = pd.read_csv(f"{self.pdf_path[:-4]}_embeddings.csv", sep=',')
        self.assertListEqual(list(df.columns), ['page_index', 'chunk', 'num_chunk_token', 'embedding'])
        self.assertEqual(len(df), len(rag.embeddings.embedding))

    def test_create_load_embeddings_for_rag_pipeline_embeddings_not_found(self):
        rag = RAGPipeline(
            self.device,
            self.llm_model_name_or_path,
            self.embedding_model_name_or_path,
            self.use_quantization_config
        )

        with self.assertRaises(FileNotFoundException):
            rag.load_embeddings(self.embeddings_path_failed)

    def test_ask_rag_pipeline_with_context_success(self):
        rag = RAGPipeline(
            self.device,
            self.llm_model_name_or_path,
            self.embedding_model_name_or_path,
            self.use_quantization_config
        )

        rag.load_embeddings(self.embeddings_path)

        query_config = QueryConfig(
            self.role,
            self.temperature,
            self.max_new_tokens,
            self.use_context_true,
            self.top_k_sentence_chunks,
            self.format_answer_text
        )

        response = rag.query(
            self.query,
            query_config
        )

        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_ask_rag_pipeline_without_context_success(self):
        rag = RAGPipeline(
            self.device,
            self.llm_model_name_or_path,
            self.embedding_model_name_or_path,
            self.use_quantization_config
        )

        query_config = QueryConfig(
            self.role,
            self.temperature,
            self.max_new_tokens,
            self.use_context_false,
            self.top_k_sentence_chunks,
            self.format_answer_text
        )

        response = rag.query(
            self.query,
            query_config
        )

        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)


if __name__ == '__main__':
    unittest.main()
