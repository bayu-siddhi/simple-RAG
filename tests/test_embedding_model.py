import torch
import unittest
import pandas as pd
from simple_RAG import Source
from simple_RAG import Embedding
from simple_RAG import EmbeddingModel


class EmbeddingModelTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.query = 'What is RAG?'
        self.pdf_path = 'data/test.pdf'
        self.embedding_model_success = 'all-mpnet-base-v2'
        self.embedding_model_failed = 'not-found-model'

    def test_get_embedding_model_success(self) -> None:
        """Test loading embedding model (success)"""
        embedding_model = EmbeddingModel(self.embedding_model_success)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.assertEqual(embedding_model.device, device)
        self.assertEqual(embedding_model.model_name, self.embedding_model_success)
        self.assertIsNotNone(embedding_model.model)

    def test_get_embedding_model_failed(self) -> None:
        """Test loading embedding model (not found)"""
        with self.assertRaises(OSError):
            EmbeddingModel(self.embedding_model_failed)

    def test_encode_query_with_cpu(self) -> None:
        """Test encode query (user question) with cpu (success)"""
        embedding_model = EmbeddingModel(self.embedding_model_success)
        embedding = embedding_model.encode(self.query, 'cpu')

        self.assertEqual(embedding_model.device, 'cpu')
        self.assertIsInstance(embedding, Embedding)
        self.assertIsInstance(embedding.embedding, torch.Tensor)
        self.assertIsInstance(embedding.df_embedding, pd.DataFrame)
        self.assertGreater(len(embedding.embedding), 0)
        self.assertEqual(len(embedding.df_embedding), 0)

    def test_encode_query_with_cuda(self) -> None:
        """Test encode query (user question) with cuda (success)"""
        embedding_model = EmbeddingModel(self.embedding_model_success)
        embedding = embedding_model.encode(self.query, 'cuda')

        self.assertEqual(embedding_model.device, 'cuda')
        self.assertIsInstance(embedding, Embedding)
        self.assertIsInstance(embedding.embedding, torch.Tensor)
        self.assertIsInstance(embedding.df_embedding, pd.DataFrame)
        self.assertGreater(len(embedding.embedding), 0)
        self.assertEqual(len(embedding.df_embedding), 0)

    def test_encode_chunks_with_cpu(self) -> None:
        """Test encode document chunks with cpu (success)"""
        source = Source(self.pdf_path)
        source.open_and_extract_pdf()
        embedding_model = EmbeddingModel(self.embedding_model_success)
        embedding = embedding_model.encode(source.filtered_chunks, 'cpu')

        self.assertEqual(embedding_model.device, 'cpu')
        self.assertIsInstance(embedding, Embedding)
        self.assertEqual(len(embedding.embedding), len(embedding.df_embedding))
        self.assertIsInstance(embedding.embedding, torch.Tensor)
        self.assertIsInstance(embedding.df_embedding, pd.DataFrame)

    def test_encode_chunks_with_cuda(self) -> None:
        """Test encode document chunks with cuda (success)"""
        source = Source(self.pdf_path)
        source.open_and_extract_pdf()
        embedding_model = EmbeddingModel(self.embedding_model_success)
        embedding = embedding_model.encode(source.filtered_chunks, 'cuda')

        self.assertEqual(embedding_model.device, 'cuda')
        self.assertIsInstance(embedding, Embedding)
        self.assertEqual(len(embedding.embedding), len(embedding.df_embedding))
        self.assertIsInstance(embedding.embedding, torch.Tensor)
        self.assertIsInstance(embedding.df_embedding, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
