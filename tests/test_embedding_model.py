import torch
import unittest
import pandas as pd
from simple_RAG import Source
from simple_RAG import Embedding
from simple_RAG import EmbeddingModel
from tests.variable import Variable


class EmbeddingModelTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.variable = Variable()

    def test_get_embedding_model_success(self) -> None:
        """Test loading embedding model (success)"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        embedding_model = EmbeddingModel(self.variable.embedding_model_success, device)
        self.assertEqual(embedding_model.device, device)
        self.assertEqual(embedding_model.model_name, self.variable.embedding_model_success)
        self.assertIsNotNone(embedding_model.model)

    def test_get_embedding_model_failed(self) -> None:
        """Test loading embedding model (not found)"""
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with self.assertRaises(OSError):
            EmbeddingModel(self.variable.embedding_model_failed, device)

    def test_encode_query_with_cpu(self) -> None:
        """Test encode query (user question) with cpu (success)"""
        embedding_model = EmbeddingModel(self.variable.embedding_model_success, self.variable.device_cpu)
        embedding = embedding_model.encode(self.variable.query)

        self.assertEqual(embedding_model.device, self.variable.device_cpu)
        self.assertIsInstance(embedding, Embedding)
        self.assertIsInstance(embedding.embedding, torch.Tensor)
        self.assertIsNone(embedding.df_embedding)
        self.assertGreater(len(embedding.embedding), 0)

    def test_encode_query_with_cuda(self) -> None:
        """Test encode query (user question) with cuda (success)"""
        embedding_model = EmbeddingModel(self.variable.embedding_model_success, self.variable.device_cuda)
        embedding = embedding_model.encode(self.variable.query)

        self.assertEqual(embedding_model.device, self.variable.device_cuda)
        self.assertIsInstance(embedding, Embedding)
        self.assertIsInstance(embedding.embedding, torch.Tensor)
        self.assertIsNone(embedding.df_embedding)
        self.assertGreater(len(embedding.embedding), 0)

    def test_encode_chunks_with_cpu(self) -> None:
        """Test encode document chunks with cpu (success)"""
        source = Source(self.variable.pdf_path_success)
        embedding_model = EmbeddingModel(self.variable.embedding_model_success, self.variable.device_cpu)
        embedding = embedding_model.encode(source.filtered_chunks)

        self.assertEqual(embedding_model.device, self.variable.device_cpu)
        self.assertIsInstance(embedding, Embedding)
        self.assertEqual(len(embedding.embedding), len(embedding.df_embedding))
        self.assertIsInstance(embedding.embedding, torch.Tensor)
        self.assertIsInstance(embedding.df_embedding, pd.DataFrame)

    def test_encode_chunks_with_cuda(self) -> None:
        """Test encode document chunks with cuda (success)"""
        source = Source(self.variable.pdf_path_success)
        embedding_model = EmbeddingModel(self.variable.embedding_model_success, self.variable.device_cuda)
        embedding = embedding_model.encode(source.filtered_chunks)

        self.assertEqual(embedding_model.device, self.variable.device_cuda)
        self.assertIsInstance(embedding, Embedding)
        self.assertEqual(len(embedding.embedding), len(embedding.df_embedding))
        self.assertIsInstance(embedding.embedding, torch.Tensor)
        self.assertIsInstance(embedding.df_embedding, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
