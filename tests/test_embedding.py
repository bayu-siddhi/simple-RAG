import torch
import unittest
import pandas as pd
from simple_RAG import Source
from simple_RAG import Embedding
from simple_RAG import EmbeddingModel
from tests.variable import Variable


class EmbeddingTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.variable = Variable()

    def test_create_and_save_new_cpu_embedding(self) -> None:
        """Test create new embedding from embedding_model.encode() and save to CSV (success)"""
        source = Source(self.variable.pdf_path_success)
        embedding_model = EmbeddingModel(self.variable.embedding_model_success, self.variable.device_cpu)
        embedding = embedding_model.encode(source.filtered_chunks)

        self.assertEqual(embedding_model.device, self.variable.device_cpu)
        self.assertIsInstance(embedding, Embedding)
        self.assertEqual(len(embedding.embedding), len(embedding.df_embedding))
        self.assertIsInstance(embedding.embedding, torch.Tensor)
        self.assertIsInstance(embedding.df_embedding, pd.DataFrame)

        embedding.save_to_csv(self.variable.embedding_csv_cpu_success)

        df = pd.read_csv(self.variable.embedding_csv_cpu_success, sep=',')
        self.assertListEqual(list(df.columns), ['page_index', 'chunk', 'num_chunk_token', 'embedding'])
        self.assertEqual(len(df), len(embedding.embedding))

    def test_create_and_save_new_cuda_embedding(self) -> None:
        """Test create new embedding from embedding_model.encode() and save to CSV (success)"""
        source = Source(self.variable.pdf_path_success)
        embedding_model = EmbeddingModel(self.variable.embedding_model_success, self.variable.device_cuda)
        embedding = embedding_model.encode(source.filtered_chunks)

        self.assertEqual(embedding_model.device, self.variable.device_cuda)
        self.assertIsInstance(embedding, Embedding)
        self.assertEqual(len(embedding.embedding), len(embedding.df_embedding))
        self.assertIsInstance(embedding.embedding, torch.Tensor)
        self.assertIsInstance(embedding.df_embedding, pd.DataFrame)

        embedding.save_to_csv(self.variable.embedding_csv_cuda_success)

        df = pd.read_csv(self.variable.embedding_csv_cuda_success, sep=',')
        self.assertListEqual(list(df.columns), ['page_index', 'chunk', 'num_chunk_token', 'embedding'])
        self.assertEqual(len(df), len(embedding.embedding))

    def test_load_cpu_embeddings_from_csv_success(self) -> None:
        """Test load embedding from CSV (success)"""
        embedding = Embedding(self.variable.device_cpu)
        embedding.load_from_csv(self.variable.embedding_csv_cpu_success)
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.assertEqual(embedding.device, self.variable.device_cpu)
        self.assertEqual(len(embedding.embedding), len(embedding.df_embedding))
        self.assertIsInstance(embedding.embedding, torch.Tensor)
        self.assertIsInstance(embedding.df_embedding, pd.DataFrame)

    def test_load_cuda_embeddings_from_csv_success(self) -> None:
        """Test load embedding from CSV (success)"""
        embedding = Embedding(self.variable.device_cuda)
        embedding.load_from_csv(self.variable.embedding_csv_cuda_success)
        # device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.assertEqual(embedding.device, self.variable.device_cuda)
        self.assertEqual(len(embedding.embedding), len(embedding.df_embedding))
        self.assertIsInstance(embedding.embedding, torch.Tensor)
        self.assertIsInstance(embedding.df_embedding, pd.DataFrame)

    def test_load_from_csv_failed(self) -> None:
        """Test load embedding from CSV not found (failed)"""
        with self.assertRaises(FileNotFoundError):
            embedding = Embedding(self.variable.device_cpu)
            embedding.load_from_csv(self.variable.embedding_csv_failed)


if __name__ == '__main__':
    unittest.main()
