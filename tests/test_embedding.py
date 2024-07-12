import torch
import unittest
import pandas as pd
from simple_RAG import Source
from simple_RAG import Embedding
from simple_RAG import EmbeddingModel
from simple_RAG import FileNotFoundException


class EmbeddingTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.pdf_path = 'data/test.pdf'
        self.embedding_csv_success = 'data/test_embedding.csv'
        self.embedding_csv_failed = 'data/not_found.csv'
        self.embedding_model_name = 'all-mpnet-base-v2'

    def test_add_and_save_new_embedding(self) -> None:
        """Test add new embedding from embedding_model.encode() and save to CSV (success)"""
        source = Source(self.pdf_path)
        source.open_and_extract_pdf()
        embedding_model = EmbeddingModel(self.embedding_model_name)
        embedding = embedding_model.encode(source.filtered_chunks, 'cuda')

        self.assertEqual(embedding_model.device, 'cuda')
        self.assertIsInstance(embedding, Embedding)
        self.assertEqual(len(embedding.embedding), len(embedding.df_embedding))
        self.assertIsInstance(embedding.embedding, torch.Tensor)
        self.assertIsInstance(embedding.df_embedding, pd.DataFrame)

        embedding.save_to_csv(self.embedding_csv_success)

        df = pd.read_csv(self.embedding_csv_success, sep=',')
        self.assertListEqual(list(df.columns), ['page_index', 'chunk', 'num_chunk_token', 'embedding'])
        self.assertEqual(len(df), len(embedding.embedding))

    def test_load_from_csv_success(self) -> None:
        """Test load embedding from CSV (success)"""
        embedding = Embedding()
        embedding.load_from_csv(self.embedding_csv_success)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.assertEqual(embedding.device, device)
        self.assertEqual(len(embedding.embedding), len(embedding.df_embedding))
        self.assertIsInstance(embedding.embedding, torch.Tensor)
        self.assertIsInstance(embedding.df_embedding, pd.DataFrame)

    def test_load_from_csv_failed(self) -> None:
        """Test load embedding from CSV not found (failed)"""
        with self.assertRaises(FileNotFoundException):
            embedding = Embedding()
            embedding.load_from_csv(self.embedding_csv_failed)


if __name__ == '__main__':
    unittest.main()
