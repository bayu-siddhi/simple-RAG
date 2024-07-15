import torch
import unittest
import pandas as pd
from simple_RAG import Retrieval
from simple_RAG import Embedding
from simple_RAG import EmbeddingModel


class RetrievalTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.device_cpu = 'cpu'
        self.device_cuda = 'cuda'
        self.query = 'what is Retrieval Augmented Generation?'
        self.embedding_csv = 'data/test_embeddings.csv'
        self.embedding_model_name = 'all-mpnet-base-v2'
        self.n_chunks = 5

    def test_retrieve_context_success(self) -> None:
        """Test retrieve scores. indices, and dataframe of relevant context (success)"""
        embeddings = Embedding(self.device_cuda)
        embeddings.load_from_csv(self.embedding_csv)

        embedding_model = EmbeddingModel(self.embedding_model_name, self.device_cuda)
        query_embedding = embedding_model.encode(self.query)

        scores, indices, df_context = Retrieval.retrieve_context(
            query_embedding.embedding,
            embeddings.embedding,
            embeddings.df_embedding,
            self.n_chunks
        )

        self.assertEqual(len(scores), self.n_chunks)
        self.assertEqual(len(indices), self.n_chunks)
        self.assertEqual(len(df_context), self.n_chunks)

        self.assertEqual(len(scores), len(indices))
        self.assertEqual(len(scores), len(df_context))
        self.assertEqual(len(df_context), len(indices))

        self.assertIsInstance(scores, torch.Tensor)
        self.assertIsInstance(indices, torch.Tensor)
        self.assertIsInstance(df_context, pd.DataFrame)

        for score in scores:
            self.assertGreater(score, 0)
            self.assertLess(score, 1)

    def test_retrieve_context_failed(self) -> None:
        """
        Test retrieve scores and indices of relevant context (failed).

        - ``RuntimeError``
        - Expected all tensors to be on the same device, but found at least two devices, cpu and cuda.
        - This occurs because the embedding device between data embeddings and query_embedding is not the same.
        """
        # Embeddings using 'cuda'
        embeddings = Embedding(self.device_cuda)
        embeddings.load_from_csv(self.embedding_csv)

        # Query embeddings using 'cpu'
        embedding_model = EmbeddingModel(self.embedding_model_name, self.device_cpu)
        query_embedding = embedding_model.encode(self.query)

        with self.assertRaises(RuntimeError):
            scores, indices, df_context = Retrieval.retrieve_context(
                query_embedding.embedding,
                embeddings.embedding,
                embeddings.df_embedding,
                self.n_chunks
            )


if __name__ == '__main__':
    unittest.main()
