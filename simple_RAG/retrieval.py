import torch
import pandas as pd
from time import perf_counter as timer
from sentence_transformers import util


class Retrieval:

    @staticmethod
    def retrieve_context(
            query_embedding: torch.tensor,
            embeddings: torch.tensor,
            df_embeddings: pd.DataFrame,
            top_k_chunks: int = 3
    ) -> (torch.Tensor, torch.Tensor, pd.DataFrame):

        start_time = timer()
        # Raise `RuntimeError`: Expected all tensors to be on the same device, but found two devices, cpu and cuda
        dot_scores = util.dot_score(query_embedding, embeddings)[0]  # Similarity Search
        scores, indices = torch.topk(dot_scores, k=top_k_chunks)  # Get Top-k Relevant Context
        end_time = timer()
        print(f"[INFO] Time to get relevant context: {end_time - start_time:.5f} seconds.")

        # Return scores and indices back to CPU
        scores = scores.cpu()
        indices = indices.cpu()

        # Create a list of context items and add score to it
        df_context = df_embeddings.iloc[indices]
        df_context.loc[:, 'score'] = scores

        return scores, indices, df_context
