class QueryConfig:

    def __init__(
            self,
            role: str = 'user',
            temperature: float = 0.7,
            max_new_tokens: int = 512,
            use_context: bool = True,
            top_k_sentence_chunks: int = 3
    ) -> None:

        self.role = role
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.use_context = use_context
        self.top_k_sentence_chunks = top_k_sentence_chunks
