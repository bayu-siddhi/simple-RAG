class Variable:
    device_cpu = 'cpu'
    device_cuda = 'cuda'

    pdf_path_success = 'data/test.pdf'
    pdf_path_failed = 'data/not-found.pdf'
    min_token_length_per_chunk = 30

    embedding_model_success = 'all-mpnet-base-v2'
    embedding_model_failed = 'not-found-model'
    embedding_csv_cpu_success = './data/test_cpu_embeddings.csv'
    embedding_csv_cuda_success = './data/test_cuda_embeddings.csv'
    embedding_csv_failed = './data/not_found_embeddings.csv'

    llm_model_success = './google/gemma-2b-it'
    llm_model_failed = './google/not-found'
    use_quantization_config_true = True
    use_quantization_config_false = False

    role = 'user'
    query = 'What is Retrieval Augmented Generation?'
    n_chunks = 2
    temperature = 0.5
    max_new_tokens = 128
    format_response_text = True
    use_context_true = True
    use_context_false = False
