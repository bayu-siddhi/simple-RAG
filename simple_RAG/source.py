import os
import re
import pymupdf
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from spacy.lang.en import English
from simple_RAG.helper import Helper
from simple_RAG.exception.file_not_found_exception import FileNotFoundException


class Source:

    def __init__(self, pdf_path: str) -> None:
        self.pdf_path: str = pdf_path
        self.content: list[dict] = list()
        self.all_chunks: list[dict] = list()
        self.filtered_chunks: list[dict] = list()
        self.slice_size: int = 0
        self.__load_pdf()
        load_dotenv()

    def __load_pdf(self) -> None:
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundException(f"File {self.pdf_path} doesn't exist.")

    def open_and_read_pdf(self) -> None:
        document = pymupdf.open(filename=self.pdf_path, filetype='pdf')
        for index, page in enumerate(document):
            text = page.get_text()
            text = Helper.text_formatter(text)
            self.content.append({'page_index': index,
                                 'text': text})
        self.__split_sentence()
        self.__chunking_sentence()
        self.__filter_chunks(int(os.getenv('MIN_TOKEN_LENGTH')))

    def __split_sentence(self) -> None:
        nlp = English()
        nlp.add_pipe('sentencizer')
        for item in self.content:
            item['sentences'] = list(nlp(item['text']).sents)
            item['sentences'] = [str(sentence) for sentence in item['sentences']]
            item['num_sentences'] = len(item['sentences'])

    def __chunking_sentence(self) -> None:
        page_sentence_count = [int(item['num_sentences']) for item in self.content]
        self.slice_size = round(np.mean(page_sentence_count))
        for item in self.content:
            item['chunks'] = list()
            for i in range(0, len(item['sentences']), self.slice_size):
                item['chunks'].append(item['sentences'][i:i + self.slice_size])
            item['num_chunks'] = len(item['chunks'])

    def __filter_chunks(self, min_token_length: int) -> None:
        for item in self.content:
            for sentence_chunk in item["chunks"]:
                chunk_dict = dict()
                chunk_dict['page_index'] = item['page_index']
                # Join the sentences together into a paragraph
                joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
                # Add space after dot (".A" -> ". A")
                joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)
                chunk_dict["chunk"] = joined_sentence_chunk
                # Get stats about the chunk, 1 token = ~4 characters
                chunk_dict["num_chunk_token"] = len(joined_sentence_chunk) / 4
                self.all_chunks.append(chunk_dict)

        df = pd.DataFrame(self.all_chunks)
        self.filtered_chunks = df[df["num_chunk_token"] > min_token_length].to_dict(orient="records")