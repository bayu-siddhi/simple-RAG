import unittest
from simple_RAG import Source
from tests.variable import Variable


class SourceTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.variable = Variable()

    def test_open_and_extract_pdf_success(self) -> None:
        """Test open and read (extract) all PDF pages to chunks (success)"""
        document = Source(self.variable.pdf_path_success, self.variable.min_token_length_per_chunk)
        self.assertEqual(document.pdf_path, self.variable.pdf_path_success)
        self.assertEqual(document.min_token_length_per_chunk, self.variable.min_token_length_per_chunk)
        self.assertIsInstance(document.content, list)
        self.assertIsInstance(document.content[0], dict)
        self.assertIsInstance(document.all_chunks, list)
        self.assertIsInstance(document.all_chunks[0], dict)
        self.assertIsInstance(document.filtered_chunks, list)
        self.assertIsInstance(document.filtered_chunks[0], dict)
        self.assertGreater(document.slice_size, 0)
        for idx, page in enumerate(document.content):
            self.assertEqual(page['page_index'], idx)
            self.assertIsNotNone(page['text'])
            self.assertIsNotNone(page['sentences'])
            self.assertIsNotNone(page['num_sentences'])
            self.assertIsNotNone(page['chunks'])
            self.assertIsNotNone(page['num_chunks'])
        self.assertGreater(len(document.all_chunks), len(document.content))
        for item in document.all_chunks:
            self.assertIsNotNone(item['chunk'])
            self.assertIsNotNone(item['num_chunk_token'])
        self.assertGreater(len(document.filtered_chunks), len(document.content))
        for item in document.filtered_chunks:
            self.assertIsNotNone(item['chunk'])
            self.assertIsNotNone(item['num_chunk_token'])

    def test_open_and_extract_pdf_failed(self) -> None:
        """Test open and read (extract) all PDF pages to chunks (file not found)"""
        with self.assertRaises(FileNotFoundError):
            Source(self.variable.pdf_path_failed, self.variable.min_token_length_per_chunk)


if __name__ == '__main__':
    unittest.main()
