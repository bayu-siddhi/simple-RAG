import unittest
from simple_RAG import Source
from simple_RAG import FileNotFoundException


class SourceTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.pdf_path_success = 'data/test.pdf'
        self.pdf_path_failed = 'data/not-found.pdf'

    def test_load_pdf_success(self) -> None:
        """Test load PDF (success)"""
        document = Source(pdf_path=self.pdf_path_success)
        self.assertEqual(document.pdf_path, self.pdf_path_success)

    def test_load_pdf_failed(self) -> None:
        """Test load PDF not found (failed)"""
        with self.assertRaises(FileNotFoundException):
            Source(pdf_path=self.pdf_path_failed)

    def test_open_and_read_pdf(self) -> None:
        """Test open and read (extract) all PDF pages to chunks"""
        document = Source(pdf_path=self.pdf_path_success)
        document.open_and_extract_pdf()
        self.assertIsInstance(document.content, list)
        self.assertIsInstance(document.content[0], dict)
        self.assertNotEquals(document.slice_size, 0)
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


if __name__ == '__main__':
    unittest.main()
