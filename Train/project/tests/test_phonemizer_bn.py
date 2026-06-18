import unittest

from project.preprocessing.phonemizer_bn import phonemize_bangla


class PhonemizerTests(unittest.TestCase):
    def test_phonemizes_bangla_and_latin(self) -> None:
        output = phonemize_bangla("আল্লাহ hello")
        self.assertTrue(output)
        self.assertIn("allah", output)


if __name__ == "__main__":
    unittest.main()
