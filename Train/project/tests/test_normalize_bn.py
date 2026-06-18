import unittest

from project.preprocessing.normalize_bn import normalize_bangla_text, number_to_bn_words


class NormalizeBanglaTextTests(unittest.TestCase):
    def test_number_expansion(self) -> None:
        self.assertEqual(number_to_bn_words(123), "একশ তেইশ")

    def test_currency_expansion(self) -> None:
        self.assertEqual(normalize_bangla_text("৳৫০০"), "পাঁচশ টাকা")

    def test_date_expansion(self) -> None:
        self.assertEqual(normalize_bangla_text("১০/০৫/২০২৬"), "দশ মে দুই হাজার ছাব্বিশ")


if __name__ == "__main__":
    unittest.main()
