import unittest

from app import add


class TestApp(unittest.TestCase):
    def test_add_returns_sum(self):
        self.assertEqual(add(1, 2), 3)


if __name__ == "__main__":
    unittest.main()
