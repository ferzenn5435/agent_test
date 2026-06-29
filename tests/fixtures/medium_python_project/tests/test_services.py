"""app.services 的单元测试。"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

FIXTURE_ROOT = Path(__file__).resolve().parents[1]
if str(FIXTURE_ROOT) not in sys.path:
    sys.path.insert(0, str(FIXTURE_ROOT))

from app.services import calculate_subtotal, calculate_tax, build_invoice_summary


class TestServices(unittest.TestCase):
    def test_calculate_subtotal_rounds_prices(self) -> None:
        self.assertEqual(20.0, calculate_subtotal([12.5, 7.5]))

    def test_calculate_tax_uses_tax_rate(self) -> None:
        self.assertEqual(2.0, calculate_tax(20.0, 0.1))

    def test_build_invoice_summary_formats_values(self) -> None:
        self.assertEqual(
            "subtotal=20.00; tax=2.00; total=22.00",
            build_invoice_summary([12.5, 7.5], tax_rate=0.1),
        )


if __name__ == "__main__":
    unittest.main()
