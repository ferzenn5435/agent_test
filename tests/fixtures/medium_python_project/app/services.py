"""订单与发票服务。"""

from __future__ import annotations


def calculate_subtotal(prices: list[float]) -> float:
    """计算商品小计。"""
    return round(sum(prices), 2)


def calculate_tax(subtotal: float, tax_rate: float) -> float:
    """根据税率计算税额。"""
    return round(subtotal * tax_rate, 2)


def build_invoice_summary(prices: list[float], tax_rate: float) -> str:
    """生成简短发票摘要。"""
    subtotal = calculate_subtotal(prices)
    tax = calculate_tax(subtotal, tax_rate)
    total = round(subtotal + tax, 2)
    return f"subtotal={subtotal:.2f}; tax={tax:.2f}; total={total:.2f}"
