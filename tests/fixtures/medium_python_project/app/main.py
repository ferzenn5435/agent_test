"""中等规模 fixture 的命令入口。"""

from app.services import build_invoice_summary


def main() -> None:
    """打印示例发票摘要。"""
    summary = build_invoice_summary([12.5, 7.5], tax_rate=0.1)
    print(summary)


if __name__ == "__main__":
    main()
