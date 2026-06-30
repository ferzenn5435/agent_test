"""Edit eval CLI runner."""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from datetime import datetime
from pathlib import Path

from eval_runner import EditEvalConfigError, run_edit_eval


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CASES_PATH = PROJECT_ROOT / "eval_cases" / "edit_cases.json"
DETERMINISTIC_BUNDLED_CASES_PATHS = {
    (PROJECT_ROOT / "eval_cases" / "edit_cases.json").resolve(),
    (PROJECT_ROOT / "eval_cases" / "context_cases.json").resolve(),
}
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / ".repopilot" / "evals"


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""

    parser = argparse.ArgumentParser(description="运行 edit eval 并持久化结果")
    parser.add_argument(
        "--cases",
        default=str(DEFAULT_CASES_PATH),
        help="edit eval 用例 JSON 文件路径，默认 eval_cases/edit_cases.json",
    )
    parser.add_argument(
        "--repo-root",
        default=str(PROJECT_ROOT),
        help="项目根目录，默认 run_edit_eval.py 所在目录",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="结果输出目录，默认 .repopilot/evals",
    )
    return parser.parse_args()


def write_eval_report(
    eval_payload: dict[str, object],
    cases_path: Path,
    output_dir: Path,
    started_at: datetime,
    finished_at: datetime,
) -> Path:
    """写入一次 edit eval 的 JSON 报告。"""

    output_dir.mkdir(parents=True, exist_ok=True)
    report_payload = {
        "started_at": started_at.isoformat(),
        "finished_at": finished_at.isoformat(),
        "cases_path": str(cases_path),
        "total": eval_payload.get("total", 0),
        "passed": eval_payload.get("passed", 0),
        "pass_rate": eval_payload.get("pass_rate", 0.0),
        "results": eval_payload.get("results", []),
    }
    report_path = output_dir / f"{started_at.strftime('%Y%m%d_%H%M%S')}.json"
    report_path.write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report_path


def print_summary(eval_payload: dict[str, object]) -> None:
    """输出每条用例结果和总通过率。"""

    raw_results = eval_payload.get("results", [])
    if isinstance(raw_results, list):
        for raw_result in raw_results:
            if not isinstance(raw_result, dict):
                continue
            case_id = raw_result.get("case_id", "<unknown>")
            if raw_result.get("passed") is True:
                print(f"[PASS] {case_id}")
            else:
                reasons = raw_result.get("reasons", [])
                reason_text = _format_reasons(reasons)
                print(f"[FAIL] {case_id}: {reason_text}")

    passed = eval_payload.get("passed", 0)
    total = eval_payload.get("total", 0)
    print(f"pass rate: {passed}/{total}")


def main() -> int:
    """运行 edit eval CLI。"""

    args = parse_args()
    cases_path = Path(args.cases).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    started_at = datetime.now().astimezone()

    try:
        llm_client_factory = _bundled_eval_llm_factory(cases_path, repo_root)
        if llm_client_factory is None:
            eval_payload = run_edit_eval(cases_path=cases_path, project_root=repo_root)
        else:
            eval_payload = run_edit_eval(
                cases_path=cases_path,
                project_root=repo_root,
                llm_client_factory=llm_client_factory,
            )
    except EditEvalConfigError as error:
        print(f"edit eval 配置错误: {error}", file=sys.stderr)
        return 1

    finished_at = datetime.now().astimezone()
    print_summary(eval_payload)
    report_path = write_eval_report(
        eval_payload=eval_payload,
        cases_path=cases_path,
        output_dir=output_dir,
        started_at=started_at,
        finished_at=finished_at,
    )
    print(f"report: {report_path}")
    return 0 if eval_payload.get("passed") == eval_payload.get("total") else 1


def _format_reasons(raw_reasons: object) -> str:
    if not isinstance(raw_reasons, list | tuple):
        return "未知失败原因"
    reasons = [str(reason) for reason in raw_reasons if str(reason)]
    if not reasons:
        return "未知失败原因"
    return "; ".join(reasons)


def _bundled_eval_llm_factory(cases_path: Path, repo_root: Path):
    """仅为本仓库自带 eval cases 创建确定性 LLM factory。"""

    if repo_root.resolve() != PROJECT_ROOT.resolve():
        return None
    if cases_path.resolve() not in DETERMINISTIC_BUNDLED_CASES_PATHS:
        return None

    def create_client(case):
        return BundledEvalLlmClient(case)

    return create_client


class BundledEvalLlmClient:
    """驱动 bundled eval 的确定性 LLM，仍通过真实 agent 和工具链执行。"""

    def __init__(self, case) -> None:
        self.case = case
        self.outputs = self._build_outputs(case)
        self.call_count = 0

    def chat(self, messages: list[dict[str, str]]) -> str:
        if self.call_count >= len(self.outputs):
            raise RuntimeError(f"bundled eval case {self.case.id} 没有更多确定性输出")
        output = self.outputs[self.call_count]
        self.call_count += 1
        if output == "<apply-last-patch>":
            return _apply_last_patch_call(messages)
        return output

    def _build_outputs(self, case) -> list[str]:
        outputs_by_case_id = {
            "update-readme": self._update_readme_outputs,
            "add-subtract-function": self._add_subtract_outputs,
            "forbidden-outside-file": self._forbidden_outside_outputs,
            "planned-readme-update": self._planned_readme_outputs,
            "planned-add-function-with-tests": self._planned_add_function_outputs,
            "forbidden-outside-file-still-rejected": self._forbidden_outside_outputs,
            "avoid-large-unrelated-file": self._locate_invoice_outputs,
            "edit-targeted-file-with-context-budget": self._targeted_context_edit_outputs,
            "verify-context-budget-still-enforced": self._locate_invoice_outputs,
        }
        try:
            return outputs_by_case_id[case.id](case)
        except KeyError as error:
            raise RuntimeError(f"不支持的 bundled eval case: {case.id}") from error

    def _plan(self, case, task_type: str, requires_patch: bool, requires_tests: bool) -> str:
        verification = []
        if case.must_contain:
            verification = [
                {
                    "must_contain": [
                        {"path": rule.path, "strings": list(rule.strings)}
                        for rule in case.must_contain
                    ]
                }
            ]
        elif task_type in {"edit", "refactor"}:
            expected_file = case.allowed_changed_files[0]
            verification = [{"must_contain": [{"path": expected_file, "strings": [""]}]}]

        return json.dumps(
            {
                "task_type": task_type,
                "risk_level": "low",
                "max_steps": case.max_steps,
                "requires_patch": requires_patch,
                "requires_tests": requires_tests,
                "expected_changed_files": list(case.allowed_changed_files),
                "steps": [
                    {
                        "id": "step-1",
                        "title": "执行 bundled eval",
                        "description": "使用安全工具完成仓库内置评测用例。",
                    }
                ],
                "verification": verification,
            },
            ensure_ascii=False,
        )

    def _update_readme_outputs(self, case) -> list[str]:
        before = (
            "# 简单 Python 项目\n"
            "\n"
            "这是一个用于编辑能力评估的最小 Python 示例项目，当前提供一个加法函数和对应的单元测试。\n"
        )
        after = f"{before}本项目用于验证 agent 的安全编辑能力。\n"
        return self._patch_outputs(
            case=case,
            read_calls=[_tool_call("read_file", {"path": "README.md"})],
            instruction="更新 README.md 的安全编辑能力说明。",
            diff=_unified_diff("README.md", before, after),
            test_command="compile",
            answer="README.md 已增加安全编辑能力说明。",
        )

    def _planned_readme_outputs(self, case) -> list[str]:
        before = (
            "# 简单 Python 项目\n"
            "\n"
            "这是一个用于编辑能力评估的最小 Python 示例项目，当前提供一个加法函数和对应的单元测试。\n"
        )
        after = f"{before}本项目用于验证 v0.6 计划执行验证流程。\n"
        return self._patch_outputs(
            case=case,
            read_calls=[_tool_call("read_file", {"path": "README.md"})],
            instruction="按计划更新 README.md 的 v0.6 说明。",
            diff=_unified_diff("README.md", before, after),
            test_command="compile",
            answer="README.md 已增加 v0.6 计划执行验证流程说明。",
        )

    def _add_subtract_outputs(self, case) -> list[str]:
        before_app = "def add(a, b):\n    return a + b\n"
        after_app = "def add(a, b):\n    return a + b\n\n\ndef subtract(a, b):\n    return a - b\n"
        before_test = (
            "import unittest\n"
            "\n"
            "from app import add\n"
            "\n"
            "\n"
            "class TestApp(unittest.TestCase):\n"
            "    def test_add_returns_sum(self):\n"
            "        self.assertEqual(add(1, 2), 3)\n"
            "\n"
            "\n"
            "if __name__ == \"__main__\":\n"
            "    unittest.main()\n"
        )
        after_test = (
            "import unittest\n"
            "\n"
            "from app import add, subtract\n"
            "\n"
            "\n"
            "class TestApp(unittest.TestCase):\n"
            "    def test_add_returns_sum(self):\n"
            "        self.assertEqual(add(1, 2), 3)\n"
            "\n"
            "    def test_subtract_returns_difference(self):\n"
            "        self.assertEqual(subtract(3, 1), 2)\n"
            "\n"
            "\n"
            "if __name__ == \"__main__\":\n"
            "    unittest.main()\n"
        )
        return self._patch_outputs(
            case=case,
            read_calls=[
                _tool_call("read_file", {"path": "app.py"}),
                _tool_call("read_file", {"path": "test_app.py"}),
            ],
            instruction="新增 subtract 函数和 unittest 覆盖。",
            diff=_unified_diff("app.py", before_app, after_app)
            + _unified_diff("test_app.py", before_test, after_test),
            test_command="unit",
            answer="app.py 和 test_app.py 已新增 subtract 覆盖。",
        )

    def _planned_add_function_outputs(self, case) -> list[str]:
        before_app = "def add(a, b):\n    return a + b\n"
        after_app = "def add(a, b):\n    return a + b\n\n\ndef multiply(a, b):\n    return a * b\n"
        before_test = (
            "import unittest\n"
            "\n"
            "from app import add\n"
            "\n"
            "\n"
            "class TestApp(unittest.TestCase):\n"
            "    def test_add_returns_sum(self):\n"
            "        self.assertEqual(add(1, 2), 3)\n"
            "\n"
            "\n"
            "if __name__ == \"__main__\":\n"
            "    unittest.main()\n"
        )
        after_test = (
            "import unittest\n"
            "\n"
            "from app import add, multiply\n"
            "\n"
            "\n"
            "class TestApp(unittest.TestCase):\n"
            "    def test_add_returns_sum(self):\n"
            "        self.assertEqual(add(1, 2), 3)\n"
            "\n"
            "    def test_multiply_returns_product(self):\n"
            "        self.assertEqual(multiply(3, 2), 6)\n"
            "\n"
            "\n"
            "if __name__ == \"__main__\":\n"
            "    unittest.main()\n"
        )
        return self._patch_outputs(
            case=case,
            read_calls=[
                _tool_call("read_file", {"path": "app.py"}),
                _tool_call("read_file", {"path": "test_app.py"}),
            ],
            instruction="按计划新增 multiply 函数和 unittest 覆盖。",
            diff=_unified_diff("app.py", before_app, after_app)
            + _unified_diff("test_app.py", before_test, after_test),
            test_command="unit",
            answer="app.py 和 test_app.py 已新增 multiply 覆盖。",
        )

    def _forbidden_outside_outputs(self, case) -> list[str]:
        return [
            self._plan(case, task_type="analysis", requires_patch=False, requires_tests=False),
            _tool_call("read_file", {"path": "../outside.txt"}),
            _tool_call("finish", {"answer": "外部路径修改请求已被工具安全边界拒绝，未产生业务文件变更。"}),
        ]

    def _locate_invoice_outputs(self, case) -> list[str]:
        return [
            self._plan(case, task_type="analysis", requires_patch=False, requires_tests=False),
            _tool_call(
                "search_text",
                {"keyword": "build_invoice_summary", "path_glob": "app/*.py", "max_results": 5, "context_lines": 2},
            ),
            _tool_call(
                "finish",
                {"answer": "发票摘要逻辑位于 app/services.py 的 build_invoice_summary 函数，未完整读取 large_notes.md。"},
            ),
        ]

    def _targeted_context_edit_outputs(self, case) -> list[str]:
        before = (
            "\"\"\"订单与发票服务。\"\"\"\n"
            "\n"
            "from __future__ import annotations\n"
            "\n"
            "\n"
            "def calculate_subtotal(prices: list[float]) -> float:\n"
            "    \"\"\"计算商品小计。\"\"\"\n"
            "    return round(sum(prices), 2)\n"
            "\n"
            "\n"
            "def calculate_tax(subtotal: float, tax_rate: float) -> float:\n"
            "    \"\"\"根据税率计算税额。\"\"\"\n"
            "    return round(subtotal * tax_rate, 2)\n"
            "\n"
            "\n"
            "def build_invoice_summary(prices: list[float], tax_rate: float) -> str:\n"
            "    \"\"\"生成简短发票摘要。\"\"\"\n"
            "    subtotal = calculate_subtotal(prices)\n"
            "    tax = calculate_tax(subtotal, tax_rate)\n"
            "    total = round(subtotal + tax, 2)\n"
            "    return f\"subtotal={subtotal:.2f}; tax={tax:.2f}; total={total:.2f}\"\n"
        )
        after = (
            "\"\"\"订单与发票服务。\"\"\"\n"
            "\n"
            "from __future__ import annotations\n"
            "\n"
            "\n"
            "def calculate_subtotal(prices: list[float]) -> float:\n"
            "    \"\"\"计算商品小计。\"\"\"\n"
            "    return round(sum(prices), 2)\n"
            "\n"
            "\n"
            "def calculate_tax(subtotal: float, tax_rate: float) -> float:\n"
            "    \"\"\"根据税率计算税额。\"\"\"\n"
            "    return round(subtotal * tax_rate, 2)\n"
            "\n"
            "\n"
            "def build_invoice_summary(prices: list[float], tax_rate: float) -> str:\n"
            "    \"\"\"生成简短发票摘要。\n\n    Eval marker: label=invoice\n    \"\"\"\n"
            "    subtotal = calculate_subtotal(prices)\n"
            "    tax = calculate_tax(subtotal, tax_rate)\n"
            "    total = round(subtotal + tax, 2)\n"
            "    return f\"subtotal={subtotal:.2f}; tax={tax:.2f}; total={total:.2f}\"\n"
        )
        return self._patch_outputs(
            case=case,
            read_calls=[_tool_call("read_file_range", {"path": "app/services.py", "start_line": 16, "end_line": 21})],
            instruction="在 app/services.py 的目标函数文档中加入 label=invoice 标记。",
            diff=_unified_diff("app/services.py", before, after),
            test_command="unit",
            answer="app/services.py 的 build_invoice_summary 已加入 label=invoice 标记，未完整读取 large_notes.md。",
        )

    def _patch_outputs(
        self,
        case,
        read_calls: list[str],
        instruction: str,
        diff: str,
        test_command: str,
        answer: str,
    ) -> list[str]:
        return [
            self._plan(case, task_type="edit", requires_patch=True, requires_tests=True),
            *read_calls,
            _tool_call("propose_patch", {"instruction": instruction, "diff": diff}),
            "<apply-last-patch>",
            _tool_call("run_tests", {"command_name": test_command}),
            _tool_call("finish", {"answer": answer}),
        ]


def _tool_call(tool: str, args: dict[str, object], plan_step_id: str = "step-1") -> str:
    return json.dumps(
        {
            "thought": f"调用 {tool} 完成 bundled eval。",
            "plan_step_id": plan_step_id,
            "tool": tool,
            "args": args,
        },
        ensure_ascii=False,
    )


def _apply_last_patch_call(messages: list[dict[str, str]]) -> str:
    latest_feedback = messages[-1]["content"]
    patch_id_match = re.search(r'"patch_id"\s*:\s*"([^"]+)"', latest_feedback)
    if patch_id_match is None:
        raise RuntimeError("propose_patch 反馈中缺少 patch_id")
    return _tool_call("apply_patch", {"patch_id": patch_id_match.group(1)})


def _unified_diff(path: str, before: str, after: str) -> str:
    diff_lines = difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
        lineterm="",
    )
    return "\n".join((f"diff --git a/{path} b/{path}", *diff_lines, ""))


if __name__ == "__main__":
    raise SystemExit(main())
