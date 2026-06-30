"""v0.6 planner unit tests."""

from __future__ import annotations

import json
import unittest

from planner import (
    PlannerError,
    build_planner_prompt,
    _validate_llm_output,
    _parse_plan_steps,
    _parse_must_contain_rules,
    _parse_verification_specs,
    _parse_expected_changed_files,
    _build_task_plan_from_dict,
    create_plan,
    LlmClientProtocol,
)
from schemas import (
    MustContainRule,
    PlanStep,
    TaskPlan,
    VerificationSpec,
)


VALID_INSPECT_REPO = {
    "ok": True,
    "index_status": "cached",
    "file_count": 15,
    "python_file_count": 8,
    "main_python_modules": [
        {"path": "src/core.py", "size_bytes": 2048, "line_count": 80, "symbol_count": 5},
    ],
    "test_files": [
        {"path": "tests/test_core.py", "size_bytes": 512, "line_count": 30},
    ],
    "entrypoint_candidates": [
        {"path": "main.py", "size_bytes": 256, "line_count": 15, "symbol_count": 2, "reasons": ["main_guard"]},
    ],
    "large_files": [],
}


VALID_EDIT_JSON = {
    "task_type": "edit",
    "risk_level": "medium",
    "requires_patch": True,
    "requires_tests": True,
    "expected_changed_files": ["src/core.py"],
    "steps": [
        {"id": "inspect", "title": "Review context", "description": "Read src/core.py to confirm current implementation"},
        {"id": "modify", "title": "Modify logic", "description": "Update core logic"},
        {"id": "verify", "title": "Run verification", "description": "Execute unit tests"},
    ],
    "verification": [
        {
            "must_contain": [
                {"path": "src/core.py", "strings": ["def run"]},
            ]
        }
    ],
}

VALID_ANALYSIS_JSON = {
    "task_type": "analysis",
    "risk_level": "low",
    "requires_patch": False,
    "requires_tests": False,
    "expected_changed_files": [],
    "steps": [
        {"id": "inspect", "title": "Analyze project structure"},
        {"id": "report", "title": "Generate report"},
    ],
    "verification": [],
}


def _valid_edit_json_str() -> str:
    return json.dumps(VALID_EDIT_JSON, ensure_ascii=False)


def _valid_analysis_json_str() -> str:
    return json.dumps(VALID_ANALYSIS_JSON, ensure_ascii=False)


class FakeLlmClient:
    """Minimal fake LLM that satisfies LlmClientProtocol."""

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.call_count = 0
        self.messages_by_call: list[list[dict[str, str]]] = []

    def chat(self, messages: list[dict[str, str]]) -> str:
        self.messages_by_call.append([dict(m) for m in messages])
        if self.call_count >= len(self.outputs):
            raise AssertionError("fake LLM has no more outputs")
        output = self.outputs[self.call_count]
        self.call_count += 1
        return output


class TestBuildPlannerPrompt(unittest.TestCase):
    """Verify planner prompt construction."""

    def test_includes_user_task(self) -> None:
        prompt = build_planner_prompt("modify entry", VALID_INSPECT_REPO, 8)
        self.assertIn("modify entry", prompt)

    def test_includes_inspect_repo_result(self) -> None:
        prompt = build_planner_prompt("analyze", VALID_INSPECT_REPO, 8)
        self.assertIn("src/core.py", prompt)
        self.assertIn("main_python_modules", prompt)

    def test_includes_max_steps(self) -> None:
        prompt = build_planner_prompt("test", VALID_INSPECT_REPO, 5)
        self.assertIn("5", prompt)

    def test_requires_strict_json(self) -> None:
        prompt = build_planner_prompt("task", VALID_INSPECT_REPO, 8)
        self.assertIn("code fence", prompt)
        self.assertIn("ONLY", prompt)

    def test_lists_task_types_and_risk_levels(self) -> None:
        prompt = build_planner_prompt("task", VALID_INSPECT_REPO, 8)
        self.assertIn("analysis", prompt)
        self.assertIn("edit", prompt)
        self.assertIn("refactor", prompt)
        self.assertIn("low", prompt)
        self.assertIn("medium", prompt)
        self.assertIn("high", prompt)


class TestValidateLlmOutput(unittest.TestCase):
    """Verify LLM output validation."""

    def test_accepts_plain_json_object(self) -> None:
        result = _validate_llm_output('{"task_type": "analysis"}')
        self.assertEqual({"task_type": "analysis"}, result)

    def test_rejects_code_fence(self) -> None:
        with self.assertRaisesRegex(PlannerError, "markdown code fence"):
            _validate_llm_output("```json\n{\"task_type\": \"analysis\"}\n```")

    def test_rejects_code_fence_without_language(self) -> None:
        with self.assertRaisesRegex(PlannerError, "markdown code fence"):
            _validate_llm_output("```\n{\"task_type\": \"analysis\"}\n```")

    def test_rejects_array(self) -> None:
        with self.assertRaisesRegex(PlannerError, "not array or scalar"):
            _validate_llm_output('[{"task_type": "analysis"}]')

    def test_rejects_scalar(self) -> None:
        with self.assertRaisesRegex(PlannerError, "not array or scalar"):
            _validate_llm_output('"just a string"')

    def test_rejects_non_json(self) -> None:
        with self.assertRaisesRegex(PlannerError, "not valid JSON"):
            _validate_llm_output("not JSON")

    def test_rejects_empty_output(self) -> None:
        with self.assertRaisesRegex(PlannerError, "empty"):
            _validate_llm_output("")

    def test_rejects_whitespace_only(self) -> None:
        with self.assertRaisesRegex(PlannerError, "empty"):
            _validate_llm_output("   \n  ")

    def test_rejects_extra_text_before_json(self) -> None:
        with self.assertRaisesRegex(PlannerError, "not valid JSON"):
            _validate_llm_output("Here is the plan:\n{\"task_type\": \"analysis\"}")

    def test_accepts_json_with_trailing_newline(self) -> None:
        result = _validate_llm_output('{"task_type": "analysis"}\n')
        self.assertEqual({"task_type": "analysis"}, result)


class TestParsePlanSteps(unittest.TestCase):
    """Verify steps parsing."""

    def test_parses_valid_steps(self) -> None:
        raw_steps = [
            {"id": "one", "title": "First step", "description": "A description"},
            {"id": "two", "title": "Second step"},
        ]
        steps = _parse_plan_steps(raw_steps)
        self.assertEqual(2, len(steps))
        self.assertEqual("one", steps[0].id)
        self.assertEqual("First step", steps[0].title)
        self.assertEqual("A description", steps[0].description)
        self.assertEqual("two", steps[1].id)
        self.assertEqual("Second step", steps[1].title)
        self.assertEqual("", steps[1].description)

    def test_rejects_non_list(self) -> None:
        with self.assertRaisesRegex(PlannerError, "steps must be a list"):
            _parse_plan_steps("not a list")

    def test_rejects_empty_list(self) -> None:
        with self.assertRaisesRegex(PlannerError, "steps must not be empty"):
            _parse_plan_steps([])

    def test_rejects_non_dict_step(self) -> None:
        with self.assertRaisesRegex(PlannerError, "must be a dict"):
            _parse_plan_steps(["string"])

    def test_rejects_missing_id(self) -> None:
        with self.assertRaisesRegex(PlannerError, "id is required"):
            _parse_plan_steps([{"title": "no id"}])

    def test_rejects_empty_id(self) -> None:
        with self.assertRaisesRegex(PlannerError, "id is required"):
            _parse_plan_steps([{"id": "", "title": "empty id"}])

    def test_rejects_missing_title(self) -> None:
        with self.assertRaisesRegex(PlannerError, "title is required"):
            _parse_plan_steps([{"id": "one"}])

    def test_rejects_empty_title(self) -> None:
        with self.assertRaisesRegex(PlannerError, "title is required"):
            _parse_plan_steps([{"id": "one", "title": ""}])

    def test_rejects_non_string_description(self) -> None:
        with self.assertRaisesRegex(PlannerError, "description must be a string"):
            _parse_plan_steps([{"id": "one", "title": "t", "description": 42}])


class TestParseMustContainRules(unittest.TestCase):
    """Verify must_contain parsing."""

    def test_parses_valid_rules(self) -> None:
        raw_rules = [
            {"path": "app.py", "strings": ["def add", "return"]},
        ]
        rules = _parse_must_contain_rules(raw_rules)
        self.assertEqual(1, len(rules))
        self.assertEqual("app.py", rules[0].path)
        self.assertEqual(("def add", "return"), rules[0].strings)

    def test_rejects_non_list(self) -> None:
        with self.assertRaisesRegex(PlannerError, "must_contain must be a list"):
            _parse_must_contain_rules("not list")

    def test_rejects_non_dict_rule(self) -> None:
        with self.assertRaisesRegex(PlannerError, "must be a dict"):
            _parse_must_contain_rules(["string"])

    def test_rejects_missing_path(self) -> None:
        with self.assertRaisesRegex(PlannerError, "path is required"):
            _parse_must_contain_rules([{"strings": ["test"]}])

    def test_rejects_empty_strings(self) -> None:
        with self.assertRaisesRegex(PlannerError, "non-empty list of strings"):
            _parse_must_contain_rules([{"path": "f.py", "strings": []}])


class TestParseVerificationSpecs(unittest.TestCase):
    """Verify verification parsing."""

    def test_parses_with_must_contain(self) -> None:
        raw_verification = [
            {
                "must_contain": [
                    {"path": "app.py", "strings": ["def run"]},
                ]
            }
        ]
        specs = _parse_verification_specs(raw_verification)
        self.assertEqual(1, len(specs))
        self.assertEqual(1, len(specs[0].must_contain))
        self.assertEqual("app.py", specs[0].must_contain[0].path)

    def test_parses_empty_must_contain(self) -> None:
        raw_verification = [{}]
        specs = _parse_verification_specs(raw_verification)
        self.assertEqual(1, len(specs))
        self.assertEqual(0, len(specs[0].must_contain))

    def test_parses_empty_array(self) -> None:
        specs = _parse_verification_specs([])
        self.assertEqual(0, len(specs))

    def test_rejects_non_list(self) -> None:
        with self.assertRaisesRegex(PlannerError, "verification must be a list"):
            _parse_verification_specs("not list")


class TestParseExpectedChangedFiles(unittest.TestCase):
    """Verify expected_changed_files parsing."""

    def test_parses_string_list(self) -> None:
        result = _parse_expected_changed_files(["src/a.py", "docs/readme.md"])
        self.assertEqual(("src/a.py", "docs/readme.md"), result)

    def test_normalizes_backslashes(self) -> None:
        result = _parse_expected_changed_files(["src\\a.py", "docs\\readme.md"])
        self.assertEqual(("src/a.py", "docs/readme.md"), result)

    def test_returns_empty_tuple_for_none(self) -> None:
        result = _parse_expected_changed_files(None)
        self.assertEqual((), result)

    def test_rejects_non_list(self) -> None:
        with self.assertRaisesRegex(PlannerError, "must be a list of strings"):
            _parse_expected_changed_files("not list")

    def test_rejects_empty_string(self) -> None:
        with self.assertRaisesRegex(PlannerError, "non-empty string"):
            _parse_expected_changed_files([""])

class TestBuildTaskPlanFromDict(unittest.TestCase):
    """Verify building TaskPlan from parsed dict."""

    def test_builds_valid_edit_plan(self) -> None:
        plan = _build_task_plan_from_dict(VALID_EDIT_JSON, max_steps=5)
        self.assertIsInstance(plan, TaskPlan)
        self.assertEqual("edit", plan.task_type)
        self.assertEqual("medium", plan.risk_level)
        self.assertEqual(5, plan.max_steps)
        self.assertTrue(plan.requires_patch)
        self.assertTrue(plan.requires_tests)
        self.assertEqual(("src/core.py",), plan.expected_changed_files)
        self.assertEqual(3, len(plan.steps))
        self.assertEqual(1, len(plan.verification))

    def test_accepts_single_must_contain_object_from_llm(self) -> None:
        raw: dict[str, object] = dict(
            VALID_EDIT_JSON,
            verification=[
                {
                    "must_contain": {"path": "src/core.py", "strings": ["def run"]},
                }
            ],
        )

        plan = _build_task_plan_from_dict(raw, max_steps=5)

        self.assertEqual(1, len(plan.verification))
        self.assertEqual(1, len(plan.verification[0].must_contain))
        self.assertEqual("src/core.py", plan.verification[0].must_contain[0].path)

    def test_builds_valid_analysis_plan(self) -> None:
        plan = _build_task_plan_from_dict(VALID_ANALYSIS_JSON, max_steps=8)
        self.assertIsInstance(plan, TaskPlan)
        self.assertEqual("analysis", plan.task_type)
        self.assertEqual("low", plan.risk_level)
        self.assertFalse(plan.requires_patch)
        self.assertFalse(plan.requires_tests)
        self.assertEqual(0, len(plan.verification))

    def test_rejects_invalid_task_type(self) -> None:
        raw: dict[str, object] = dict(VALID_EDIT_JSON, task_type="deploy")
        with self.assertRaisesRegex(PlannerError, "task_type must be one of"):
            _build_task_plan_from_dict(raw, max_steps=5)

    def test_rejects_invalid_risk_level(self) -> None:
        raw: dict[str, object] = dict(VALID_EDIT_JSON, risk_level="critical")
        with self.assertRaisesRegex(PlannerError, "risk_level must be one of"):
            _build_task_plan_from_dict(raw, max_steps=5)

    def test_rejects_missing_task_type(self) -> None:
        raw = dict(VALID_EDIT_JSON)
        del raw["task_type"]
        with self.assertRaisesRegex(PlannerError, "task_type is required"):
            _build_task_plan_from_dict(raw, max_steps=5)

    def test_rejects_missing_risk_level(self) -> None:
        raw = dict(VALID_EDIT_JSON)
        del raw["risk_level"]
        with self.assertRaisesRegex(PlannerError, "risk_level is required"):
            _build_task_plan_from_dict(raw, max_steps=5)

    def test_rejects_steps_exceeding_max_steps(self) -> None:
        raw = dict(VALID_EDIT_JSON)
        with self.assertRaisesRegex(PlannerError, "steps \u6570\u91cf\u4e0d\u80fd\u8d85\u8fc7 max_steps"):
            _build_task_plan_from_dict(raw, max_steps=2)

    def test_rejects_missing_requires_patch(self) -> None:
        raw = dict(VALID_EDIT_JSON)
        del raw["requires_patch"]
        with self.assertRaisesRegex(PlannerError, "requires_patch must be a boolean"):
            _build_task_plan_from_dict(raw, max_steps=5)

    def test_rejects_non_bool_requires_patch(self) -> None:
        raw: dict[str, object] = dict(VALID_EDIT_JSON, requires_patch="yes")
        with self.assertRaisesRegex(PlannerError, "requires_patch must be a boolean"):
            _build_task_plan_from_dict(raw, max_steps=5)

    def test_rejects_missing_requires_tests(self) -> None:
        raw = dict(VALID_EDIT_JSON)
        del raw["requires_tests"]
        with self.assertRaisesRegex(PlannerError, "requires_tests must be a boolean"):
            _build_task_plan_from_dict(raw, max_steps=5)

    def test_rejects_requires_patch_without_expected_files(self) -> None:
        raw: dict[str, object] = dict(VALID_EDIT_JSON, requires_patch=True, expected_changed_files=[])
        with self.assertRaisesRegex(PlannerError, "expected_changed_files"):
            _build_task_plan_from_dict(raw, max_steps=5)

    def test_rejects_edit_without_verification(self) -> None:
        raw: dict[str, object] = dict(VALID_EDIT_JSON, verification=[])
        with self.assertRaisesRegex(PlannerError, "verification"):
            _build_task_plan_from_dict(raw, max_steps=5)

    def test_rejects_duplicate_step_ids(self) -> None:
        raw: dict[str, object] = dict(
            VALID_ANALYSIS_JSON,
            steps=[
                {"id": "same", "title": "First step"},
                {"id": "same", "title": "Second step"},
            ],
        )
        with self.assertRaisesRegex(PlannerError, "step id"):
            _build_task_plan_from_dict(raw, max_steps=5)

    def test_rejects_empty_step_id(self) -> None:
        raw: dict[str, object] = dict(
            VALID_ANALYSIS_JSON,
            steps=[
                {"id": "", "title": "Empty id step"},
            ],
        )
        with self.assertRaisesRegex(PlannerError, "id is required"):
            _build_task_plan_from_dict(raw, max_steps=5)


class TestCreatePlanWithFakeLlm(unittest.TestCase):
    """Verify create_plan end-to-end with fake LLM."""

    def test_valid_edit_plan(self) -> None:
        fake_llm = FakeLlmClient([_valid_edit_json_str()])
        plan = create_plan(
            user_task="modify main logic",
            inspect_repo_result=VALID_INSPECT_REPO,
            max_steps=5,
            llm_client=fake_llm,
        )
        self.assertIsInstance(plan, TaskPlan)
        self.assertEqual("edit", plan.task_type)
        self.assertEqual(1, fake_llm.call_count)

    def test_valid_analysis_plan(self) -> None:
        fake_llm = FakeLlmClient([_valid_analysis_json_str()])
        plan = create_plan(
            user_task="analyze project structure",
            inspect_repo_result=VALID_INSPECT_REPO,
            max_steps=8,
            llm_client=fake_llm,
        )
        self.assertIsInstance(plan, TaskPlan)
        self.assertEqual("analysis", plan.task_type)

    def test_rejects_non_json_output(self) -> None:
        fake_llm = FakeLlmClient(["not JSON"])
        with self.assertRaisesRegex(PlannerError, "not valid JSON"):
            create_plan(
                user_task="task",
                inspect_repo_result=VALID_INSPECT_REPO,
                max_steps=5,
                llm_client=fake_llm,
            )

    def test_rejects_fenced_json(self) -> None:
        fenced = "```json\n" + _valid_edit_json_str() + "\n```"
        fake_llm = FakeLlmClient([fenced])
        with self.assertRaisesRegex(PlannerError, "markdown code fence"):
            create_plan(
                user_task="task",
                inspect_repo_result=VALID_INSPECT_REPO,
                max_steps=5,
                llm_client=fake_llm,
            )

    def test_rejects_extra_text_output(self) -> None:
        extra = "Here is the plan:\n" + _valid_edit_json_str()
        fake_llm = FakeLlmClient([extra])
        with self.assertRaisesRegex(PlannerError, "not valid JSON"):
            create_plan(
                user_task="task",
                inspect_repo_result=VALID_INSPECT_REPO,
                max_steps=5,
                llm_client=fake_llm,
            )

    def test_rejects_invalid_task_type_from_llm(self) -> None:
        raw = dict(VALID_EDIT_JSON, task_type="deploy")
        fake_llm = FakeLlmClient([json.dumps(raw, ensure_ascii=False)])
        with self.assertRaisesRegex(PlannerError, "task_type must be one of"):
            create_plan(
                user_task="task",
                inspect_repo_result=VALID_INSPECT_REPO,
                max_steps=5,
                llm_client=fake_llm,
            )

    def test_rejects_too_many_steps_from_llm(self) -> None:
        raw = dict(VALID_EDIT_JSON)
        fake_llm = FakeLlmClient([json.dumps(raw, ensure_ascii=False)])
        with self.assertRaisesRegex(PlannerError, "steps"):
            create_plan(
                user_task="task",
                inspect_repo_result=VALID_INSPECT_REPO,
                max_steps=2,
                llm_client=fake_llm,
            )

    def test_rejects_missing_required_fields_from_llm(self) -> None:
        raw = {
            "task_type": "analysis",
            "risk_level": "low",
        }
        fake_llm = FakeLlmClient([json.dumps(raw, ensure_ascii=False)])
        with self.assertRaisesRegex(PlannerError, "steps must be a list"):
            create_plan(
                user_task="task",
                inspect_repo_result=VALID_INSPECT_REPO,
                max_steps=5,
                llm_client=fake_llm,
            )

    def test_llm_failure_wraps_in_planner_error(self) -> None:
        class BrokenLlmClient:
            """A broken client that raises."""

            def chat(self, messages: list[dict[str, str]]) -> str:
                raise RuntimeError("network error")

        with self.assertRaisesRegex(PlannerError, "LLM call failed"):
            create_plan(
                user_task="task",
                inspect_repo_result=VALID_INSPECT_REPO,
                max_steps=5,
                llm_client=BrokenLlmClient(),
            )


if __name__ == "__main__":
    unittest.main()
