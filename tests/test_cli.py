"""Tests for the CLI: help text quality, schema subcommand, and argument parsing."""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ai_pipelines.cli import _build_parser, _cli_schema, _parse_inputs, main


# ── _parse_inputs ─────────────────────────────────────────────────────────────

class TestParseInputs:
    def test_empty(self):
        assert _parse_inputs([], None) == {}

    def test_string_value(self):
        result = _parse_inputs(["key=hello"], None)
        assert result == {"key": "hello"}

    def test_integer_value_parsed_from_json(self):
        result = _parse_inputs(["n=42"], None)
        assert result["n"] == 42
        assert isinstance(result["n"], int)

    def test_boolean_value_parsed_from_json(self):
        result = _parse_inputs(["flag=true"], None)
        assert result["flag"] is True

    def test_array_value_parsed_from_json(self):
        result = _parse_inputs(["ids=[1,2,3]"], None)
        assert result["ids"] == [1, 2, 3]

    def test_string_fallback_for_non_json(self):
        result = _parse_inputs(["path=/tmp/foo"], None)
        assert result["path"] == "/tmp/foo"

    def test_equals_in_value(self):
        result = _parse_inputs(["expr=a=b"], None)
        assert result["expr"] == "a=b"

    def test_input_json_file(self, tmp_path):
        f = tmp_path / "inputs.json"
        f.write_text(json.dumps({"x": 1, "y": "hello"}))
        result = _parse_inputs([], f)
        assert result == {"x": 1, "y": "hello"}

    def test_cli_overrides_json_file(self, tmp_path):
        f = tmp_path / "inputs.json"
        f.write_text(json.dumps({"key": "from_file"}))
        result = _parse_inputs(["key=from_flag"], f)
        assert result["key"] == "from_flag"

    def test_missing_equals_exits(self):
        with pytest.raises(SystemExit):
            _parse_inputs(["noequals"], None)

    def test_input_json_array_exits(self, tmp_path):
        f = tmp_path / "bad.json"
        f.write_text("[1, 2, 3]")
        with pytest.raises(SystemExit):
            _parse_inputs([], f)


# ── _cli_schema ───────────────────────────────────────────────────────────────

class TestCliSchema:
    def setup_method(self):
        self.schema = _cli_schema()

    def test_top_level_keys_present(self):
        for key in ("tool", "description", "when_to_use", "not_for", "commands", "pipeline_yaml_format"):
            assert key in self.schema, f"Missing key: {key}"

    def test_tool_name(self):
        assert self.schema["tool"] == "ai-pipelines"

    def test_not_for_is_list_of_strings(self):
        assert isinstance(self.schema["not_for"], list)
        assert all(isinstance(s, str) for s in self.schema["not_for"])

    def test_commands_list(self):
        names = [c["name"] for c in self.schema["commands"]]
        assert "run" in names
        assert "validate" in names
        assert "schema" in names

    def test_run_command_has_required_structure(self):
        run = next(c for c in self.schema["commands"] if c["name"] == "run")
        assert "description" in run
        assert "arguments" in run
        assert "output" in run
        assert "exit_codes" in run
        assert "examples" in run

    def test_run_arguments_have_descriptions(self):
        run = next(c for c in self.schema["commands"] if c["name"] == "run")
        for arg_name, arg_spec in run["arguments"].items():
            assert "description" in arg_spec, f"Missing description on argument {arg_name}"

    def test_run_examples_are_non_empty(self):
        run = next(c for c in self.schema["commands"] if c["name"] == "run")
        assert len(run["examples"]) >= 3
        for ex in run["examples"]:
            assert "description" in ex
            assert "command" in ex

    def test_validate_exit_codes(self):
        validate = next(c for c in self.schema["commands"] if c["name"] == "validate")
        assert "0" in validate["exit_codes"]
        assert "1" in validate["exit_codes"]

    def test_pipeline_yaml_format_has_step_kinds(self):
        fmt = self.schema["pipeline_yaml_format"]
        kinds = [s["kind"] for s in fmt["step_kinds"]]
        for expected in ("find_files", "read_file", "transform", "chunk", "prompt", "evaluate", "for_each"):
            assert expected in kinds

    def test_reserved_step_names_listed(self):
        fmt = self.schema["pipeline_yaml_format"]
        assert "input" in fmt["reserved_step_names"]
        assert "item" in fmt["reserved_step_names"]
        assert "item_index" in fmt["reserved_step_names"]

    def test_common_mistakes_non_empty(self):
        fmt = self.schema["pipeline_yaml_format"]
        assert len(fmt["common_mistakes"]) >= 1

    def test_schema_is_json_serialisable(self):
        # Should not raise
        dumped = json.dumps(self.schema)
        reloaded = json.loads(dumped)
        assert reloaded["tool"] == "ai-pipelines"


# ── schema subcommand (CLI invocation) ───────────────────────────────────────

class TestSchemaSubcommand:
    def test_schema_outputs_valid_json(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["schema"])
        assert exc.value.code == 0
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert parsed["tool"] == "ai-pipelines"

    def test_schema_contains_all_commands(self, capsys):
        with pytest.raises(SystemExit):
            main(["schema"])
        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        names = {c["name"] for c in parsed["commands"]}
        assert names == {"run", "validate", "schema"}

    def test_schema_nothing_written_to_stderr(self, capsys):
        with pytest.raises(SystemExit):
            main(["schema"])
        captured = capsys.readouterr()
        assert captured.err == ""


# ── Help text quality ─────────────────────────────────────────────────────────

class TestHelpText:
    def _get_help(self, argv: list[str]) -> str:
        """Capture --help output as a string."""
        with pytest.raises(SystemExit):
            main(argv)
        # argparse writes --help to stdout
        return ""  # handled via capsys in individual tests

    def test_top_level_help_exits_zero(self):
        with pytest.raises(SystemExit) as exc:
            main(["--help"])
        assert exc.value.code == 0

    def test_top_level_help_mentions_not_for(self, capsys):
        with pytest.raises(SystemExit):
            main(["--help"])
        out = capsys.readouterr().out
        assert "Do NOT" in out

    def test_top_level_help_mentions_schema_subcommand(self, capsys):
        with pytest.raises(SystemExit):
            main(["--help"])
        out = capsys.readouterr().out
        assert "schema" in out

    def test_run_help_exits_zero(self):
        with pytest.raises(SystemExit) as exc:
            main(["run", "--help"])
        assert exc.value.code == 0

    def test_run_help_mentions_output_schema(self, capsys):
        with pytest.raises(SystemExit):
            main(["run", "--help"])
        out = capsys.readouterr().out
        assert "total_cost_usd" in out

    def test_run_help_mentions_input_parsing_rule(self, capsys):
        with pytest.raises(SystemExit):
            main(["run", "--help"])
        out = capsys.readouterr().out
        assert "JSON" in out

    def test_run_help_has_examples(self, capsys):
        with pytest.raises(SystemExit):
            main(["run", "--help"])
        out = capsys.readouterr().out
        assert "Examples:" in out

    def test_run_help_has_not_for_section(self, capsys):
        with pytest.raises(SystemExit):
            main(["run", "--help"])
        out = capsys.readouterr().out
        assert "NOT" in out or "does not" in out.lower() or "not for" in out.lower()

    def test_validate_help_exits_zero(self):
        with pytest.raises(SystemExit) as exc:
            main(["validate", "--help"])
        assert exc.value.code == 0

    def test_validate_help_mentions_exit_codes(self, capsys):
        with pytest.raises(SystemExit):
            main(["validate", "--help"])
        out = capsys.readouterr().out
        assert "Exit codes" in out or "exit code" in out.lower()

    def test_validate_help_mentions_no_llm_calls(self, capsys):
        with pytest.raises(SystemExit):
            main(["validate", "--help"])
        out = capsys.readouterr().out
        assert "LLM" in out or "llm" in out.lower()


# ── No-command behaviour ──────────────────────────────────────────────────────

class TestNoCommand:
    def test_no_args_exits_zero(self):
        with pytest.raises(SystemExit) as exc:
            main([])
        assert exc.value.code == 0

    def test_no_args_prints_help(self, capsys):
        with pytest.raises(SystemExit):
            main([])
        out = capsys.readouterr().out
        assert "ai-pipelines" in out
        assert "run" in out
        assert "validate" in out
