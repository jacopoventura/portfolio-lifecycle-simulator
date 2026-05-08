from unittest.mock import MagicMock, patch

import pytest

import agent
from agent import _DEFAULT_MODEL, _FALLBACK_MODEL, _SYSTEM_PROMPT, generate_action_plan


def _mock_message(text="AI advice.", input_tokens=100, output_tokens=50):
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    msg.usage.input_tokens = input_tokens
    msg.usage.output_tokens = output_tokens
    return msg


def _call(**kwargs):
    defaults = dict(config_toml="cfg", verbose_report="report", clean_summary="summary")
    defaults.update(kwargs)
    return generate_action_plan(**defaults)


class TestGenerateActionPlan:
    def test_returns_model_text(self):
        with patch("agent.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.return_value = _mock_message(text="Do X")
            assert _call() == "Do X"

    def test_user_message_contains_all_sections(self):
        with patch("agent.anthropic.Anthropic") as mock_cls:
            create = mock_cls.return_value.messages.create
            create.return_value = _mock_message()
            _call(config_toml="CFG_DATA", verbose_report="REPORT_DATA", clean_summary="SUMMARY_DATA")
            user_content = create.call_args[1]["messages"][0]["content"]
        assert "## Configuration" in user_content
        assert "CFG_DATA" in user_content
        assert "## Full Detailed Report" in user_content
        assert "REPORT_DATA" in user_content
        assert "## Clean Summary" in user_content
        assert "SUMMARY_DATA" in user_content

    def test_sections_appear_in_correct_order(self):
        with patch("agent.anthropic.Anthropic") as mock_cls:
            create = mock_cls.return_value.messages.create
            create.return_value = _mock_message()
            _call(config_toml="CFG", verbose_report="RPT", clean_summary="SUM")
            user_content = create.call_args[1]["messages"][0]["content"]
        cfg_pos = user_content.index("CFG")
        rpt_pos = user_content.index("RPT")
        sum_pos = user_content.index("SUM")
        assert cfg_pos < rpt_pos < sum_pos

    def test_system_prompt_passed(self):
        with patch("agent.anthropic.Anthropic") as mock_cls:
            create = mock_cls.return_value.messages.create
            create.return_value = _mock_message()
            _call()
        assert create.call_args[1]["system"] == _SYSTEM_PROMPT

    def test_default_model_used(self):
        with patch("agent.anthropic.Anthropic") as mock_cls:
            create = mock_cls.return_value.messages.create
            create.return_value = _mock_message()
            _call()
        assert create.call_args[1]["model"] == _DEFAULT_MODEL

    def test_custom_model_forwarded(self):
        with patch("agent.anthropic.Anthropic") as mock_cls:
            create = mock_cls.return_value.messages.create
            create.return_value = _mock_message()
            _call(model="my-model")
        assert create.call_args[1]["model"] == "my-model"

    def test_max_tokens_forwarded(self):
        with patch("agent.anthropic.Anthropic") as mock_cls:
            create = mock_cls.return_value.messages.create
            create.return_value = _mock_message()
            _call(max_tokens=500)
        assert create.call_args[1]["max_tokens"] == 500

    def test_fallback_model_on_not_found_error(self):
        fallback_msg = _mock_message(text="fallback")
        error = agent.anthropic.NotFoundError(
            message="model not found",
            response=MagicMock(headers={}, status_code=404),
            body={},
        )
        with patch("agent.anthropic.Anthropic") as mock_cls:
            create = mock_cls.return_value.messages.create
            create.side_effect = [error, fallback_msg]
            result = _call()
        assert result == "fallback"
        assert create.call_count == 2
        assert create.call_args_list[1][1]["model"] == _FALLBACK_MODEL

    def test_cost_and_tokens_printed(self, capsys):
        with patch("agent.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.return_value = _mock_message(
                input_tokens=1_000, output_tokens=200
            )
            _call()
        out = capsys.readouterr().out
        assert "1000" in out
        assert "200" in out
        assert "$" in out

    def test_cost_calculation(self, capsys):
        # cost = 1000 * 0.00000025 + 200 * 0.00000125 = 0.00025 + 0.00025 = 0.0005
        with patch("agent.anthropic.Anthropic") as mock_cls:
            mock_cls.return_value.messages.create.return_value = _mock_message(
                input_tokens=1_000, output_tokens=200
            )
            _call()
        out = capsys.readouterr().out
        assert "0.0005" in out

    def test_user_message_role(self):
        with patch("agent.anthropic.Anthropic") as mock_cls:
            create = mock_cls.return_value.messages.create
            create.return_value = _mock_message()
            _call()
        messages = create.call_args[1]["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
