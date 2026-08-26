"""Azure GPT-OSS empty-content handling (no network, no Azure key)."""

from __future__ import annotations

import sys
import unittest
from types import SimpleNamespace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from runtime import (
    azure_answer_text,
    azure_chat_create,
    azure_create_kwargs,
    content_to_text,
    describe_completion,
    empty_retry_tokens,
    extract_reasoning_text,
    extract_visible_text,
    followup_max_new_tokens,
    is_unsupported_parameter,
)


def make_response(
    content="",
    finish_reason="length",
    reasoning_tokens=300,
    reasoning_content="",
    completion_tokens=None,
):
    message = SimpleNamespace(
        content=content,
        reasoning_content=reasoning_content or None,
    )
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    details = SimpleNamespace(reasoning_tokens=reasoning_tokens)
    usage = SimpleNamespace(
        completion_tokens=completion_tokens if completion_tokens is not None else reasoning_tokens,
        completion_tokens_details=details,
    )
    return SimpleNamespace(choices=[choice], usage=usage)


class ScriptedClient:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.chat = SimpleNamespace(completions=self)

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self.script:
            raise AssertionError(f"unexpected extra Azure call: {kwargs}")
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class ExtractTests(unittest.TestCase):
    def test_visible_text_ignores_hidden_reasoning(self):
        message = SimpleNamespace(
            content="",
            reasoning_content="The capital is Paris because...",
        )
        self.assertEqual(extract_visible_text(message), "")
        self.assertIn("Paris", extract_reasoning_text(message))

    def test_list_content_parts(self):
        self.assertEqual(
            content_to_text([{"type": "text", "text": "Hello"}, {"type": "text", "text": "world"}]),
            "Hello\nworld",
        )

    def test_describe_length_with_reasoning_tokens(self):
        response = make_response(content="", finish_reason="length", reasoning_tokens=300)
        text = describe_completion(response.choices[0], response.usage)
        self.assertIn("finish_reason=length", text)
        self.assertIn("reasoning_tokens=300", text)


class PayloadTests(unittest.TestCase):
    def test_prefers_max_tokens_and_low_effort(self):
        kwargs = azure_create_kwargs(
            "gpt-oss-20b",
            [{"role": "user", "content": "hi"}],
            32768,
            temperature=0.7,
            send_temperature=False,
            reasoning_effort="low",
        )
        self.assertEqual(kwargs["max_tokens"], 32768)
        self.assertNotIn("max_completion_tokens", kwargs)
        self.assertNotIn("temperature", kwargs)
        self.assertEqual(kwargs["reasoning_effort"], "low")

    def test_empty_retry_is_at_least_4096_and_capped(self):
        self.assertEqual(empty_retry_tokens(300), 4096)
        self.assertEqual(empty_retry_tokens(3000), 6000)
        self.assertEqual(empty_retry_tokens(32768), 32768)

    def test_followup_budget_uses_max_tokens_default(self):
        self.assertGreaterEqual(followup_max_new_tokens(), 32768)

    def test_temperature_rejection_is_detected(self):
        error = ValueError("Unsupported value: 'temperature' does not support 0.7")
        self.assertTrue(is_unsupported_parameter(error, "temperature"))
        self.assertFalse(is_unsupported_parameter(error, "reasoning_effort"))


class AzureAnswerTests(unittest.TestCase):
    def test_retries_empty_content_then_returns_visible_text(self):
        client = ScriptedClient(
            [
                make_response(content="", finish_reason="length", reasoning_tokens=300),
                make_response(content="The treaty was signed in 1815.", finish_reason="stop"),
            ]
        )
        text = azure_answer_text(
            client,
            [{"role": "user", "content": "When was it signed?"}],
            300,
            model_name="gpt-oss-20b",
            send_temperature=False,
            reasoning_effort="low",
            use_reasoning_fallback=False,
        )
        self.assertEqual(text, "The treaty was signed in 1815.")
        self.assertEqual(len(client.calls), 2)
        self.assertEqual(client.calls[0]["max_tokens"], 300)
        self.assertEqual(client.calls[1]["max_tokens"], 4096)
        self.assertEqual(client.calls[0]["reasoning_effort"], "low")

    def test_does_not_use_reasoning_as_the_public_answer_when_fallback_off(self):
        client = ScriptedClient(
            [
                make_response(
                    content="",
                    finish_reason="length",
                    reasoning_tokens=300,
                    reasoning_content="hidden chain of thought",
                ),
                make_response(
                    content="",
                    finish_reason="length",
                    reasoning_tokens=4096,
                    reasoning_content="hidden chain of thought",
                ),
            ]
        )
        text = azure_answer_text(
            client,
            [{"role": "user", "content": "q"}],
            300,
            model_name="gpt-oss-20b",
            send_temperature=False,
            reasoning_effort="low",
            use_reasoning_fallback=False,
        )
        self.assertEqual(text, "")

    def test_last_resort_uses_reasoning_content_after_retry(self):
        client = ScriptedClient(
            [
                make_response(content="", finish_reason="length", reasoning_content="first"),
                make_response(
                    content="",
                    finish_reason="length",
                    reasoning_content="The compound is X47.",
                ),
            ]
        )
        text = azure_answer_text(
            client,
            [{"role": "user", "content": "q"}],
            300,
            model_name="gpt-oss-20b",
            send_temperature=False,
            reasoning_effort="low",
            use_reasoning_fallback=True,
        )
        self.assertEqual(text, "The compound is X47.")

    def test_drops_unsupported_temperature_and_retries(self):
        client = ScriptedClient(
            [
                ValueError("Unsupported parameter: 'temperature' is not supported for this model"),
                make_response(content="Visible answer.", finish_reason="stop"),
            ]
        )
        text = azure_answer_text(
            client,
            [{"role": "user", "content": "q"}],
            2048,
            temperature=0.7,
            model_name="gpt-oss-20b",
            send_temperature=True,
            reasoning_effort="low",
            use_reasoning_fallback=False,
        )
        self.assertEqual(text, "Visible answer.")
        self.assertIn("temperature", client.calls[0])
        self.assertNotIn("temperature", client.calls[1])

    def test_moves_reasoning_effort_to_extra_body_on_typeerror(self):
        client = ScriptedClient(
            [
                TypeError("create() got an unexpected keyword argument 'reasoning_effort'"),
                make_response(content="ok", finish_reason="stop"),
            ]
        )
        text = azure_answer_text(
            client,
            [{"role": "user", "content": "q"}],
            2048,
            model_name="gpt-oss-20b",
            send_temperature=False,
            reasoning_effort="low",
            use_reasoning_fallback=False,
        )
        self.assertEqual(text, "ok")
        self.assertNotIn("reasoning_effort", client.calls[1])
        self.assertEqual(client.calls[1].get("extra_body", {}).get("reasoning_effort"), "low")


class AzureCreateLoopTests(unittest.TestCase):
    def test_max_tokens_falls_back_to_max_completion_tokens(self):
        client = ScriptedClient(
            [
                ValueError("Unsupported parameter: max_tokens"),
                make_response(content="ok", finish_reason="stop"),
            ]
        )
        response = azure_chat_create(
            client,
            {
                "model": "gpt-oss-20b",
                "messages": [{"role": "user", "content": "q"}],
                "max_tokens": 32768,
            },
        )
        self.assertEqual(response.choices[0].message.content, "ok")
        self.assertNotIn("max_tokens", client.calls[1])
        self.assertEqual(client.calls[1]["max_completion_tokens"], 32768)


class CredentialAndTlsTests(unittest.TestCase):
    def test_readme_endpoint_is_a_placeholder(self):
        from runtime import looks_like_placeholder
        self.assertTrue(looks_like_placeholder("https://YOUR-RESOURCE.openai.azure.com/"))
        self.assertTrue(looks_like_placeholder("(new key after rotate)"))
        self.assertTrue(looks_like_placeholder("<your-key>"))
        self.assertFalse(looks_like_placeholder("https://myres.openai.azure.com"))
        self.assertFalse(looks_like_placeholder("sk-proj-abc123"))

    def test_require_azure_rejects_placeholder_host(self):
        import os
        from unittest import mock
        from runtime import require_azure_credentials
        env = {
            "AZURE_OPENAI_ENDPOINT": "https://YOUR-RESOURCE.openai.azure.com/",
            "AZURE_OPENAI_API_KEY": "not-a-placeholder-key",
            "AZURE_ENDPOINT": "",
            "AZURE_API_KEY": "",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            with self.assertRaises(SystemExit) as ctx:
                require_azure_credentials()
        self.assertIn("YOUR-RESOURCE", str(ctx.exception))

    def test_require_openai_rejects_example_key(self):
        import os
        from unittest import mock
        from runtime import require_openai_key
        with mock.patch.dict(os.environ, {"OPENAI_API_KEY": "(new key after rotate)"}, clear=False):
            with self.assertRaises(SystemExit) as ctx:
                require_openai_key()
        self.assertIn("example string", str(ctx.exception))

    def test_handshake_hint_mentions_http1_and_python312(self):
        from runtime import connection_error_hint
        text = connection_error_hint(RuntimeError("[SSL: SSLV3_ALERT_HANDSHAKE_FAILURE] handshake failure"))
        self.assertIn("HTTP/1.1", text)
        self.assertIn("python@3.12", text)

    def test_reraise_wraps_connect_error(self):
        from runtime import reraise_connection_error
        with self.assertRaises(SystemExit) as ctx:
            reraise_connection_error(RuntimeError("[SSL: CERTIFICATE_VERIFY_FAILED] unable to get local issuer certificate"))
        self.assertIn("Command Line Tools", str(ctx.exception))

    def test_http_client_uses_explicit_verify(self):
        import runtime
        runtime._http_client = None
        try:
            client = runtime.make_openai_http_client()
        except RuntimeError as err:
            if "Need httpx or openai" in str(err):
                self.skipTest("httpx/openai not installed in this environment")
            raise
        try:
            self.assertIsNotNone(client)
        finally:
            if runtime._http_client is not None:
                try:
                    runtime._http_client.close()
                except Exception:
                    pass
                runtime._http_client = None


if __name__ == "__main__":
    unittest.main()
