# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for interactive CLI utilities."""

import asyncio
import threading
import unittest
from unittest import mock

from google.antigravity import agent
from google.antigravity import types
from google.antigravity.connections import local as local_connection
from google.antigravity.conversation import conversation
from google.antigravity.hooks import hooks
from google.antigravity.hooks import policy
from google.antigravity.utils import interactive


class AsyncInputTest(unittest.IsolatedAsyncioTestCase):
  """Tests for async_input."""

  @mock.patch("builtins.input")
  async def test_returns_user_input(self, mock_input):
    """Verifies that async_input returns the value from input()."""
    mock_input.return_value = "hello"
    result = await interactive.async_input("prompt> ")
    self.assertEqual(result, "hello")
    mock_input.assert_called_once_with("prompt> ")

  @mock.patch("builtins.input")
  async def test_default_prompt(self, mock_input):
    """Verifies that async_input passes an empty prompt by default."""
    mock_input.return_value = "test"
    result = await interactive.async_input()
    self.assertEqual(result, "test")
    mock_input.assert_called_once_with("")

  @mock.patch("builtins.input")
  async def test_propagates_eof_error(self, mock_input):
    """Verifies that EOFError from input() is propagated."""
    mock_input.side_effect = EOFError("end of file")
    with self.assertRaises(EOFError):
      await interactive.async_input("prompt> ")

  @mock.patch("builtins.input")
  async def test_cancellation(self, mock_input):
    """Verifies that cancelling the future does not crash the thread."""
    started = threading.Event()
    blocker = threading.Event()

    def blocking_input(prompt):
      del prompt
      started.set()
      blocker.wait()
      return "unused"

    mock_input.side_effect = blocking_input

    task = asyncio.create_task(interactive.async_input("prompt> "))
    # Wait for the thread to actually start and call input().
    await asyncio.get_event_loop().run_in_executor(None, started.wait)
    task.cancel()
    with self.assertRaises(asyncio.CancelledError):
      await task
    blocker.set()  # Let the daemon thread exit cleanly.


class ToolConfirmationHookTest(unittest.TestCase):
  """Tests for ToolConfirmationHook."""

  def setUp(self):
    super().setUp()
    self.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(self.loop)
    self.ctx = hooks.HookContext()

  def tearDown(self):
    super().tearDown()
    self.loop.close()
    asyncio.set_event_loop(None)

  @mock.patch("builtins.input")
  def test_tool_confirmation_hook_allow(self, mock_input):
    """Verifies that the hook allows execution when the user confirms.

    What: Tests ToolConfirmationHook with 'y' input.
    Why: Ensures positive confirmation allows tool execution.
    How: Asserts that the returned HookResult has allow=True.

    Args:
      mock_input: The patched builtins.input function.
    """
    mock_input.return_value = "y"
    hook = interactive.ToolConfirmationHook()
    tool_call = types.ToolCall(name="test_tool", args={"foo": "bar"})
    res = self.loop.run_until_complete(hook.run(self.ctx, tool_call))
    self.assertTrue(res.allow)

  @mock.patch("builtins.input")
  def test_tool_confirmation_hook_deny(self, mock_input):
    """Verifies that the hook denies execution when the user declines.

    What: Tests ToolConfirmationHook with 'n' input.
    Why: Ensures negative confirmation blocks tool execution.
    How: Asserts that the returned HookResult has allow=False.

    Args:
      mock_input: The patched builtins.input function.
    """
    mock_input.return_value = "n"
    hook = interactive.ToolConfirmationHook()
    tool_call = types.ToolCall(name="test_tool", args={})
    res = self.loop.run_until_complete(hook.run(self.ctx, tool_call))
    self.assertFalse(res.allow)
    self.assertEqual(res.message, "User denied tool call.")

  @mock.patch("builtins.input")
  def test_tool_confirmation_hook_eof(self, mock_input):
    """Verifies that the hook denies execution on EOFError.

    What: Tests ToolConfirmationHook when input raises EOFError.
    Why: Ensures non-interactive execution defaults to denial.
    How: Asserts that the returned HookResult has allow=False.

    Args:
      mock_input: The patched builtins.input function.
    """
    mock_input.side_effect = EOFError
    hook = interactive.ToolConfirmationHook()
    tool_call = types.ToolCall(name="test_tool", args={})
    res = self.loop.run_until_complete(hook.run(self.ctx, tool_call))
    self.assertFalse(res.allow)


class AskQuestionHookTest(unittest.TestCase):
  """Tests for AskQuestionHook."""

  def setUp(self):
    super().setUp()
    self.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(self.loop)

  def tearDown(self):
    super().tearDown()
    self.loop.close()
    asyncio.set_event_loop(None)

  @mock.patch("builtins.input")
  def test_ask_question_hook_option_number(self, mock_input):
    """Verifies that the user can select an option by its index."""
    mock_input.return_value = "1"
    hook = interactive.AskQuestionHook()
    q = types.AskQuestionEntry(
        question="Choose?",
        options=[
            types.AskQuestionOption(id="opt1", text="Option 1"),
            types.AskQuestionOption(id="opt2", text="Option 2"),
        ],
    )
    spec = types.AskQuestionInteractionSpec(questions=[q])
    session_ctx = hooks.SessionContext()
    turn_ctx = hooks.TurnContext(session_ctx)
    op_ctx = hooks.OperationContext(turn_ctx)
    res = self.loop.run_until_complete(hook.run(op_ctx, spec))
    self.assertEqual(len(res.responses), 1)
    self.assertEqual(res.responses[0].selected_option_ids, ["opt1"])

  @mock.patch("builtins.input")
  def test_ask_question_hook_option_text(self, mock_input):
    """Verifies that the user can select an option by its exact text."""
    mock_input.return_value = "Option 2"
    hook = interactive.AskQuestionHook()
    q = types.AskQuestionEntry(
        question="Choose?",
        options=[
            types.AskQuestionOption(id="opt1", text="Option 1"),
            types.AskQuestionOption(id="opt2", text="Option 2"),
        ],
    )
    spec = types.AskQuestionInteractionSpec(questions=[q])
    session_ctx = hooks.SessionContext()
    turn_ctx = hooks.TurnContext(session_ctx)
    op_ctx = hooks.OperationContext(turn_ctx)
    res = self.loop.run_until_complete(hook.run(op_ctx, spec))
    self.assertEqual(len(res.responses), 1)
    self.assertEqual(res.responses[0].selected_option_ids, ["opt2"])

  @mock.patch("builtins.input")
  def test_ask_question_hook_write_in(self, mock_input):
    """Verifies that the user can provide a write-in response."""
    mock_input.return_value = "custom answer"
    hook = interactive.AskQuestionHook()
    q = types.AskQuestionEntry(question="What?", options=[])
    spec = types.AskQuestionInteractionSpec(questions=[q])
    session_ctx = hooks.SessionContext()
    turn_ctx = hooks.TurnContext(session_ctx)
    op_ctx = hooks.OperationContext(turn_ctx)
    res = self.loop.run_until_complete(hook.run(op_ctx, spec))
    self.assertEqual(len(res.responses), 1)
    self.assertEqual(res.responses[0].freeform_response, "custom answer")

  @mock.patch("builtins.input")
  def test_ask_question_hook_skip(self, mock_input):
    """Verifies that the user can skip a question by providing empty input."""
    mock_input.return_value = ""
    hook = interactive.AskQuestionHook()
    q = types.AskQuestionEntry(question="What?", options=[])
    spec = types.AskQuestionInteractionSpec(questions=[q])
    session_ctx = hooks.SessionContext()
    turn_ctx = hooks.TurnContext(session_ctx)
    op_ctx = hooks.OperationContext(turn_ctx)
    res = self.loop.run_until_complete(hook.run(op_ctx, spec))
    self.assertEqual(len(res.responses), 1)
    self.assertTrue(res.responses[0].skipped)

  @mock.patch("builtins.input")
  def test_ask_question_hook_eof(self, mock_input):
    """Verifies that EOFError results in a cancelled response."""
    mock_input.side_effect = EOFError
    hook = interactive.AskQuestionHook()
    q = types.AskQuestionEntry(question="What?", options=[])
    spec = types.AskQuestionInteractionSpec(questions=[q])
    session_ctx = hooks.SessionContext()
    turn_ctx = hooks.TurnContext(session_ctx)
    op_ctx = hooks.OperationContext(turn_ctx)
    res = self.loop.run_until_complete(hook.run(op_ctx, spec))
    self.assertFalse(res.responses)
    self.assertTrue(res.cancelled)


class AskUserHandlerTest(unittest.TestCase):
  """Tests for ask_user_handler."""

  def setUp(self):
    super().setUp()
    self.loop = asyncio.new_event_loop()
    asyncio.set_event_loop(self.loop)

  def tearDown(self):
    super().tearDown()
    self.loop.close()
    asyncio.set_event_loop(None)

  @mock.patch("builtins.input")
  def test_ask_user_handler_allow(self, mock_input):
    """Verifies that the handler returns True when the user confirms."""
    mock_input.return_value = "y"
    tc = types.ToolCall(name="test_tool", args={"key": "val"})
    result = self.loop.run_until_complete(interactive.ask_user_handler(tc))
    self.assertTrue(result)

  @mock.patch("builtins.input")
  def test_ask_user_handler_deny(self, mock_input):
    """Verifies that the handler returns False when the user declines."""
    mock_input.return_value = "n"
    tc = types.ToolCall(name="test_tool", args={})
    result = self.loop.run_until_complete(interactive.ask_user_handler(tc))
    self.assertFalse(result)

  @mock.patch("builtins.input")
  def test_ask_user_handler_eof(self, mock_input):
    """Verifies that the handler returns False on EOFError."""
    mock_input.side_effect = EOFError
    tc = types.ToolCall(name="test_tool", args={})
    result = self.loop.run_until_complete(interactive.ask_user_handler(tc))
    self.assertFalse(result)


class UpgradePoliciesListTest(unittest.TestCase):
  """Tests for _upgrade_policies_list."""

  def test_upgrade_policies_list(self):
    """Verifies that deny policies on run_command are converted to ask_user."""
    deny_policy = policy.deny(types.BuiltinTools.RUN_COMMAND.value)
    other_policy = policy.allow("other_tool")
    policies = [deny_policy, other_policy]

    upgraded = interactive._upgrade_policies_list(policies)
    self.assertEqual(len(upgraded), 2)
    self.assertEqual(upgraded[0].decision, policy.Decision.ASK_USER)
    self.assertEqual(upgraded[0].tool, types.BuiltinTools.RUN_COMMAND.value)
    self.assertEqual(upgraded[1], other_policy)

  def test_upgrade_nested_policy_sequences(self):
    """Verifies that nested policy lists from builders are flattened and upgraded."""
    policies = [
        policy.confirm_run_command(),
    ]
    upgraded = interactive._upgrade_policies_list(policies)
    self.assertEqual(len(upgraded), 2)
    self.assertEqual(upgraded[0].decision, policy.Decision.ASK_USER)
    self.assertEqual(upgraded[0].tool, types.BuiltinTools.RUN_COMMAND.value)
    self.assertEqual(upgraded[1].decision, policy.Decision.APPROVE)

  def test_preserves_conditional_deny(self):
    """Verifies that conditional deny policies (with a `when` predicate) are kept."""
    p_cond = policy.deny(
        types.BuiltinTools.RUN_COMMAND.value,
        when=lambda tc: "rm" in tc.args.get("cmd", ""),
    )
    upgraded = interactive._upgrade_policies_list([p_cond])
    self.assertEqual(len(upgraded), 1)
    self.assertEqual(upgraded[0].decision, policy.Decision.DENY)
    self.assertIsNotNone(upgraded[0].when)

  def test_upgrade_auto_policy_preserves_model(self):
    """Verifies that when an auto policy is upgraded to ASK_USER, the model is preserved."""
    p_auto = policy.auto(model="my-custom-model")
    upgraded = interactive._upgrade_policies_list([p_auto])
    self.assertEqual(len(upgraded), 1)
    self.assertEqual(upgraded[0].decision, policy.Decision.ASK_USER)
    self.assertTrue(upgraded[0].auto)
    self.assertEqual(upgraded[0].model, "my-custom-model")


class RunInteractiveLoopTest(unittest.IsolatedAsyncioTestCase):
  """Tests for run_interactive_loop."""

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  @mock.patch(
      "google.antigravity.utils.interactive.async_input",
      new_callable=mock.AsyncMock,
  )
  async def test_run_interactive_loop(
      self, mock_async_input, mock_conv_create, mock_strategy_class
  ):
    """Verifies the basic interactive loop flow.

    What: Simulates empty input, a valid prompt, and 'exit'.
    Why: Ensures the loop correctly skips blanks, sends prompts,
         prints responses, and exits on 'exit'.
    How: Mocks async_input (stdin) and conversation methods,
         then asserts send was called and output was printed.
    """
    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    mock_conversation = mock.MagicMock(spec=conversation.Conversation)
    mock_conversation._connection = mock.MagicMock()
    mock_conversation.send = mock.AsyncMock()

    async def mock_receive_steps():
      yield types.Step(is_complete_response=True, content="Agent response")

    mock_conversation.receive_steps = mock_receive_steps

    mock_cm = mock.AsyncMock()
    mock_cm.__aenter__.return_value = mock_conversation
    mock_conv_create.return_value = mock_cm

    # Mock input to return '', 'hello' then 'exit'
    mock_async_input.side_effect = ["", "hello", "exit"]

    config = local_connection.LocalAgentConfig(system_instructions="test")
    with mock.patch("builtins.print") as mock_print:
      await interactive.run_interactive_loop(config)

    mock_conversation.send.assert_called_once_with("hello")
    mock_print.assert_any_call("Agent: Agent response")

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  @mock.patch(
      "google.antigravity.utils.interactive.async_input",
      new_callable=mock.AsyncMock,
  )
  async def test_run_interactive_loop_interrupt(
      self, mock_async_input, mock_conv_create, mock_strategy_class
  ):
    """Verifies clean exit on KeyboardInterrupt.

    What: Simulates Ctrl+C during input.
    Why: Ensures graceful shutdown without traceback.
    How: Asserts 'Goodbye!' is printed after KeyboardInterrupt.
    """
    del mock_conv_create  # Unused.
    mock_strategy_instance = mock.MagicMock()
    mock_strategy_instance.stop = mock.AsyncMock()
    mock_strategy_class.return_value = mock_strategy_instance

    mock_async_input.side_effect = KeyboardInterrupt()

    config = local_connection.LocalAgentConfig(system_instructions="test")
    with mock.patch("builtins.print") as mock_print:
      await interactive.run_interactive_loop(config)

    mock_print.assert_any_call("\nGoodbye!")

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  @mock.patch(
      "google.antigravity.utils.interactive.async_input",
      new_callable=mock.AsyncMock,
  )
  async def test_run_interactive_loop_appends_ask_question_hook_when_missing(
      self, mock_async_input, mock_conv_create, mock_strategy_class
  ):
    """Verifies that AskQuestionHook is added when not present."""
    del mock_strategy_class, mock_conv_create  # Unused.
    mock_async_input.side_effect = ["exit"]

    mock_agent_instance = mock.MagicMock(spec=agent.Agent)
    mock_agent_instance.__aenter__ = mock.AsyncMock(
        return_value=mock_agent_instance
    )
    mock_agent_instance.__aexit__ = mock.AsyncMock()
    mock_agent_class = mock.Mock(return_value=mock_agent_instance)

    config = local_connection.LocalAgentConfig(system_instructions="test")
    self.assertEqual(len(config.hooks), 0)

    with mock.patch("builtins.print"):
      await interactive.run_interactive_loop(
          config, agent_class=mock_agent_class
      )

    mock_agent_class.assert_called_once()
    called_config = mock_agent_class.call_args[0][0]
    self.assertEqual(len(called_config.hooks), 1)
    self.assertIsInstance(called_config.hooks[0], interactive.AskQuestionHook)

  @mock.patch(
      "google.antigravity.connections."
      "local.local_connection.LocalConnectionStrategy"
  )
  @mock.patch.object(conversation.Conversation, "create")
  @mock.patch(
      "google.antigravity.utils.interactive.async_input",
      new_callable=mock.AsyncMock,
  )
  async def test_run_interactive_loop_does_not_duplicate_ask_question_hook_when_present(
      self, mock_async_input, mock_conv_create, mock_strategy_class
  ):
    """Verifies that AskQuestionHook is not duplicated if already present."""
    del mock_strategy_class, mock_conv_create  # Unused.
    mock_async_input.side_effect = ["exit"]

    mock_agent_instance = mock.MagicMock(spec=agent.Agent)
    mock_agent_instance.__aenter__ = mock.AsyncMock(
        return_value=mock_agent_instance
    )
    mock_agent_instance.__aexit__ = mock.AsyncMock()
    mock_agent_class = mock.Mock(return_value=mock_agent_instance)

    existing_hook = interactive.AskQuestionHook()
    config = local_connection.LocalAgentConfig(
        system_instructions="test", hooks=[existing_hook]
    )

    with mock.patch("builtins.print"):
      await interactive.run_interactive_loop(
          config, agent_class=mock_agent_class
      )

    mock_agent_class.assert_called_once()
    called_config = mock_agent_class.call_args[0][0]
    self.assertEqual(len(called_config.hooks), 1)
    self.assertIs(called_config.hooks[0], existing_hook)

  @mock.patch(
      "google.antigravity.utils.interactive.async_input",
      new_callable=mock.AsyncMock,
  )
  async def test_run_interactive_loop_upgrades_agent_behavior_to_interactive(
      self, mock_async_input
  ):
    """Verifies that agent_behavior is upgraded to INTERACTIVE.

    Also verifies that other configured capability fields are preserved.
    """
    mock_async_input.side_effect = ["exit"]

    mock_agent_instance = mock.MagicMock(spec=agent.Agent)
    mock_agent_instance.__aenter__ = mock.AsyncMock(
        return_value=mock_agent_instance
    )
    mock_agent_instance.__aexit__ = mock.AsyncMock()
    mock_agent_class = mock.Mock(return_value=mock_agent_instance)

    config = local_connection.LocalAgentConfig(
        system_instructions="test",
        capabilities=types.CapabilitiesConfig(
            enable_subagents=True,
            disabled_tools=[types.BuiltinTools.SEARCH_WEB],
        ),
    )
    self.assertEqual(
        config.capabilities.agent_behavior, types.AgentBehavior.AUTONOMOUS
    )

    with mock.patch("builtins.print"):
      await interactive.run_interactive_loop(
          config, agent_class=mock_agent_class
      )

    mock_agent_class.assert_called_once()
    called_config = mock_agent_class.call_args[0][0]
    self.assertEqual(
        called_config.capabilities.agent_behavior,
        types.AgentBehavior.INTERACTIVE,
    )
    self.assertTrue(called_config.capabilities.enable_subagents)
    self.assertEqual(
        called_config.capabilities.disabled_tools,
        [types.BuiltinTools.SEARCH_WEB],
    )

  @mock.patch(
      "google.antigravity.utils.interactive.async_input",
      new_callable=mock.AsyncMock,
  )
  async def test_run_interactive_loop_preserves_existing_interactive_behavior(
      self, mock_async_input
  ):
    """Verifies that agent_behavior remains INTERACTIVE if already configured."""
    mock_async_input.side_effect = ["exit"]

    mock_agent_instance = mock.MagicMock(spec=agent.Agent)
    mock_agent_instance.__aenter__ = mock.AsyncMock(
        return_value=mock_agent_instance
    )
    mock_agent_instance.__aexit__ = mock.AsyncMock()
    mock_agent_class = mock.Mock(return_value=mock_agent_instance)

    config = local_connection.LocalAgentConfig(
        system_instructions="test",
        capabilities=types.CapabilitiesConfig(
            agent_behavior=types.AgentBehavior.INTERACTIVE,
            enable_subagents=True,
        ),
    )

    with mock.patch("builtins.print"):
      await interactive.run_interactive_loop(
          config, agent_class=mock_agent_class
      )

    mock_agent_class.assert_called_once()
    called_config = mock_agent_class.call_args[0][0]
    self.assertEqual(
        called_config.capabilities.agent_behavior,
        types.AgentBehavior.INTERACTIVE,
    )
    self.assertTrue(called_config.capabilities.enable_subagents)


class FormatStepSpinnerMessageTest(unittest.TestCase):
  """Tests for _format_step_spinner_message (mock-free)."""

  def test_single_tool_call(self):
    step = types.Step(
        type=types.StepType.TOOL_CALL,
        tool_calls=[types.ToolCall(name="tool_a", args={})],
    )
    self.assertEqual(
        interactive._format_step_spinner_message(step),
        "Running tool 'tool_a'...",
    )

  def test_multiple_tool_calls(self):
    step = types.Step(
        type=types.StepType.TOOL_CALL,
        tool_calls=[
            types.ToolCall(name="tool_a", args={}),
            types.ToolCall(name="tool_b", args={}),
        ],
    )
    self.assertEqual(
        interactive._format_step_spinner_message(step),
        "Running tools 'tool_a', 'tool_b'...",
    )

  def test_empty_tool_calls(self):
    step = types.Step(type=types.StepType.TOOL_CALL, tool_calls=[])
    self.assertEqual(
        interactive._format_step_spinner_message(step),
        "Running tool...",
    )

  def test_compaction_and_reasoning(self):
    compaction_step = types.Step(type=types.StepType.COMPACTION)
    self.assertEqual(
        interactive._format_step_spinner_message(compaction_step),
        "Compacting context...",
    )
    reasoning_step = types.Step(
        source=types.StepSource.MODEL,
        thinking_delta="thinking...",
    )
    self.assertEqual(
        interactive._format_step_spinner_message(reasoning_step),
        "Reasoning...",
    )

  def test_other_step_returns_none(self):
    step = types.Step(content="Hello")
    self.assertIsNone(interactive._format_step_spinner_message(step))


class SpinnerInteractionTest(unittest.IsolatedAsyncioTestCase):
  """Tests for Spinner pausing and resuming during interactions."""

  async def test_spinner_pause_resume(self):
    """Verifies that Spinner.pause() and resume() correctly control the task."""
    spinner = interactive.Spinner("Testing")
    spinner._enabled = True  # Force enabled for test.

    async with spinner:
      self.assertTrue(spinner.is_running)
      self.assertIsNotNone(spinner._task)
      self.assertIs(interactive.Spinner.get_active(), spinner)

      spinner.pause()
      self.assertFalse(spinner.is_running)
      self.assertIsNone(spinner._task)

      spinner.resume()
      self.assertTrue(spinner.is_running)
      self.assertIsNotNone(spinner._task)

    self.assertIsNone(interactive.Spinner.get_active())

  @mock.patch("builtins.input")
  async def test_async_input_pauses_active_spinner(self, mock_input):
    """Verifies that async_input automatically pauses an active spinner."""
    mock_input.return_value = "hello"
    spinner = interactive.Spinner("Testing")
    spinner._enabled = True

    async with spinner:
      self.assertTrue(spinner.is_running)
      with (
          mock.patch.object(
              spinner, "pause", wraps=spinner.pause
          ) as mock_pause,
          mock.patch.object(
              spinner, "resume", wraps=spinner.resume
          ) as mock_resume,
      ):
        res = await interactive.async_input("prompt> ")
        self.assertEqual(res, "hello")
        mock_pause.assert_called_once()
        mock_resume.assert_called_once()
        self.assertTrue(spinner.is_running)

  @mock.patch("builtins.input")
  async def test_ask_user_handler_pauses_active_spinner(self, mock_input):
    """Verifies that ask_user_handler pauses and resumes the active spinner."""
    mock_input.return_value = "y"
    spinner = interactive.Spinner("Testing")
    spinner._enabled = True

    async with spinner:
      self.assertTrue(spinner.is_running)
      with (
          mock.patch.object(
              interactive.Spinner,
              "pause_active",
              wraps=interactive.Spinner.pause_active,
          ) as mock_pause,
          mock.patch.object(
              interactive.Spinner,
              "resume_active",
              wraps=interactive.Spinner.resume_active,
          ) as mock_resume,
      ):
        tc = types.ToolCall(name="test_tool", args={"foo": "bar"})
        res = await interactive.ask_user_handler(tc)
        self.assertTrue(res)
        mock_pause.assert_called()
        mock_resume.assert_called()
        self.assertTrue(spinner.is_running)

  @mock.patch("builtins.print")
  @mock.patch("builtins.input")
  async def test_ask_user_handler_sanitizes_terminal_output(
      self, mock_input, mock_print
  ):
    """Verifies ANSI escape sequences and control chars are stripped."""
    mock_input.return_value = "y"
    tc = types.ToolCall(
        name="run_\x1b[31mcommand\x07",
        args={"cmd": "echo\r\x1b[2A\x1b[2K spoofed"},
    )
    reason = "Assessor flagged:\x1b[32m risky\x08\x00"
    res = await interactive.ask_user_handler(tc, reason=reason)
    self.assertTrue(res)
    printed = "\n".join(str(call.args[0]) for call in mock_print.call_args_list)
    self.assertNotIn("\x1b", printed)
    self.assertNotIn("\r", printed)
    self.assertNotIn("\x08", printed)
    self.assertIn("run_command", printed)
    self.assertIn("Assessor flagged: risky", printed)

  @mock.patch("builtins.print")
  @mock.patch("builtins.input")
  async def test_tool_confirmation_hook_sanitizes_terminal_output(
      self, mock_input, mock_print
  ):
    """Verifies ToolConfirmationHook.run strips ANSI escape sequences and control chars."""
    mock_input.return_value = "y"
    hook = interactive.ToolConfirmationHook()
    tc = types.ToolCall(
        name="tool_\x1b[31mname\x07",
        args={"arg": "val\r\x1b[2A\x1b[2K spoofed"},
    )
    ctx = mock.MagicMock()
    res = await hook.run(ctx, tc)
    self.assertTrue(res.allow)
    printed = "\n".join(str(call.args[0]) for call in mock_print.call_args_list)
    self.assertNotIn("\x1b", printed)
    self.assertNotIn("\r", printed)
    self.assertIn("tool_name", printed)

  def test_sanitize_terminal_text_preserves_newlines_and_tabs(self):
    r"""Verifies _sanitize_terminal_text strips control chars while keeping \n and \t."""
    raw = "line1\n\tindented\r\x07\x1b[31mred\x1b[0m"
    sanitized = interactive._sanitize_terminal_text(raw)  # pylint: disable=protected-access
    self.assertEqual(sanitized, "line1\n\tindentedred")


if __name__ == "__main__":
  unittest.main()
