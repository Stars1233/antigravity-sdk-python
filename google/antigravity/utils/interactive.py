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

"""Interactive CLI utilities for agent debugging and development.

This module provides stdin-based interactive utilities for running agents
in a terminal. These are intended for local development and debugging,
not for production use.

Includes:

- ``run_interactive_loop``: A REPL that reads user input, sends it to the
  agent, and prints responses.
- ``ToolConfirmationHook``: A hook that prompts the user for confirmation
  before executing a tool call.
- ``AskQuestionHook``: A hook that prompts the user to answer questions
  asked by the agent.
- ``ask_user_handler``: A policy handler that prompts the user for
  confirmation before executing a tool call.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
import contextvars
import re
import sys
import threading
from typing import Any
from google.antigravity import agent as agent_module
from google.antigravity import types
from google.antigravity.connections import connection as connection_module
from google.antigravity.hooks import hooks
from google.antigravity.hooks import policy as policy_module
from google.antigravity.types import QuestionResponse

_global_active_spinner: Spinner | None = None
_active_spinner: contextvars.ContextVar[Spinner | None] = (
    contextvars.ContextVar("_active_spinner", default=None)
)


async def async_input(prompt: str = "") -> str:
  """Async version of `input` that handles asyncio cancellations.

  Using `asyncio.to_thread(input)` is not an option as executor runs in a
  non-daemon thread and will hang waiting for "enter" to be pressed on the
  asyncio loop teardown.

  Args:
    prompt: The prompt to display.

  Returns:
    The user input string.
  """
  spinner = Spinner.get_active()
  was_running = False
  if spinner and spinner.is_running:
    spinner.pause()
    was_running = True

  loop = asyncio.get_running_loop()
  future = loop.create_future()

  def _read_input():
    try:
      result = input(prompt)
      if not future.cancelled():
        loop.call_soon_threadsafe(future.set_result, result)
    except BaseException as e:
      if not future.cancelled():
        loop.call_soon_threadsafe(future.set_exception, e)

  thread = threading.Thread(target=_read_input, daemon=True)
  thread.start()

  try:
    return await future
  finally:
    if was_running and spinner:
      spinner.resume()


class Spinner:
  """A lightweight terminal spinner for async processing feedback."""

  def __init__(self, message: str = "Thinking..."):
    self._message = message
    self._running = False
    self._task = None
    self._frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    # TTY Check: Spinner escape sequences only write to standard out if it's a
    # real terminal. This prevents log file corruption during redirection.
    # Note: Ancient cmd environments that lack ANSI support will output escape
    # codes literally; users are expected to use modern terminal systems.
    self._enabled = sys.stdout.isatty()
    self._token: contextvars.Token[Any] | None = None

  @property
  def is_running(self) -> bool:
    """Returns whether the spinner is currently running."""
    return self._running

  def update(self, message: str) -> None:
    """Updates the spinner display message."""
    self._message = message

  def pause(self) -> None:
    """Pauses the spinner output and clears the current line."""
    if not self._enabled or not self._running:
      return
    self._running = False
    if self._task:
      self._task.cancel()
      self._task = None
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()

  def resume(self) -> None:
    """Resumes the spinner output."""
    if not self._enabled or self._running:
      return
    self._running = True
    self._task = asyncio.create_task(self._spin())

  @classmethod
  def get_active(cls) -> Spinner | None:
    """Returns the currently active Spinner instance, if any."""
    return _active_spinner.get() or _global_active_spinner

  @classmethod
  def pause_active(cls) -> None:
    """Pauses the currently active spinner, if any."""
    spinner = cls.get_active()
    if spinner:
      spinner.pause()

  @classmethod
  def resume_active(cls) -> None:
    """Resumes the currently active spinner, if any."""
    spinner = cls.get_active()
    if spinner:
      spinner.resume()

  async def _spin(self) -> None:
    idx = 0
    try:
      while self._running:
        sys.stdout.write(f"\r\033[K{self._frames[idx]} {self._message}")
        sys.stdout.flush()
        idx = (idx + 1) % len(self._frames)
        await asyncio.sleep(0.08)
    except asyncio.CancelledError:
      pass

  async def __aenter__(self) -> "Spinner":
    global _global_active_spinner
    _global_active_spinner = self
    self._token = _active_spinner.set(self)
    if self._enabled:
      self._running = True
      self._task = asyncio.create_task(self._spin())
    return self

  async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
    global _global_active_spinner
    if _global_active_spinner is self:
      _global_active_spinner = None
    if self._token is not None:
      _active_spinner.reset(self._token)
      self._token = None
    if not self._enabled:
      return
    self._running = False
    if self._task:
      self._task.cancel()
      try:
        await self._task
      except asyncio.CancelledError:
        pass
      self._task = None
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()


_ANSI_CSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]|\x1b[@-Z\\-_]")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def _sanitize_terminal_text(text: Any) -> str:
  """Strips ANSI escape sequences and control characters to prevent UI spoofing."""
  s = str(text)
  s = _ANSI_CSI_RE.sub("", s)
  s = _CONTROL_CHARS_RE.sub("", s)
  return s


class ToolConfirmationHook(hooks.PreToolCallDecideHook):
  """Hook that prompts the user for confirmation before executing a tool."""

  async def run(
      self, context: hooks.HookContext, data: types.ToolCall
  ) -> hooks.HookResult:
    """Asks the user for confirmation via standard input.

    Args:
      context: The hook context.
      data: The tool call requested by the agent.

    Returns:
      A HookResult indicating whether to allow or deny execution.
    """
    Spinner.pause_active()
    try:
      print(
          "\nTool execution requested:"
          f" {_sanitize_terminal_text(data.name)}"
      )

      if data.args:
        print(f"Arguments: {_sanitize_terminal_text(data.args)}")

      try:
        ans = await async_input("Allow execution? (y/n) [n]: ")
      except EOFError:
        ans = "n"

      if ans.strip().lower() in ("y", "yes"):
        return hooks.HookResult(allow=True)
      return hooks.HookResult(allow=False, message="User denied tool call.")
    finally:
      Spinner.resume_active()


async def ask_user_handler(tc: types.ToolCall, reason: str = "") -> bool:
  """Prompts the user for confirmation in the terminal before executing a tool.

  This is a built-in interactive CLI confirmation handler designed for use with
  ASK_USER policies and auto policy mode.

  When is this called?
    This handler is invoked automatically by the policy evaluation runtime
    whenever an agent attempts to execute a tool call governed by an ASK_USER
    policy, or when a safety assessor in auto policy mode flags an action for
    human confirmation.

  Args:
    tc: The tool call requested by the agent (contains tool name and arguments).
    reason: An explanation string supplied by the policy evaluation runtime
      (e.g., safety assessment in auto policy mode or rule denial reason)
      explaining why confirmation is requested and what security/safety risks
      were flagged.
      Handlers do not set this parameter; they receive it from the runtime to
      display to the user so the user can make an informed decision.

  Returns:
    True if the human user confirms and allows execution (entering 'y' or
    'yes'), False if denied.
  """
  Spinner.pause_active()
  try:
    print(
        "\nPolicy check: Tool execution requested:"
        f" {_sanitize_terminal_text(tc.name)}"
    )
    if reason:
      print(f"Reason: {_sanitize_terminal_text(reason)}")
    if tc.args:
      print(f"Arguments: {_sanitize_terminal_text(tc.args)}")

    try:
      ans = await async_input("Allow execution? (y/n) [n]: ")
    except EOFError:
      ans = "n"

    return ans.strip().lower() in ("y", "yes")
  finally:
    Spinner.resume_active()


class AskQuestionHook(hooks.OnInteractionHook):
  """Hook that prompts the user to answer questions asked by the agent."""

  async def run(
      self, context: hooks.HookContext, data: types.AskQuestionInteractionSpec
  ) -> hooks.QuestionHookResult:
    """Asks the user for answers to each question via standard input.

    Args:
      context: The hook context.
      data: Specification of the interaction.

    Returns:
      A QuestionHookResult containing the user's responses.
    """
    Spinner.pause_active()
    try:
      questions = data.questions
      responses = []
      try:
        for q in questions:
          print(f"\nQuestion: {q.question}")
          options = list(q.options) if hasattr(q, "options") else []
          for idx, opt in enumerate(options):
            print(f"  {idx + 1}. {opt.text}")

          ans = await async_input("Response: ")
          ans = ans.strip()
          if not ans:
            responses.append(QuestionResponse(skipped=True))
            continue

          # Try to match by option number
          matched_id = None
          if options:
            try:
              selected_idx = int(ans) - 1
              if 0 <= selected_idx < len(options):
                matched_id = options[selected_idx].id
            except ValueError:
              pass

            # Try to match by exact option text or ID
            if not matched_id:
              for opt in options:
                if (
                    ans.lower() == opt.text.lower()
                    or ans.lower() == opt.id.lower()
                ):
                  matched_id = opt.id
                  break

          if matched_id:
            responses.append(QuestionResponse(selected_option_ids=[matched_id]))
          else:
            responses.append(QuestionResponse(freeform_response=ans))

      except EOFError:
        return hooks.QuestionHookResult(responses=responses, cancelled=True)

      return hooks.QuestionHookResult(responses=responses)
    finally:
      Spinner.resume_active()


def _upgrade_policies_list(
    policies: Sequence[policy_module.Policy | Sequence[policy_module.Policy]],
) -> list[policy_module.Policy]:
  """Upgrades RUN_COMMAND deny policies in place to ASK_USER policy."""
  upgraded = []
  for p in policy_module.flatten_policies(policies):
    if (
        p.tool == types.BuiltinTools.RUN_COMMAND.value
        and p.decision == policy_module.Decision.DENY
        and p.when is None
    ):
      upgraded.append(
          policy_module.ask_user(
              types.BuiltinTools.RUN_COMMAND.value,
              handler=ask_user_handler,
              name=p.name or "interactive_confirm",
          )
      )
    else:
      upgraded.append(p)
  return upgraded


def _format_step_spinner_message(step: types.Step) -> str | None:
  """Returns the spinner display message for a given trajectory step."""
  if step.type == types.StepType.TOOL_CALL:
    if step.tool_calls:
      if len(step.tool_calls) == 1:
        return f"Running tool '{step.tool_calls[0].name}'..."
      tool_names = ", ".join(f"'{tc.name}'" for tc in step.tool_calls)
      return f"Running tools {tool_names}..."
    return "Running tool..."
  elif step.type == types.StepType.COMPACTION:
    return "Compacting context..."
  elif step.source == types.StepSource.MODEL and step.thinking_delta:
    return "Reasoning..."
  return None


async def run_interactive_loop(
    config: connection_module.AgentConfig,
    agent_class: type[agent_module.Agent] = agent_module.Agent,
) -> None:
  """Runs an interactive CLI loop for debugging and development.

  Constructs and runs the agent within an interactive session, registering an
  ``AskQuestionHook``, upgrading ``confirm_run_command()`` to ASK_USER, and
  ensuring ``agent_behavior`` is set to ``AgentBehavior.INTERACTIVE``
  so the user can answer prompts from the model.

  Type ``exit`` or ``quit`` to end the session. Ctrl+C also exits cleanly.

  Args:
    config: Declarative agent configuration.
    agent_class: The Agent class to instantiate. Defaults to the base Agent.
  """
  hooks_list = list(config.hooks)
  if not any(isinstance(hook, AskQuestionHook) for hook in hooks_list):
    hooks_list.append(AskQuestionHook())

  policies_list = _upgrade_policies_list(config.policies)

  capabilities = config.capabilities
  if capabilities.agent_behavior != types.AgentBehavior.INTERACTIVE:
    capabilities = capabilities.model_copy(
        update={"agent_behavior": types.AgentBehavior.INTERACTIVE}
    )

  upgraded_config = config.model_copy(
      update={
          "hooks": hooks_list,
          "policies": policies_list,
          "capabilities": capabilities,
      }
  )
  agent = agent_class(upgraded_config)

  async with agent:
    await _run_loop(agent)


async def _run_loop(agent: agent_module.Agent) -> None:
  """Internal helper that runs the REPL execution loop."""
  print("Starting interactive loop. Type 'exit' or 'quit' to end.")
  while True:
    try:
      user_input = await async_input("User: ")
      user_input = user_input.strip()
      if not user_input:
        continue
      if user_input.lower() in ("exit", "quit"):
        print("Goodbye!")
        break

      await agent.conversation.send(user_input)

      async with Spinner() as spinner:
        async for step in agent.conversation.receive_steps():
          msg = _format_step_spinner_message(step)
          if msg:
            spinner.update(msg)

          if step.is_complete_response:
            break
        else:
          continue

      print(f"Agent: {step.content}")

    except (KeyboardInterrupt, EOFError):
      print("\nGoodbye!")
      break
