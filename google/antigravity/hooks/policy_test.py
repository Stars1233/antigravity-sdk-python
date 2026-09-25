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

"""Tests for the tool call policy system.

Covers:
- Builder functions (allow, deny, ask_user, allow_all, deny_all)
- Startup validation (missing ASK_USER handler)
- Priority-based evaluation order across all 6 levels
- Short-circuit behavior (first match wins within a group)
- Sync and async predicates, including exception fail-closed
- ASK_USER handler invocation (approve, deny, async, exception)
- Default behavior when no policies match
- Edge cases (empty policy list, policy name in deny reason)
"""

from collections.abc import Mapping
import inspect
from typing import Any
import unittest
from unittest import mock

from absl.testing import absltest
import pydantic

from google.antigravity.proto import localharness_pb2
from google.antigravity import types
from google.antigravity.hooks import hooks
from google.antigravity.hooks import policy


class RunCommandArgs(pydantic.BaseModel):
  """Arguments for run_command tool."""

  command_line: str


def _make_tool_call(
    name: str = "run_command", server_name: str | None = None, **args: Any
) -> types.ToolCall:
  # Simulate Connection layer path normalization for tests
  canonical_path = None
  for path_key in ("path", "file_path", "TargetFile", "directory_path"):
    if path_key in args and isinstance(args[path_key], str):
      canonical_path = args[path_key]
      break
  return types.ToolCall(
      name=name,
      args=args,
      canonical_path=canonical_path,
      server_name=server_name,
  )


class BuilderTest(unittest.TestCase):
  """Verifies that builder functions construct Policy objects correctly."""

  def test_allow_creates_approve_policy(self):
    """allow() must produce a Policy with decision=APPROVE."""
    p = policy.allow("read_file", name="allow-read")
    self.assertEqual(p.tool, "read_file")
    self.assertEqual(p.decision, policy.Decision.APPROVE)
    self.assertIsNone(p.when)
    self.assertIsNone(p.ask_user)
    self.assertEqual(p.name, "allow-read")

  def test_deny_creates_deny_policy(self):
    """deny() must produce a Policy with decision=DENY."""
    p = policy.deny("run_command", name="block-cmd")
    self.assertEqual(p.tool, "run_command")
    self.assertEqual(p.decision, policy.Decision.DENY)
    self.assertEqual(p.name, "block-cmd")

  def test_ask_user_creates_ask_user_policy(self):
    """ask_user() must produce a Policy with decision=ASK_USER and handler."""
    def handler(_):
      return True

    p = policy.ask_user("run_command", handler=handler, name="confirm-cmd")
    self.assertEqual(p.decision, policy.Decision.ASK_USER)
    self.assertIs(p.ask_user, handler)

  def test_ask_user_without_handler_creates_ask_user_policy(self):
    """ask_user() without handler produces a Policy with ask_user=None."""
    p = policy.ask_user("run_command", name="confirm-cmd")
    self.assertEqual(p.tool, "run_command")
    self.assertEqual(p.decision, policy.Decision.ASK_USER)
    self.assertIsNone(p.ask_user)
    self.assertEqual(p.name, "confirm-cmd")

  def test_ask_user_mcp_server_without_handler(self):
    """ask_user(BaseMcpServerConfig, handler=None) returns list[Policy] with ask_user=None."""
    mcp = types.McpStdioServer(name="my_server", command="cmd")
    policies = policy.ask_user(mcp)
    self.assertIsInstance(policies, list)
    self.assertGreater(len(policies), 0)
    for p in policies:
      self.assertEqual(p.decision, policy.Decision.ASK_USER)
      self.assertIsNone(p.ask_user)

  def test_deny_with_predicate(self):
    """deny() with a when clause stores the predicate."""
    def pred(args):
      return "rm" in args.get("CommandLine", "")

    p = policy.deny("run_command", when=pred)
    self.assertIs(p.when, pred)

  def test_allow_all_creates_wildcard_approve(self):
    """allow_all() must produce a wildcard APPROVE policy."""
    p = policy.allow_all()
    self.assertEqual(p.tool, "*")
    self.assertEqual(p.decision, policy.Decision.APPROVE)
    self.assertEqual(p.name, "allow_all")

  def test_deny_all_creates_wildcard_deny(self):
    """deny_all() must produce a wildcard DENY policy."""
    p = policy.deny_all()
    self.assertEqual(p.tool, "*")
    self.assertEqual(p.decision, policy.Decision.DENY)
    self.assertEqual(p.name, "deny_all")


class ValidationTest(unittest.TestCase):
  """Verifies startup validation in enforce()."""

  def test_enforce_rejects_ask_user_without_handler(self):
    """enforce() must raise ValueError when ASK_USER has no handler."""
    bad_policy = policy.Policy(
        tool="run_command", decision=policy.Decision.ASK_USER, name="oops"
    )
    with self.assertRaises(ValueError) as ctx:
      policy.enforce([bad_policy])
    self.assertIn("oops", str(ctx.exception))
    self.assertIn("missing an ask_user handler", str(ctx.exception))

  def test_enforce_rejects_ask_user_without_handler_unnamed(self):
    """enforce() error message includes tool name when policy has no name."""
    bad_policy = policy.Policy(
        tool="my_tool", decision=policy.Decision.ASK_USER
    )
    with self.assertRaises(ValueError) as ctx:
      policy.enforce([bad_policy])
    self.assertIn("my_tool", str(ctx.exception))

  def test_enforce_rejects_mcp_policies_when_mcp_servers_is_none(self):
    """enforce() raises ValueError when MCP policies are used without mcp_servers."""
    mcp_policy = policy.allow("github/create_issue")
    with self.assertRaisesRegex(
        ValueError, "MCP policies .* were detected, but 'mcp_servers' was not"
    ):
      policy.enforce([mcp_policy])

  def test_enforce_allows_mcp_policies_when_mcp_servers_is_empty_list(self):
    """enforce() allows MCP policies when mcp_servers is passed as empty list (Hub mode)."""
    mcp_policy = policy.allow("github/create_issue")
    hook = policy.enforce([mcp_policy], mcp_servers=[])
    self.assertIsInstance(hook, policy._PolicyDecideHook)


class PriorityEvaluationTest(unittest.IsolatedAsyncioTestCase):
  """Verifies precedence ordering in enforce() (specific < prefix < wildcard, DENY < ASK < ALLOW)."""

  async def test_specific_deny_overrides_wildcard_allow(self):
    """Specific deny beats wildcard allow."""
    hook = policy.enforce([
        policy.allow("*"),
        policy.deny("dangerous_tool"),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("dangerous_tool"))
    self.assertFalse(result.allow)

  async def test_specific_deny_overrides_specific_allow(self):
    """Specific deny beats specific allow."""
    hook = policy.enforce([
        policy.allow("run_command"),
        policy.deny("run_command"),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertFalse(result.allow)

  async def test_specific_ask_overrides_wildcard_deny(self):
    """Specific ask beats wildcard deny."""
    hook = policy.enforce([
        policy.deny("*"),
        policy.ask_user("run_command", handler=lambda tc: True),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertTrue(result.allow)

  async def test_specific_allow_overrides_wildcard_deny(self):
    """Specific allow beats wildcard deny."""
    hook = policy.enforce([
        policy.deny("*"),
        policy.allow("read_file"),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("read_file"))
    self.assertTrue(result.allow)
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertFalse(result.allow)

  async def test_specific_mcp_allow_beats_prefix_mcp_deny(self):
    """Specific MCP allow (math/calc) beats prefix MCP deny (math/*)."""
    mcp = types.McpStdioServer(name="math", command="npx")
    hook = policy.enforce(
        [
            policy.allow(mcp, ["calc"]),
            policy.deny(mcp),
        ],
        mcp_servers=[mcp],
    )
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("calc", server_name="math"))
    self.assertTrue(result.allow)
    result = await hook.run(
        ctx, _make_tool_call("multiply", server_name="math")
    )
    self.assertFalse(result.allow)

  async def test_bare_tool_policy_does_not_match_mcp_tool_call(self):
    """Local tool policy for 'calc' must not match MCP tool call 'math/calc'."""
    mcp = types.McpStdioServer(name="math", command="npx")
    hook = policy.enforce(
        [
            policy.allow("calc"),  # Local tool only
            policy.deny(mcp),  # math/*
        ],
        mcp_servers=[mcp],
    )
    ctx = hooks.HookContext()
    # math/calc must be denied by deny(math/*), not allowed by allow("calc")
    result = await hook.run(ctx, _make_tool_call("calc", server_name="math"))
    self.assertFalse(result.allow)

  async def test_workspace_only_skipped_in_process(self):
    """workspace_only() policies are skipped by _PolicyDecideHook."""
    policies = policy.workspace_only(["/tmp/workspace"])
    hook = policy.enforce(policies)
    ctx = hooks.HookContext()
    # Should be allowed by default (boundary checks handled by platform)
    result = await hook.run(ctx, _make_tool_call("view_file", path="/any/path"))
    self.assertTrue(result.allow)


class ShortCircuitTest(unittest.IsolatedAsyncioTestCase):
  """Verifies first-match-wins and predicate short-circuiting."""

  async def test_first_match_wins_within_same_priority(self):
    """When two policies have identical priority, the first registered is evaluated."""
    call_count = 0

    def counting_predicate(unused_args: Mapping[str, Any]) -> bool:
      nonlocal call_count
      call_count += 1
      return True

    hook = policy.enforce([
        policy.deny("run_command", when=counting_predicate, name="first"),
        policy.deny("run_command", when=counting_predicate, name="second"),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertFalse(result.allow)
    self.assertEqual(call_count, 1)

  async def test_skips_non_matching_predicate(self):
    """A policy whose predicate returns False is skipped; next matching policy wins."""
    hook = policy.enforce([
        policy.deny("run_command", when=lambda args: False, name="skip-me"),
        policy.deny("run_command", when=lambda args: True, name="catch-me"),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertFalse(result.allow)
    self.assertIn("catch-me", result.message)


class PredicateTest(unittest.IsolatedAsyncioTestCase):
  """Verifies sync, async, and failing predicates."""

  async def test_sync_predicate_true(self):
    """Sync predicate returning True causes the policy to match."""
    hook = policy.enforce([
        policy.deny(
            "run_command",
            when=lambda args: args.get("CommandLine", "").startswith("rm"),
        ),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(
        ctx, _make_tool_call("run_command", CommandLine="rm -rf /")
    )
    self.assertFalse(result.allow)

  async def test_sync_predicate_false(self):
    """Sync predicate returning False skips the policy."""
    hook = policy.enforce([
        policy.deny(
            "run_command",
            when=lambda args: args.get("CommandLine", "").startswith("rm"),
        ),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(
        ctx, _make_tool_call("run_command", CommandLine="echo hi")
    )
    self.assertTrue(result.allow)

  async def test_async_predicate_true(self):
    """Async predicate returning True causes the policy to match."""

    async def is_dangerous(args: Mapping[str, Any]) -> bool:
      return "rm" in args.get("CommandLine", "")

    hook = policy.enforce([
        policy.deny("run_command", when=is_dangerous),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(
        ctx, _make_tool_call("run_command", CommandLine="rm -rf")
    )
    self.assertFalse(result.allow)

  async def test_async_predicate_false(self):
    """Async predicate returning False skips the policy."""

    async def is_dangerous(args: Mapping[str, Any]) -> bool:
      return "rm" in args.get("CommandLine", "")

    hook = policy.enforce([
        policy.deny("run_command", when=is_dangerous),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(
        ctx, _make_tool_call("run_command", CommandLine="echo")
    )
    self.assertTrue(result.allow)

  async def test_predicate_exception_matches_fail_closed(self):
    """Exception in predicate → policy matches (fail-closed).

    This is the critical safety property: a deny policy with a broken
    predicate still denies, preventing accidental allow-through.
    """

    def exploding_predicate(_: Mapping[str, Any]) -> bool:
      raise RuntimeError("boom")

    hook = policy.enforce([
        policy.deny("run_command", when=exploding_predicate, name="broken"),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertFalse(result.allow)
    self.assertIn("broken", result.message)
    self.assertIn("boom", result.message)

  async def test_parameterless_predicate(self):
    """Predicate with no arguments should be called without arguments."""

    def no_args_predicate():
      return True

    hook = policy.enforce([
        policy.deny("run_command", when=no_args_predicate, name="no-args"),
    ])
    ctx = hooks.HookContext()

    # This calls no_args_predicate() with 0 arguments.
    # It succeeds and returns True (match), leading to denial.
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertFalse(result.allow)
    self.assertEqual(result.message, "Denied by policy 'no-args'.")

  async def test_parameterless_predicate_allow(self):
    """Parameterless predicate in ALLOW policy works and allows/denies correctly."""
    is_allowed = False

    def my_predicate():
      return is_allowed

    hook = policy.enforce([
        policy.allow("run_command", when=my_predicate, name="paramless-allow"),
        policy.deny("*"),
    ])
    ctx = hooks.HookContext()

    # When predicate returns False -> should deny (via deny("*"))
    is_allowed = False
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertFalse(result.allow)

    # When predicate returns True -> should allow
    is_allowed = True
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertTrue(result.allow)

  async def test_typed_predicate(self):

    """Predicate expecting a Pydantic model receives the parsed object."""

    def my_typed_predicate(args: RunCommandArgs) -> bool:
      return "rm" in args.command_line

    hook = policy.enforce([
        policy.deny("run_command", when=my_typed_predicate),
    ])
    ctx = hooks.HookContext()

    # Matches
    result = await hook.run(
        ctx, _make_tool_call("run_command", command_line="rm -rf")
    )
    self.assertFalse(result.allow)

    # Doesn't match
    result = await hook.run(
        ctx, _make_tool_call("run_command", command_line="echo hi")
    )
    self.assertTrue(result.allow)

  async def test_allow_predicate_exception_denies(self):
    """Exception in allow policy predicate must deny (fail-closed)."""

    def exploding_predicate(_: Mapping[str, Any]) -> bool:
      raise RuntimeError("boom")

    hook = policy.enforce([
        policy.allow(
            "run_command", when=exploding_predicate, name="broken-allow"
        ),
        policy.allow("*"),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertFalse(result.allow)
    self.assertIn("broken-allow", result.message)
    self.assertIn("boom", result.message)


class AskUserTest(unittest.IsolatedAsyncioTestCase):
  """Verifies ASK_USER handler invocation."""

  async def test_handler_approve(self):
    """Handler returning True → tool is allowed."""
    hook = policy.enforce([
        policy.ask_user("run_command", handler=lambda tc: True),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertTrue(result.allow)

  async def test_handler_deny(self):
    """Handler returning False → tool is denied."""
    hook = policy.enforce([
        policy.ask_user("run_command", handler=lambda tc: False),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertFalse(result.allow)
    self.assertIn("User denied", result.message)

  async def test_handler_async(self):
    """Async handler is awaited correctly."""

    async def async_handler(tc: types.ToolCall) -> bool:
      return tc.args.get("safe", False)

    hook = policy.enforce([
        policy.ask_user("run_command", handler=async_handler),
    ])
    ctx = hooks.HookContext()

    result = await hook.run(ctx, _make_tool_call("run_command", safe=True))
    self.assertTrue(result.allow)

    result = await hook.run(ctx, _make_tool_call("run_command", safe=False))
    self.assertFalse(result.allow)

  async def test_handler_exception_denies(self):
    """Handler exception is caught and denies the tool call."""

    def broken_handler(_: types.ToolCall) -> bool:
      raise RuntimeError("handler broke")

    hook = policy.enforce([
        policy.ask_user(
            "run_command", handler=broken_handler, name="broken-ask"
        ),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertFalse(result.allow)
    self.assertIn("broken-ask", result.message)
    self.assertIn("handler broke", result.message)

  async def test_handler_receives_tool_call(self):
    """Handler receives the full ToolCall object, not just args."""
    received = []

    def capturing_handler(tc: types.ToolCall) -> bool:
      received.append(tc)
      return True

    hook = policy.enforce([
        policy.ask_user("run_command", handler=capturing_handler),
    ])
    ctx = hooks.HookContext()
    tc = _make_tool_call("run_command", CommandLine="echo hi")
    await hook.run(ctx, tc)
    self.assertEqual(len(received), 1)
    self.assertIs(received[0], tc)

  async def test_handler_1_arg(self):
    """1-arg handler receives ToolCall."""
    received = []

    def handler(tc: types.ToolCall) -> bool:
      received.append(tc)
      return True

    p = policy.ask_user("run_command", handler=handler)
    tc = _make_tool_call("run_command")
    result = await policy._execute_ask_user(p, tc)
    self.assertTrue(result)
    self.assertEqual(len(received), 1)

  async def test_handler_2_arg_with_reason(self):
    """2-arg handler receives ToolCall and reason."""
    received = []

    def handler(tc: types.ToolCall, reason: str) -> bool:
      received.append((tc, reason))
      return True

    p = policy.ask_user("run_command", handler=handler)
    tc = _make_tool_call("run_command")
    result = await policy._execute_ask_user(p, tc, reason="test reason")
    self.assertTrue(result)
    self.assertEqual(received[0][1], "test reason")

  async def test_handler_kwargs(self):
    """**kwargs handler receives reason."""
    received = []

    def handler(tc: types.ToolCall, **kwargs) -> bool:
      del tc
      received.append(kwargs.get("reason"))
      return True

    p = policy.ask_user("run_command", handler=handler)
    tc = _make_tool_call("run_command")
    result = await policy._execute_ask_user(p, tc, reason="test reason")
    self.assertTrue(result)
    self.assertEqual(received[0], "test reason")

  async def test_internal_type_error_not_swallowed(self):
    """Internal TypeError inside handler is not swallowed by fallback."""
    def handler(tc: types.ToolCall) -> bool:
      del tc
      # This raises TypeError internally, which could have been incorrectly
      # caught by the old dispatch fallback logic trying to inject reason.
      return len(10) > 0  # TypeError: object of type 'int' has no len()

    p = policy.ask_user("run_command", handler=handler)
    tc = _make_tool_call("run_command")

    # Evaluate using _PolicyDecideHook to simulate real flow and confirm it
    # fails closed due to the exception. The hook runner uses try/except
    # Exception on evaluate_predicate and _execute_ask_user, returning
    # Allow=False on exception.
    hook = policy.enforce([p])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, tc)
    self.assertFalse(result.allow)
    self.assertIn("object of type 'int' has no len", result.message)


class DefaultBehaviorTest(unittest.IsolatedAsyncioTestCase):
  """Verifies behavior when no policies match."""

  async def test_no_matching_policy_allows(self):
    """When no policy matches, the tool call is allowed (open system)."""
    hook = policy.enforce([
        policy.deny("other_tool"),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("unrelated_tool"))
    self.assertTrue(result.allow)

  async def test_empty_policies_allows_all(self):
    """An empty policy list allows everything."""
    hook = policy.enforce([])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("any_tool"))
    self.assertTrue(result.allow)


class ConvenienceBuilderTest(unittest.IsolatedAsyncioTestCase):
  """Verifies allow_all() and deny_all() evaluate correctly through enforce."""

  async def test_allow_all_approves_any_tool(self):
    """allow_all() approves arbitrary tool calls."""
    hook = policy.enforce([policy.allow_all()])
    ctx = hooks.HookContext()
    for tool in ("run_command", "view_file", "create_file", "unknown_tool"):
      result = await hook.run(ctx, _make_tool_call(tool))
      self.assertTrue(result.allow, f"{tool} should be allowed")

  async def test_deny_all_denies_any_tool(self):
    """deny_all() denies arbitrary tool calls."""
    hook = policy.enforce([policy.deny_all()])
    ctx = hooks.HookContext()
    for tool in ("run_command", "view_file", "create_file"):
      result = await hook.run(ctx, _make_tool_call(tool))
      self.assertFalse(result.allow, f"{tool} should be denied")

  async def test_deny_all_with_specific_allow_override(self):
    """deny_all() + allow(tool) creates deny-by-default with exceptions."""
    hook = policy.enforce([
        policy.deny_all(),
        policy.allow("view_file"),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("view_file"))
    self.assertTrue(result.allow)
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertFalse(result.allow)


class DenyReasonTest(unittest.IsolatedAsyncioTestCase):
  """Verifies that deny reasons include useful context."""

  async def test_named_policy_in_deny_reason(self):
    """Policy name appears in the deny reason message."""
    hook = policy.enforce([
        policy.deny("run_command", name="no-commands"),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertIn("no-commands", result.message)

  async def test_unnamed_policy_uses_tool_name(self):
    """When a policy has no name, the tool name is used in the reason."""
    hook = policy.enforce([
        policy.deny("run_command"),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertIn("run_command", result.message)

  async def test_custom_reason_in_deny_policy(self):
    """When a policy has a custom reason, it is used as the deny message."""
    hook = policy.enforce([
        policy.deny(
            "run_command",
            name="block-cmd",
            reason="Custom security policy: shell execution forbidden.",
        ),
    ])
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertFalse(result.allow)
    self.assertEqual(
        result.message, "Custom security policy: shell execution forbidden."
    )


class IntegrationWithHookRunnerTest(unittest.IsolatedAsyncioTestCase):
  """Verifies the policy hook integrates with HookRunner dispatch."""

  async def test_policy_hook_in_hook_runner(self):
    """Policy hook works when dispatched through HookRunner.

    This confirms the hook is a proper PreToolCallDecideHook subclass
    that the HookRunner can dispatch.
    """
    from google.antigravity.hooks import hook_runner  # pylint: disable=g-import-not-at-top

    hook = policy.enforce([
        policy.deny("*"),
        policy.allow("read_file"),
    ])

    runner = hook_runner.HookRunner(pre_tool_call_decide_hooks=[hook])
    turn_context = hooks.TurnContext(runner.session_context)

    # read_file should be allowed
    result, _, _ = await runner.dispatch_pre_tool_call(
        turn_context, _make_tool_call("read_file")
    )
    self.assertTrue(result.allow)

    # run_command should be denied
    result, _, _ = await runner.dispatch_pre_tool_call(
        turn_context, _make_tool_call("run_command")
    )
    self.assertFalse(result.allow)


class SafeDefaultsTest(unittest.IsolatedAsyncioTestCase):
  """Verifies safe_defaults() preset."""

  async def test_safe_defaults_allows_read_only_tools(self):
    """safe_defaults() must allow read-only tools."""

    def handler(_):
      return False

    policies = policy.safe_defaults(handler=handler)
    hook = policy.enforce(policies)
    ctx = hooks.HookContext()

    for tool in (
        "list_directory",
        "search_directory",
        "find_file",
        "view_file",
        "finish",
    ):
      result = await hook.run(ctx, _make_tool_call(tool))
      self.assertTrue(result.allow, f"{tool} should be allowed")

  async def test_safe_defaults_asks_for_other_tools(self):
    """safe_defaults() must ask for non-read-only tools."""
    handler_called = False

    def handler(_):
      nonlocal handler_called
      handler_called = True
      return True

    policies = policy.safe_defaults(handler=handler)
    hook = policy.enforce(policies)
    ctx = hooks.HookContext()

    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertTrue(result.allow)
    self.assertTrue(handler_called)


class ConfirmCommandsTest(unittest.IsolatedAsyncioTestCase):
  """Verifies the confirm_run_command() preset — the default for LocalAgentConfig.

  confirm_run_command() is the safe-by-default policy: it denies run_command
  while allowing all other tools.  When a handler is provided, run_command
  is upgraded to ASK_USER instead of DENY.
  """

  async def test_denies_run_command_by_default(self):
    """Without a handler, run_command is denied with a clear message."""
    policies = policy.confirm_run_command()
    hook = policy.enforce(policies)
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertFalse(result.allow)
    self.assertIn("confirm_run_command", result.message)

  async def test_allows_other_tools_by_default(self):
    """Without a handler, all non-run_command tools are allowed."""
    policies = policy.confirm_run_command()
    hook = policy.enforce(policies)
    ctx = hooks.HookContext()
    for tool in types.BuiltinTools:
      if tool == types.BuiltinTools.RUN_COMMAND:
        continue
      result = await hook.run(ctx, _make_tool_call(tool.value))
      self.assertTrue(result.allow, f"{tool.value} should be allowed")

  async def test_with_handler_asks_user_for_run_command(self):
    """With a handler, run_command triggers ASK_USER instead of DENY."""
    handler_calls = []

    def handler(tc: types.ToolCall) -> bool:
      handler_calls.append(tc)
      return True

    policies = policy.confirm_run_command(handler=handler)
    hook = policy.enforce(policies)
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertTrue(result.allow)
    self.assertEqual(len(handler_calls), 1)

  async def test_with_handler_deny_propagates(self):
    """Handler returning False denies the tool call."""
    policies = policy.confirm_run_command(handler=lambda tc: False)
    hook = policy.enforce(policies)
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("run_command"))
    self.assertFalse(result.allow)
    self.assertIn("User denied", result.message)

  async def test_with_handler_allows_other_tools(self):
    """With a handler, non-run_command tools are still auto-allowed."""
    policies = policy.confirm_run_command(handler=lambda tc: False)
    hook = policy.enforce(policies)
    ctx = hooks.HookContext()
    result = await hook.run(ctx, _make_tool_call("view_file"))
    self.assertTrue(result.allow)

  def test_returns_list_of_policies(self):
    """confirm_run_command() always returns a list of Policy objects."""
    for policies in (
        policy.confirm_run_command(),
        policy.confirm_run_command(handler=lambda tc: True),
    ):
      self.assertIsInstance(policies, list)
      self.assertGreaterEqual(len(policies), 2)
      for p in policies:
        self.assertIsInstance(p, policy.Policy)


class WorkspaceOnlyTest(unittest.TestCase):
  """Verifies workspace_only() builder."""

  def test_workspace_only_returns_deny_policies_for_file_tools(self):
    """workspace_only() returns deny policies for all file tools."""
    policies = policy.workspace_only(["/tmp/workspace"])
    self.assertIsInstance(policies, list)
    file_tools = [t.value for t in types.BuiltinTools.file_tools()]
    self.assertEqual(len(policies), len(file_tools))
    for p in policies:
      self.assertEqual(p.decision, policy.Decision.DENY)
      self.assertEqual(p.name, "workspace_only")
      self.assertIn(p.tool, file_tools)


class McpPolicyTest(unittest.IsolatedAsyncioTestCase):
  """Unit tests for overloaded MCP policy methods and structured target alignment."""

  def setUp(self):
    super().setUp()
    self.mcp_config = types.McpStdioServer(name="math", command="npx")
    self.mcp_config_adv = types.McpStdioServer(
        name="math_advanced", command="npx"
    )

  def test_allow_mcp_builder_wildcard(self):
    """allow(mcp_config) must produce a single wildcard policy in structured target format."""
    policies = policy.allow(self.mcp_config)
    self.assertIsInstance(policies, list)
    (p,) = policies  # Implicitly asserts length is exactly 1
    self.assertEqual(p.tool, "math/*")
    self.assertEqual(p.decision, policy.Decision.APPROVE)

  def test_allow_mcp_builder_specific_tools(self):
    """allow(mcp_config, tools) must produce policies in 'server/tool' format."""
    policies = policy.allow(self.mcp_config, ["calc", "multiply"])
    self.assertIsInstance(policies, list)
    p1, p2 = policies  # Implicitly asserts length is exactly 2
    self.assertEqual(p1.tool, "math/calc")
    self.assertEqual(p2.tool, "math/multiply")
    self.assertEqual(p1.name, "approve_math_calc")

  def test_deny_mcp_builder_specific_tools(self):
    """deny(mcp_config, tools) must produce policies in 'server/tool' format."""
    policies = policy.deny(self.mcp_config, ["calc"])
    self.assertIsInstance(policies, list)
    (p,) = policies  # Implicitly asserts length is exactly 1
    self.assertEqual(p.tool, "math/calc")
    self.assertEqual(p.decision, policy.Decision.DENY)

  def test_ask_user_mcp_builder_specific_tools(self):
    """ask_user(mcp_config, tools) must produce policies in 'server/tool' format with handler."""

    def dummy_handler(tc):
      return True

    policies = policy.ask_user(self.mcp_config, ["calc"], handler=dummy_handler)
    self.assertIsInstance(policies, list)
    (p,) = policies  # Implicitly asserts length is exactly 1
    self.assertEqual(p.tool, "math/calc")
    self.assertEqual(p.decision, policy.Decision.ASK_USER)
    self.assertIs(p.ask_user, dummy_handler)

  def test_builder_custom_name_unique(self):
    """Builders must append tool suffix to custom name for unique logging."""
    policies = policy.allow(self.mcp_config, ["calc"], name="custom")
    (p,) = policies  # Implicitly asserts length is exactly 1
    self.assertEqual(p.name, "custom_calc")

  def test_builder_rejects_string_with_mcp_tools(self):
    """Builders must raise ValueError if mcp_tools is provided for a string tool name."""
    with self.assertRaises(ValueError) as ctx:
      policy.allow("read_file", ["tool1"])
    self.assertIn("mcp_tools cannot be specified", str(ctx.exception))

  def test_mcp_tools_string_type_guard(self):
    """Builders must raise ValueError if mcp_tools is passed as a raw string."""
    with self.assertRaises(ValueError) as ctx:
      policy.allow(self.mcp_config, "calc")
    self.assertIn("mcp_tools must be a sequence of strings", str(ctx.exception))

  def test_enforce_flattens_nested_policies(self):
    """enforce() must successfully flatten mixed nested lists of policies."""
    policies = [
        policy.allow("read_file"),
        policy.allow(self.mcp_config),  # Returns list[Policy]
    ]
    hook = policy.enforce(policies, mcp_servers=[self.mcp_config])
    self.assertIsInstance(hook, hooks.PreToolCallDecideHook)

  async def test_secure_longest_match_matching(self):
    """math prefix must not eagerly match math_advanced tools."""
    # Register both servers (math_advanced has longer name)
    policies = [
        policy.allow(self.mcp_config),  # math/*
        policy.deny(self.mcp_config_adv),  # math_advanced/*
    ]
    hook = policy.enforce(
        policies, mcp_servers=[self.mcp_config, self.mcp_config_adv]
    )
    ctx = hooks.HookContext()

    result = await hook.run(
        ctx, _make_tool_call("calc", server_name="math_advanced")
    )
    self.assertFalse(result.allow)
    self.assertIn("math_advanced_all", result.message)

    result = await hook.run(ctx, _make_tool_call("calc", server_name="math"))
    self.assertTrue(result.allow)

  def test_enforce_rejects_non_policy_in_sequence(self):
    """enforce() must raise ValueError if a non-Policy is found in a nested sequence."""
    bad_policies = [
        policy.allow("read_file"),
        ["not_a_policy"],  # Invalid element in nested list
    ]
    with self.assertRaises(ValueError) as ctx:
      policy.enforce(bad_policies, mcp_servers=[self.mcp_config])
    self.assertIn("Expected Policy, got <class 'str'>", str(ctx.exception))

  def test_enforce_rejects_invalid_policy_type(self):
    """enforce() must raise ValueError if a direct invalid type is passed."""
    bad_policies = [
        policy.allow("read_file"),
        123,  # Invalid direct type
    ]
    with self.assertRaises(ValueError) as ctx:
      policy.enforce(bad_policies, mcp_servers=[self.mcp_config])
    self.assertIn(
        "Expected Policy or Sequence of Policies, got <class 'int'>",
        str(ctx.exception),
    )

  async def test_matches_target_unknown_server_prefix(self):
    """Prefixed calls with unknown server name must be treated as standard tools."""
    policies = [
        policy.allow("math/*"),  # math/*
    ]
    # 'unknown' is NOT in mcp_servers
    hook = policy.enforce(policies, mcp_servers=[self.mcp_config])
    ctx = hooks.HookContext()

    # Since 'unknown' is unregistered, 'calc' with server_name='unknown' is treated as a standard tool.
    # It should NOT match the 'math/*' prefix wildcard.
    result = await hook.run(ctx, _make_tool_call("calc", server_name="unknown"))
    self.assertTrue(result.allow)  # Default open since no policy matches


class ToPolicyConfigProtoTest(absltest.TestCase):
  """Tests for to_policy_config_proto serialization."""

  def test_empty_policies(self):
    """Empty policy list produces config with no rules."""
    config, dynamic_policy_map = policy._to_policy_config_proto([])
    self.assertEmpty(config.rules)
    self.assertEmpty(dynamic_policy_map)

  def test_static_deny(self):
    """Static deny produces a non-dynamic rule."""
    config, dynamic_policy_map = policy._to_policy_config_proto(
        [policy.deny("run_command", name="block_cmd")]
    )
    self.assertLen(config.rules, 1)
    rule = config.rules[0]
    self.assertEqual(rule.tool, "run_command")
    self.assertEqual(rule.server_name, "")
    self.assertEqual(rule.decision, localharness_pb2.POLICY_DECISION_DENY)
    self.assertEqual(rule.name, "block_cmd")
    self.assertFalse(rule.is_dynamic)
    self.assertEqual(rule.rule_id, "")
    self.assertEmpty(dynamic_policy_map)

  def test_static_deny_with_reason(self):
    """Static deny with custom reason serializes deny_reason."""
    p = policy.deny(
        "run_command", name="block_cmd", reason="Shell access disabled."
    )
    self.assertEqual(p.reason, "Shell access disabled.")
    config, _ = policy._to_policy_config_proto([p])
    self.assertLen(config.rules, 1)
    self.assertEqual(config.rules[0].deny_reason, "Shell access disabled.")

  def test_unnamed_policy_defaults_to_tool(self):
    """Unnamed policy gets name defaulted to tool target."""
    config, _ = policy._to_policy_config_proto([policy.deny("run_command")])
    self.assertLen(config.rules, 1)
    self.assertEqual(config.rules[0].name, "run_command")

  def test_static_allow_wildcard(self):
    """Static allow('*') produces global wildcard."""
    config, _ = policy._to_policy_config_proto([policy.allow("*")])
    self.assertLen(config.rules, 1)
    rule = config.rules[0]
    self.assertEqual(rule.tool, "*")
    self.assertEqual(rule.server_name, "")
    self.assertEqual(rule.decision, localharness_pb2.POLICY_DECISION_ALLOW)
    self.assertFalse(rule.is_dynamic)

  def test_dynamic_when_predicate(self):
    """Rule with `when` predicate is dynamic."""
    pred = lambda args: "rm" in args.get("CommandLine", "")
    config, dynamic_policy_map = policy._to_policy_config_proto(
        [policy.deny("run_command", when=pred, name="block_rm")]
    )
    self.assertLen(config.rules, 1)
    rule = config.rules[0]
    self.assertTrue(rule.is_dynamic)
    self.assertEqual(rule.rule_id, "rule_0")
    self.assertIn("rule_0", dynamic_policy_map)
    self.assertEqual(dynamic_policy_map["rule_0"].name, "block_rm")
    self.assertIs(dynamic_policy_map["rule_0"].when, pred)

  def test_dynamic_ask_user(self):
    """ASK_USER rules are always dynamic."""
    handler = lambda tc: True
    config, dynamic_policy_map = policy._to_policy_config_proto(
        [policy.ask_user("run_command", handler=handler, name="confirm")]
    )
    self.assertLen(config.rules, 1)
    rule = config.rules[0]
    self.assertTrue(rule.is_dynamic)
    self.assertEqual(rule.decision, localharness_pb2.POLICY_DECISION_ASK_USER)
    self.assertIn(rule.rule_id, dynamic_policy_map)

  def test_mcp_specific_tool(self):
    """MCP 'server/tool' is decomposed into separate fields."""
    mcp = types.McpStdioServer(name="my_server", command="cmd")
    policies = policy.deny(mcp, mcp_tools=["dangerous_tool"])
    config, _ = policy._to_policy_config_proto(policies)
    self.assertLen(config.rules, 1)
    rule = config.rules[0]
    self.assertEqual(rule.tool, "dangerous_tool")
    self.assertEqual(rule.server_name, "my_server")

  def test_mcp_prefix_wildcard(self):
    """MCP 'server/*' decomposes to tool='*', server_name='server'."""
    mcp = types.McpStdioServer(name="my_server", command="cmd")
    policies = policy.deny(mcp)  # No mcp_tools = server-wide wildcard
    config, _ = policy._to_policy_config_proto(policies)
    self.assertLen(config.rules, 1)
    rule = config.rules[0]
    self.assertEqual(rule.tool, "*")
    self.assertEqual(rule.server_name, "my_server")

  def test_confirm_run_command_no_handler(self):
    """confirm_run_command() without handler produces 2 static rules."""
    config, dynamic_policy_map = policy._to_policy_config_proto(
        policy.confirm_run_command()
    )
    self.assertLen(config.rules, 2)
    # First rule: deny run_command
    self.assertEqual(config.rules[0].tool, "run_command")
    self.assertEqual(
        config.rules[0].decision, localharness_pb2.POLICY_DECISION_DENY
    )
    self.assertFalse(config.rules[0].is_dynamic)
    # Second rule: allow *
    self.assertEqual(config.rules[1].tool, "*")
    self.assertEqual(
        config.rules[1].decision, localharness_pb2.POLICY_DECISION_ALLOW
    )
    self.assertFalse(config.rules[1].is_dynamic)
    self.assertEmpty(dynamic_policy_map)

  def test_confirm_run_command_with_handler(self):
    """confirm_run_command(handler) produces 1 dynamic + 1 static."""
    handler = lambda tc: True
    config, dynamic_policy_map = policy._to_policy_config_proto(
        policy.confirm_run_command(handler=handler)
    )
    self.assertLen(config.rules, 2)
    # First: ask_user (dynamic)
    self.assertTrue(config.rules[0].is_dynamic)
    self.assertEqual(
        config.rules[0].decision, localharness_pb2.POLICY_DECISION_ASK_USER
    )
    # Second: allow * (static)
    self.assertFalse(config.rules[1].is_dynamic)
    self.assertLen(dynamic_policy_map, 1)

  def test_rule_ordering_preserved(self):
    """Proto rules preserve the declaration order of input policies."""
    config, _ = policy._to_policy_config_proto([
        policy.deny("a"),
        policy.allow("b"),
        policy.deny("c"),
    ])
    self.assertLen(config.rules, 3)
    self.assertEqual(config.rules[0].tool, "a")
    self.assertEqual(config.rules[1].tool, "b")
    self.assertEqual(config.rules[2].tool, "c")

  def test_nested_policies_flattened(self):
    """Nested policy lists (from MCP builders) are correctly flattened."""
    mcp = types.McpStdioServer(name="srv", command="cmd")
    config, _ = policy._to_policy_config_proto([
        policy.deny(mcp, mcp_tools=["t1", "t2"]),
        policy.allow("read_file"),
    ])
    self.assertLen(config.rules, 3)
    self.assertEqual(config.rules[0].tool, "t1")
    self.assertEqual(config.rules[1].tool, "t2")
    self.assertEqual(config.rules[2].tool, "read_file")

  def test_dynamic_policy_map_only_dynamic(self):
    """Only dynamic rules appear in the dynamic_policy_map."""
    config, dynamic_policy_map = policy._to_policy_config_proto([
        policy.deny("a"),  # static
        policy.deny("b", when=lambda args: True),  # dynamic
        policy.allow("c"),  # static
        policy.ask_user("d", handler=lambda tc: True),  # dynamic
    ])
    self.assertLen(config.rules, 4)
    self.assertLen(dynamic_policy_map, 2)
    self.assertIn("rule_1", dynamic_policy_map)  # index 1 = deny("b", when=...)
    self.assertIn("rule_3", dynamic_policy_map)  # index 3 = ask_user("d", ...)
    self.assertNotIn("rule_0", dynamic_policy_map)
    self.assertNotIn("rule_2", dynamic_policy_map)

  def test_allow_all_disables_workspace_containment(self):
    """allow_all() without workspace_only() disables built-in containment."""
    config, _ = policy._to_policy_config_proto([policy.allow_all()])
    self.assertEqual(
        config.workspace_containment,
        localharness_pb2.PolicyConfig.WORKSPACE_CONTAINMENT_DISABLED,
    )

  def test_allow_all_with_workspace_only_keeps_workspace_containment(self):
    """allow_all() combined with workspace_only() keeps built-in containment."""
    config, _ = policy._to_policy_config_proto(
        [policy.allow_all(), policy.workspace_only(["/tmp/ws"])]
    )
    self.assertEqual(
        config.workspace_containment,
        localharness_pb2.PolicyConfig.WORKSPACE_CONTAINMENT_UNSPECIFIED,
    )

  def test_confirm_run_command_keeps_workspace_containment(self):
    """confirm_run_command() keeps built-in workspace containment."""
    config, _ = policy._to_policy_config_proto(policy.confirm_run_command())
    self.assertEqual(
        config.workspace_containment,
        localharness_pb2.PolicyConfig.WORKSPACE_CONTAINMENT_UNSPECIFIED,
    )


class ExecuteAskUserTest(unittest.IsolatedAsyncioTestCase):
  """Tests _execute_ask_user handler dispatch and reason handling."""

  async def test_execute_ask_user_with_keyword_reason(self):
    received_reason = []

    def handler_with_reason(tc, reason=""):
      del tc
      received_reason.append(reason)
      return True

    p = policy.ask_user("run_command", handler=handler_with_reason)
    res = await policy._execute_ask_user(
        p, _make_tool_call("run_command"), reason="Command is dangerous"
    )
    self.assertTrue(res)
    self.assertEqual(received_reason, ["Command is dangerous"])

  async def test_execute_ask_user_positional_two_args(self):
    received = []

    def handler_positional(tc, r):
      del tc
      received.append(r)
      return True

    p = policy.ask_user("run_command", handler=handler_positional)
    res = await policy._execute_ask_user(
        p, _make_tool_call("run_command"), reason="Positional reason"
    )
    self.assertTrue(res)
    self.assertEqual(received, ["Positional reason"])

  async def test_execute_ask_user_var_kwargs(self):
    received = {}

    def handler_kwargs(tc, **kwargs):
      del tc
      received.update(kwargs)
      return True

    p = policy.ask_user("run_command", handler=handler_kwargs)
    res = await policy._execute_ask_user(
        p, _make_tool_call("run_command"), reason="Kwargs reason"
    )
    self.assertTrue(res)
    self.assertEqual(received.get("reason"), "Kwargs reason")

  async def test_execute_ask_user_single_arg(self):
    called = []

    def handler_single_arg(tc):
      called.append(tc.name)
      return True

    p = policy.ask_user("run_command", handler=handler_single_arg)
    res = await policy._execute_ask_user(
        p, _make_tool_call("run_command"), reason="Command is dangerous"
    )
    self.assertTrue(res)
    self.assertEqual(called, ["run_command"])

  async def test_execute_ask_user_two_arg_auto_handler_receives_reason(self):
    received_reason = []

    def handler_two_args(tc, reason=None):
      del tc
      received_reason.append(reason)
      return True

    p = policy.auto(handler=handler_two_args)
    tc = _make_tool_call("run_command")
    res = await policy._execute_ask_user(p, tc, reason="Command is dangerous")
    self.assertTrue(res)
    self.assertEqual(received_reason, ["Command is dangerous"])

  async def test_execute_ask_user_internal_type_error_propagates(self):
    call_count = 0

    def buggy_handler(tc, reason=""):
      nonlocal call_count
      call_count += 1
      del tc, reason
      raise TypeError("custom internal type error")

    p = policy.ask_user("run_command", handler=buggy_handler)
    with self.assertRaisesRegex(TypeError, "custom internal type error"):
      await policy._execute_ask_user(
          p, _make_tool_call("run_command"), reason="test"
      )
    self.assertEqual(call_count, 1)

  async def test_execute_ask_user_no_signature_calls_without_reason(self):
    called = []

    class CustomCallable:

      def __call__(self, tc):
        called.append(tc.name)
        return True

    custom = CustomCallable()
    # Mock inspect.signature to raise TypeError for this callable
    with mock.patch.object(
        inspect, "signature", side_effect=TypeError("No signature")
    ):
      p = policy.ask_user("run_command", handler=custom)
      res = await policy._execute_ask_user(
          p, _make_tool_call("run_command"), reason="Command is dangerous"
      )
      self.assertTrue(res)
      self.assertEqual(called, ["run_command"])

  async def test_execute_ask_user_positional_only_reason(self):
    received = []

    def handler_pos_only(tc, reason, /):
      del tc
      received.append(reason)
      return True

    p = policy.ask_user("run_command", handler=handler_pos_only)
    res = await policy._execute_ask_user(
        p, _make_tool_call("run_command"), reason="Positional only reason"
    )
    self.assertTrue(res)
    self.assertEqual(received, ["Positional only reason"])

  async def test_execute_ask_user_unrelated_keyword_only(self):
    called = []

    def handler_kw_only(tc, *, some_other_kw=True):
      called.append((tc.name, some_other_kw))
      return True

    p = policy.ask_user("run_command", handler=handler_kw_only)
    res = await policy._execute_ask_user(
        p, _make_tool_call("run_command"), reason="Ignored reason"
    )
    self.assertTrue(res)
    self.assertEqual(called, [("run_command", True)])

  async def test_ask_user_with_custom_reason(self):
    received_reasons = []

    def handler_with_reason(tc, reason=""):
      del tc
      received_reasons.append(reason)
      return False

    p = policy.ask_user(
        "run_command",
        handler=handler_with_reason,
        reason="Requires approval before modifying production state",
    )
    self.assertEqual(
        p.reason, "Requires approval before modifying production state"
    )

    proto_config, _ = policy._to_policy_config_proto([p])
    self.assertEqual(
        proto_config.rules[0].deny_reason,
        "Requires approval before modifying production state",
    )

    hook = policy.enforce([p])
    res = await hook.run(hooks.HookContext(), _make_tool_call("run_command"))
    self.assertFalse(res.allow)
    self.assertEqual(
        received_reasons,
        ["Requires approval before modifying production state"],
    )
    self.assertEqual(
        res.message, "Requires approval before modifying production state"
    )


class AutoPolicyTest(unittest.IsolatedAsyncioTestCase):
  """Verifies policy.auto() builder and proto conversion."""

  def test_auto_defaults(self):
    p = policy.auto()
    self.assertIsInstance(p, policy.AutoPolicy)
    self.assertIsInstance(p, policy.Policy)
    self.assertTrue(p.auto)
    self.assertEqual(p.tool, "*")
    self.assertEqual(p.decision, policy.Decision.DENY)
    self.assertIsNone(p.ask_user)

  def test_auto_with_handler(self):
    handler = lambda tc: True
    p = policy.auto(handler=handler)
    self.assertIsInstance(p, policy.AutoPolicy)
    self.assertTrue(p.auto)
    self.assertEqual(p.decision, policy.Decision.ASK_USER)
    self.assertIs(p.ask_user, handler)

  def test_auto_policy_direct_instantiation(self):
    handler = lambda tc: True
    p = policy.AutoPolicy(name="my-auto", ask_user=handler, model="flash")
    self.assertIsInstance(p, policy.Policy)
    self.assertTrue(p.auto)
    self.assertEqual(p.name, "my-auto")
    self.assertEqual(p.model, "flash")
    self.assertEqual(p.decision, policy.Decision.ASK_USER)
    self.assertIs(p.ask_user, handler)

  def test_auto_with_model(self):
    p = policy.auto(model="gemini-3.5-flash-lite")
    self.assertTrue(p.auto)
    self.assertEqual(p.model, "gemini-3.5-flash-lite")

    config, _ = policy._to_policy_config_proto([p])
    self.assertTrue(config.auto_config.enabled)
    self.assertEqual(config.auto_config.model, "gemini-3.5-flash-lite")

  def test_auto_to_proto(self):
    config, dynamic_policy_map = policy._to_policy_config_proto([
        policy.auto(),
    ])
    self.assertTrue(config.auto_config.enabled)
    self.assertIn("auto", dynamic_policy_map)
    self.assertNotIn("cortex_hook", dynamic_policy_map)

  def test_user_policy_named_auto_does_not_collide(self):
    user_rule = policy.allow(
        "run_command", name="auto", when=lambda args: True
    )
    self.assertFalse(user_rule.auto)
    config, dynamic_policy_map = policy._to_policy_config_proto([
        user_rule,
        policy.auto(),
    ])
    self.assertTrue(config.auto_config.enabled)
    self.assertEqual(len(config.rules), 1)
    self.assertEqual(config.rules[0].name, "auto")
    self.assertEqual(config.rules[0].rule_id, "rule_0")
    self.assertIs(dynamic_policy_map["rule_0"], user_rule)
    self.assertIsInstance(dynamic_policy_map["auto"], policy.AutoPolicy)

  def test_multiple_auto_policies_raises_error(self):
    with self.assertRaisesRegex(
        ValueError, "Multiple AutoPolicy rules found"
    ):
      policy._to_policy_config_proto([
          policy.auto(),
          policy.auto(model="gemini-2.5-flash"),
      ])

  async def test_auto_skipped_in_in_process_enforce(self):
    from google.antigravity.hooks import hook_runner  # pylint: disable=g-import-not-at-top

    hook = policy.enforce([policy.auto()])
    runner = hook_runner.HookRunner(pre_tool_call_decide_hooks=[hook])
    turn_context = hooks.TurnContext(runner.session_context)
    result, _, _ = await runner.dispatch_pre_tool_call(
        turn_context, _make_tool_call("run_command")
    )
    self.assertTrue(result.allow)
if __name__ == "__main__":
  absltest.main()
