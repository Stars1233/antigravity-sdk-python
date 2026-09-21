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

r"""Example demonstrating auto policy mode safety evaluation.

This example shows how to configure and run an agent in auto policy mode using
policy.auto(). Commands and tool calls are evaluated for safety before
execution. When a tool call is flagged, the interactive CLI handler prompts the
user for confirmation with the safety reasoning.

To run:
  python auto_policy_cli.py

Tip: Pass --alsologtostderr to see execution steps and safety assessment in
detail.
"""

import asyncio
from collections.abc import Sequence
import logging as std_logging
import sys

from absl import app
from absl import flags
from absl import logging

from google.antigravity import Agent
from google.antigravity import LocalAgentConfig
from google.antigravity import types
from google.antigravity.hooks import policy
from google.antigravity.utils import interactive  # pyrefly: ignore[missing-module-attribute]
from google.antigravity.utils.interactive import async_input  # pyrefly: ignore[missing-import]

_MODEL_NAME = flags.DEFINE_string(
    "model_name",
    None,
    "Optional Gemini model name override for the agent. Defaults to the SDK"
    " default model.",
)
_AUTO_MODEL = flags.DEFINE_string(
    "auto_model",
    None,
    "Optional Gemini model name override for auto policy mode safety"
    " evaluation. Defaults to the runtime's default safety assessment model.",
)
_PROMPT = flags.DEFINE_multi_string(
    "prompt",
    None,
    "Optional prompt(s) to send non-interactively before exiting.",
)
_AUTO_CONFIRM = flags.DEFINE_enum(
    "auto_confirm",
    None,
    ["approve", "reject"],
    "Optional non-interactive response to safety confirmation prompts.",
)


async def _non_interactive_handler(
    tc: types.ToolCall, reason: str | None = None
) -> bool:
  """Handler that prints confirmation details and responds non-interactively."""
  print(f"\nPolicy check: Tool execution requested: {tc.name}")
  if reason:
    print(f"Reason: {reason}")
  if tc.args:
    print(f"Arguments: {tc.args}")
  approved = _AUTO_CONFIRM.value == "approve"
  decision_str = "y" if approved else "n"
  print(f"Allow execution? (y/n) [n]: {decision_str} (auto_confirm)")
  return approved


async def run() -> None:
  """Runs the interactive conversation loop with auto policy mode enforcement."""
  handler = (
      _non_interactive_handler
      if _AUTO_CONFIRM.value is not None
      else interactive.ask_user_handler
  )
  auto_policy = policy.auto(
      handler=handler,
      model=_AUTO_MODEL.value,
  )

  config = LocalAgentConfig(
      policies=[auto_policy],
      hooks=[interactive.AskQuestionHook()],
      capabilities=types.CapabilitiesConfig(
          agent_behavior=types.AgentBehavior.INTERACTIVE,
      ),
      model=_MODEL_NAME.value,
  )

  async with Agent(config) as agent:
    print("\nGoogle Antigravity SDK - Auto Policy Mode Demo")
    print("Automated safety evaluation enabled.")

    if _PROMPT.value:
      for prompt_text in _PROMPT.value:
        print(f"\n→ {prompt_text}")
        response = await agent.chat(prompt_text)
        async for chunk in response:
          sys.stdout.write(chunk)
          sys.stdout.flush()
        print()
      return

    print("Type your message and press Enter • Ctrl+C to exit\n")

    while True:
      try:
        user_input = await async_input("\n→ ")
        user_input = user_input.strip()
        if not user_input:
          continue
        if user_input.lower() in ("exit", "quit"):
          print("\nGoodbye! 👋")
          break

        response = await agent.chat(user_input)

        # Stream the response to stdout
        async for chunk in response:
          sys.stdout.write(chunk)
          sys.stdout.flush()
        print()

      except (KeyboardInterrupt, asyncio.CancelledError, EOFError):
        print("\nGoodbye! 👋")
        break


def main(argv: Sequence[str]) -> None:
  """Entry point for the auto policy mode CLI example."""
  del argv
  logging.set_verbosity(logging.INFO)
  std_logging.basicConfig(level=std_logging.INFO, format="%(message)s")
  asyncio.run(run())


if __name__ == "__main__":
  app.run(main)
