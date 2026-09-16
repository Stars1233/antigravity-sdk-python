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

r"""Local models getting started example for Google Antigravity SDK.

This example demonstrates how to run agents completely on-device without cloud
connectivity or API keys using local models. Any compatible local model will
work, but we recommend and use Gemma 4 26B here for simplicity and strong
reasoning capabilities:
1. LiteRT (Managed on-device): Managed local execution using LiteRT-LM,
   illustrated with Gemma 4 26B (or any .litertlm model file).
2. Local OpenAI-compatible server: Connecting to an external local server
   such as Ollama or LM Studio running Gemma or other open-weights models.
3. Lightweight optimization: LiteRTAgentConfig auto-applies lightweight
   presets upon instantiation, while LocalOpenAIAgentConfig applies them via
   `.lightweight()` to optimize prompt overhead, restrict tools to essential
   actions, disable subagents, and tune compaction thresholds for local hardware.
4. Streaming tokens in real time from the local model.

To run:
  # Any compatible local model can be used. For simplicity, we recommend
  # Gemma 4 26B.

  # Option 1: LiteRT with Gemma 4 26B (On-device runtime)
  # Download Gemma 4 26B checkpoint using LiteRT-LM CLI:
  litert-lm import \
    --from-huggingface-repo=litert-community/gemma-4-26B-A4B-it-litert-lm \
    gemma-4-26B-A4B-it-web.litertlm \
    gemma4-26b

  # Run the agent:
  python local_models.py

  # Or specify any custom .litertlm model path:
  python local_models.py --provider litert \
    --model_path ~/.litert-lm/models/gemma4-26b/model.litertlm

  # Option 2: OpenAI-compatible local server (e.g. Ollama)
  # Any model registered with your server will work (e.g. gemma-4-26B-A4B-it,
  # gemma2:27b, llama3.1, etc.):
  #   ollama run gemma-4-26B-A4B-it
  python local_models.py --provider openai \
    --model gemma-4-26B-A4B-it \
    --base_url http://localhost:11434/v1

Criteria for correct script performance:
  1. The script exits cleanly with return code 0 (no unhandled exceptions).
  2. The agent initializes the local model configuration without error.
  3. The agent produces a streaming text response.
"""

import argparse
import asyncio
import os

from google.antigravity import Agent
from google.antigravity import LiteRTAgentConfig
from google.antigravity import LocalOpenAIAgentConfig

DEFAULT_GEMMA4_MODEL_PATH = os.path.expanduser(
    "~/.litert-lm/models/gemma4-26b/model.litertlm"
)
DEFAULT_OPENAI_BASE_URL = "http://localhost:11434/v1"
DEFAULT_OPENAI_MODEL = "gemma-4-26B-A4B-it"


def parse_args() -> argparse.Namespace:
  """Parses command-line arguments."""
  parser = argparse.ArgumentParser(
      description="Run an Antigravity agent using local models (Gemma 4 26B)."
  )
  parser.add_argument(
      "--provider",
      choices=["litert", "openai"],
      default="litert",
      help=(
          "Local model runtime provider. 'litert' manages on-device LiteRT-LM;"
          " 'openai' connects to an external local server like Ollama."
      ),
  )
  parser.add_argument(
      "--model_path",
      default=DEFAULT_GEMMA4_MODEL_PATH,
      help="Path to the .litertlm model file (used with --provider litert).",
  )
  parser.add_argument(
      "--model",
      default=DEFAULT_OPENAI_MODEL,
      help=(
          "Model identifier registered on the local server (for --provider"
          " openai)."
      ),
  )
  parser.add_argument(
      "--base_url",
      default=DEFAULT_OPENAI_BASE_URL,
      help="OpenAI-compatible server endpoint (for --provider openai).",
  )
  parser.add_argument(
      "--prompt",
      default="Explain how local LLM inference works in two sentences.",
      help="Initial prompt to send to the local model agent.",
  )
  parser.add_argument(
      "--dry_run",
      action="store_true",
      help="Validate configuration without starting local inference.",
  )
  return parser.parse_args()


def create_litert_config(
    args: argparse.Namespace,
) -> LiteRTAgentConfig | None:
  """Creates a LiteRTAgentConfig using on-device LiteRT-LM (Gemma 4 26B)."""
  model_path = os.path.abspath(os.path.expanduser(args.model_path))
  model_exists = os.path.exists(model_path)

  print("=" * 60)
  print("Google Antigravity SDK: Local Model Agent (LiteRT + Gemma)")
  print("=" * 60)
  print(f"  Model Path:     {model_path}")
  print("  Preset:         Lightweight (auto-applied)")
  print("=" * 60)

  if not model_exists and not args.dry_run:
    print(
        f"\nNotice: Model file not found at:\n  {model_path}\n\n"
        "To download and import Gemma 4 26B via LiteRT-LM, run:\n"
        "  litert-lm import \\\n"
        "    --from-huggingface-repo="
        "litert-community/gemma-4-26B-A4B-it-litert-lm \\\n"
        "    gemma-4-26B-A4B-it-web.litertlm \\\n"
        "    gemma4-26b\n\n"
        "This registers the checkpoint at "
        "~/.litert-lm/models/gemma4-26b/model.litertlm.\n"
        "Alternatively, pass an existing model path via --model_path, or "
        "connect\n"
        "to an external local server via:\n"
        "  python local_models.py --provider openai\n"
    )
    return None

  # Configure local LiteRT agent:
  # - LiteRTAgentConfig automatically applies the lightweight preset upon
  #   instantiation (minimal prompt overhead, essential development tools,
  #   disabled subagents, and auto-derived safe compaction thresholds).
  #
  # Note: Custom tools, hooks, triggers, or policies can also be passed here
  # to customize agent capabilities (e.g., tools=[my_tool], hooks=[...]).
  return LiteRTAgentConfig(model_path=model_path)


def create_local_openai_config(
    args: argparse.Namespace,
) -> LocalOpenAIAgentConfig:
  """Creates a LocalOpenAIAgentConfig using a local OpenAI-compatible server."""
  print("=" * 60)
  print(
      "Google Antigravity SDK: Local Model Agent (OpenAI-Compatible Server)"
  )
  print("=" * 60)
  print(f"  Server URL:     {args.base_url}")
  print(f"  Model Target:   {args.model}")
  print("  Preset:         .lightweight()")
  print("=" * 60)

  # Configure agent pointing to a local OpenAI-compatible server.
  #
  # Note: Custom tools, hooks, triggers, or policies can also be passed here
  # to customize agent capabilities (e.g., tools=[my_tool], hooks=[...]).
  return LocalOpenAIAgentConfig(
      model=args.model,
      base_url=args.base_url,
  ).lightweight()


async def main() -> None:
  args = parse_args()

  if args.provider == "litert":
    config = create_litert_config(args)
  else:
    config = create_local_openai_config(args)

  if config is None:
    return

  if args.dry_run:
    print("\n[Dry Run] Configuration built successfully:")
    print(f"  Config type:        {type(config).__name__}")
    print(f"  Enabled tools:      {config.capabilities.enabled_tools}")
    print(f"  Enable subagents:   {config.capabilities.enable_subagents}")
    print(f"  Agent behavior:     {config.capabilities.agent_behavior}")
    print(f"  Compaction config:  {config.compaction_config}")
    return

  print("\nStarting local agent session...")
  async with Agent(config) as agent:
    print(f"\n  User: {args.prompt}")
    print("  Agent: ", end="", flush=True)

    response = await agent.chat(args.prompt)
    async for token in response:
      print(token, end="", flush=True)
    print()


if __name__ == "__main__":
  asyncio.run(main())
