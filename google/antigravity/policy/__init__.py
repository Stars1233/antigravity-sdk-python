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

"""Tool call policy system for the Google Antigravity SDK."""

from google.antigravity.hooks.policy import allow
from google.antigravity.hooks.policy import allow_all
from google.antigravity.hooks.policy import ask_user
from google.antigravity.hooks.policy import auto
from google.antigravity.hooks.policy import AutoPolicy
from google.antigravity.hooks.policy import confirm_run_command
from google.antigravity.hooks.policy import Decision
from google.antigravity.hooks.policy import deny
from google.antigravity.hooks.policy import deny_all
from google.antigravity.hooks.policy import Policy
from google.antigravity.hooks.policy import safe_defaults
from google.antigravity.hooks.policy import workspace_only

__all__ = [
    "allow",
    "allow_all",
    "ask_user",
    "auto",
    "AutoPolicy",
    "confirm_run_command",
    "Decision",
    "deny",
    "deny_all",
    "Policy",
    "safe_defaults",
    "workspace_only",
]
