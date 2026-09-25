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

import asyncio
import json
import unittest
from unittest import mock

from google.protobuf import json_format

from google.antigravity.proto import steps_pb2
from google.antigravity.proto import localharness_pb2
from google.antigravity.connections.local import local_connection
from google.antigravity.connections.local import test_utils
from google.antigravity.tools import tool_runner


class TestWebSocketTest(unittest.IsolatedAsyncioTestCase):

  async def test_send_adds_to_sent_messages(self):
    ws = test_utils.TestWebSocket()
    await ws.send("hello")
    self.assertEqual(ws.sent_messages, ["hello"])

  async def test_send_puts_to_sent_queue(self):
    ws = test_utils.TestWebSocket()
    await ws.send("hello")
    msg = await ws.sent_queue.get()
    self.assertEqual(msg, "hello")

  async def test_put_event_enqueues_json(self):
    ws = test_utils.TestWebSocket()
    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(text="test")
    )
    await ws.put_event(event)
    await ws.close()

    items = []
    async for msg in ws:
      items.append(msg)

    self.assertEqual(len(items), 1)
    data = json.loads(items[0])
    self.assertIn("stepUpdate", data)
    self.assertEqual(data["stepUpdate"]["text"], "test")

  async def test_close_ends_iteration(self):
    ws = test_utils.TestWebSocket()
    await ws.close()

    items = []
    async for item in ws:
      items.append(item)

    self.assertEqual(items, [])


class TestLocalHarnessTest(unittest.IsolatedAsyncioTestCase):

  async def asyncSetUp(self):
    await super().asyncSetUp()
    self.ws = test_utils.TestWebSocket()
    self.mock_process = mock.MagicMock()
    self.tool_runner = tool_runner.ToolRunner()
    self.harness = test_utils.TestLocalHarness(
        test_case=self,
        ws=self.ws,
        process=self.mock_process,
        tool_runner=self.tool_runner,
    )

  async def test_harness_creates_default_ws(self):
    harness = test_utils.TestLocalHarness(
        test_case=self,
        process=self.mock_process,
    )
    self.assertIsNotNone(harness.ws)
    self.assertIsInstance(harness.ws, test_utils.TestWebSocket)

  async def test_send_event_puts_to_socket(self):
    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(text="test")
    )
    await self.harness.send_event(event)
    await self.ws.close()

    items = []
    async for msg in self.ws:
      items.append(msg)

    self.assertEqual(len(items), 1)
    data = json.loads(items[0])
    self.assertEqual(data["stepUpdate"]["text"], "test")

  async def test_wait_for_response_succeeds(self):
    await self.ws.send(json.dumps({"status": "ok"}))
    data = await self.harness.wait_for_response()
    self.assertEqual(data["status"], "ok")

  async def test_wait_for_response_times_out(self):
    with self.assertRaises(asyncio.TimeoutError):
      await self.harness.wait_for_response(timeout=0.1)

  async def test_send_tool_call(self):
    await self.harness.send_tool_call(
        id="1", name="test_tool", arguments_json="{}"
    )
    await self.ws.close()

    items = []
    async for msg in self.ws:
      items.append(msg)

    self.assertEqual(len(items), 1)
    data = json.loads(items[0])
    self.assertIn("toolCall", data)
    self.assertEqual(data["toolCall"]["id"], "1")
    self.assertEqual(data["toolCall"]["name"], "test_tool")

  async def test_send_tool_confirmation_request(self):
    await self.harness.send_tool_confirmation_request(
        trajectory_id="test_traj",
        step_index=5,
        view_file=localharness_pb2.ActionViewFile(file_path="/foo"),
    )
    await self.ws.close()

    items = []
    async for msg in self.ws:
      items.append(msg)

    self.assertEqual(len(items), 1)
    data = json.loads(items[0])
    self.assertIn("stepUpdate", data)
    step_update = data["stepUpdate"]
    self.assertEqual(step_update["trajectoryId"], "test_traj")
    self.assertEqual(step_update["stepIndex"], 5)
    self.assertIn("toolConfirmationRequest", step_update)
    self.assertIn("viewFile", step_update)
    self.assertEqual(step_update["viewFile"]["filePath"], "/foo")

  async def test_send_skill_lookup_action_wire_and_binary(self):
    lookup_action = localharness_pb2.ActionSkillLookup(
        operation=localharness_pb2.ActionSkillLookup.OPERATION_LOOKUP_SKILLS,
        requested_skill_names=["alpha", "beta"],
        resolved_skill_names=["alpha"],
        error_message="",
    )
    event = localharness_pb2.OutputEvent(
        step_update=localharness_pb2.StepUpdate(
            trajectory_id="test_traj",
            step_index=1,
            state=localharness_pb2.StepUpdate.STATE_DONE,
            skill_lookup=lookup_action,
        )
    )
    await self.harness.send_event(event)
    await self.ws.close()

    items = []
    async for msg in self.ws:
      items.append(msg)

    self.assertEqual(len(items), 1)
    data = json.loads(items[0])
    self.assertIn("stepUpdate", data)
    step_update = data["stepUpdate"]
    self.assertIn("skillLookup", step_update)
    self.assertEqual(
        step_update["skillLookup"]["operation"], "OPERATION_LOOKUP_SKILLS"
    )
    self.assertEqual(
        step_update["skillLookup"]["requestedSkillNames"], ["alpha", "beta"]
    )
    self.assertEqual(
        step_update["skillLookup"]["resolvedSkillNames"], ["alpha"]
    )

    # Verify JSON deserialization via standard json_format into OutputEvent.
    parsed_event = json_format.Parse(items[0], localharness_pb2.OutputEvent())
    self.assertEqual(
        parsed_event.step_update.skill_lookup.operation,
        localharness_pb2.ActionSkillLookup.OPERATION_LOOKUP_SKILLS,
    )
    self.assertEqual(
        list(parsed_event.step_update.skill_lookup.requested_skill_names),
        ["alpha", "beta"],
    )
    self.assertEqual(
        list(parsed_event.step_update.skill_lookup.resolved_skill_names),
        ["alpha"],
    )

    # Verify binary protobuf serialization/deserialization.
    binary_bytes = event.SerializeToString()
    deserialized = localharness_pb2.OutputEvent.FromString(binary_bytes)
    self.assertEqual(
        deserialized.step_update.skill_lookup.operation,
        localharness_pb2.ActionSkillLookup.OPERATION_LOOKUP_SKILLS,
    )
    self.assertEqual(
        list(deserialized.step_update.skill_lookup.resolved_skill_names),
        ["alpha"],
    )

  def test_interactions_steps_skill_lookup(self):
    call_step = steps_pb2.Step(
        tool_call=steps_pb2.ToolCallStep(
            id="call_1",
            skill_lookup_call=steps_pb2.SkillLookupCallStep(
                operation=steps_pb2.SkillLookupCallStep.OPERATION_LOOKUP_SKILLS,
                requested_skill_names=["alpha"],
            ),
        )
    )
    result_step = steps_pb2.Step(
        tool_result=steps_pb2.ToolResultStep(
            call_id="call_1",
            skill_lookup_result=steps_pb2.SkillLookupResultStep(
                resolved_skill_names=["alpha"],
            ),
        )
    )

    # Wire JSON format verification.
    call_json = json_format.MessageToJson(call_step)
    self.assertIn("skillLookupCall", call_json)
    parsed_call = json_format.Parse(call_json, steps_pb2.Step())
    self.assertEqual(
        parsed_call.tool_call.skill_lookup_call.operation,
        steps_pb2.SkillLookupCallStep.OPERATION_LOOKUP_SKILLS,
    )

    result_json = json_format.MessageToJson(result_step)
    self.assertIn("skillLookupResult", result_json)
    parsed_result = json_format.Parse(result_json, steps_pb2.Step())
    self.assertEqual(
        list(
            parsed_result.tool_result.skill_lookup_result.resolved_skill_names
        ),
        ["alpha"],
    )

    # Binary roundtrip verification.
    call_bytes = call_step.SerializeToString()
    deser_call = steps_pb2.Step.FromString(call_bytes)
    self.assertEqual(
        deser_call.tool_call.skill_lookup_call.operation,
        steps_pb2.SkillLookupCallStep.OPERATION_LOOKUP_SKILLS,
    )


class PatchDefaultBinaryPathTest(unittest.TestCase):

  def test_patches_binary_path_during_test(self):
    test_utils.patch_default_binary_path(
        self, return_value="/custom/test/binary"
    )
    self.assertEqual(
        local_connection._get_default_binary_path(env=None),
        "/custom/test/binary",
    )


if __name__ == "__main__":
  unittest.main()
