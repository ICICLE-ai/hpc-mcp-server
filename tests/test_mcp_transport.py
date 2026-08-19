import json
import unittest

from fastapi.testclient import TestClient

from hpc_mcp_server_cpu.main import MCP_PUBLIC_HOST, app


MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


def _response_message(response) -> dict:
    if response.headers.get("content-type", "").startswith("application/json"):
        return response.json()
    for line in response.text.splitlines():
        if line.startswith("data: "):
            return json.loads(line.removeprefix("data: "))
    raise AssertionError(f"No MCP message in response: {response.text}")


class McpTransportTest(unittest.TestCase):
    def test_public_endpoint_lists_registered_tools(self) -> None:
        with TestClient(app, base_url=f"https://{MCP_PUBLIC_HOST}") as client:
            initialize = client.post(
                "/mcp/",
                headers=MCP_HEADERS,
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-06-18",
                        "capabilities": {},
                        "clientInfo": {"name": "transport-test", "version": "1.0"},
                    },
                },
            )
            self.assertEqual(initialize.status_code, 200, initialize.text)
            self.assertEqual(_response_message(initialize)["id"], 1)

            session_id = initialize.headers["mcp-session-id"]
            session_headers = {**MCP_HEADERS, "Mcp-Session-Id": session_id}

            initialized = client.post(
                "/mcp/",
                headers=session_headers,
                json={"jsonrpc": "2.0", "method": "notifications/initialized"},
            )
            self.assertEqual(initialized.status_code, 202, initialized.text)

            tool_list = client.post(
                "/mcp/",
                headers=session_headers,
                json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            )
            self.assertEqual(tool_list.status_code, 200, tool_list.text)
            tools = _response_message(tool_list)["result"]["tools"]
            self.assertEqual({tool["name"] for tool in tools}, {"chat", "predict_gpu_time"})
            self.assertTrue(all(tool.get("description") for tool in tools))

            untrusted = client.post(
                "https://untrusted.example/mcp/",
                headers=MCP_HEADERS,
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            )
            self.assertEqual(untrusted.status_code, 421)


if __name__ == "__main__":
    unittest.main()
