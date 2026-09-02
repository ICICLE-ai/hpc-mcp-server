# HPC-MCP: LLM-Assisted HPC Utility Server

HPC-MCP is a service-based ICICLE software component that gives HPC users a natural-language interface to backend HPC utilities. The current release provides distributed LLM training-time prediction for Vista and Perlmutter through HTTP endpoints and MCP clients.

**Tags:** Software, AI4CI

For guidance on what to include in Tutorials, How-To Guides, Explanation, and Reference, see [Diataxis](https://diataxis.fr/).

### License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ICICLE-ai/hpc-mcp-server?tab=MIT-1-ov-file)

## References

- Deployed service: `https://hpcmcpservercpu.pods.icicleai.tapis.io`
- MCP endpoint: `https://hpcmcpservercpu.pods.icicleai.tapis.io/mcp`
- OpenAPI specification: `docs/openapi.json`
- Live OpenAPI specification: `https://hpcmcpservercpu.pods.icicleai.tapis.io/openapi.json`
- Component metadata file: `component.yaml`
- Tapis Pod example: `tapis-pod.example.json`
- Upstream estimator: `https://github.com/ICICLE-ai/distributed_training_estimator_of_LLM`

## Acknowledgements

*National Science Foundation (NSF) funded AI institute for Intelligent Cyberinfrastructure with Computational Learning in the Environment (ICICLE) (OAC 2112606)*

## Issue reporting

Report bugs, documentation issues, or release questions through GitHub Issues:

```text
https://github.com/ICICLE-ai/hpc-mcp-server/issues
```

Release and ICICLE Training Catalog contacts:

- Carlos Guzman: `guzman.109@osu.edu`
- Amit Vyas: `vyas.154@osu.edu`

---

# Tutorials

## Tutorial: Test the Deployed Training-Time Prediction Service

This tutorial shows how to check the deployed HPC-MCP service and request distributed LLM training-time predictions for Vista and Perlmutter.

### Prerequisites

- Network access to the ICICLE Tapis Pod endpoint.
- A terminal with `curl`.
- 5 minutes to test the deployed HTTP service.

### Step 1: Set the Service URL

```bash
export HPC_MCP_URL=https://hpcmcpservercpu.pods.icicleai.tapis.io
```

### Step 2: Check Service Health

```bash
curl "$HPC_MCP_URL/health"
```

Expected response shape:

```json
{
  "status": "ok",
  "service": "hpc-mcp-server-cpu",
  "version": "0.1.0"
}
```

### Step 3: Request a Vista Prediction

```bash
curl --http1.1 --max-time 180 "$HPC_MCP_URL/predict-gpu-time" \
  -H "Content-Type: application/json" \
  --data-raw '{"system":"vista"}'
```

### Step 4: Request a Perlmutter Prediction

```bash
curl --http1.1 --max-time 180 "$HPC_MCP_URL/predict-gpu-time" \
  -H "Content-Type: application/json" \
  --data-raw '{"system":"perlmutter"}'
```

Supported `system` values:

- `vista`
- `perlmutter`

Common misspellings such as `permutter` and `permultter` are also accepted.

The response includes the selected system, estimator configuration path, and estimator output.

### Step 5: Ask Through Natural Language

Users can also request predictions through the `/chat` endpoint:

```bash
curl --http1.1 --max-time 180 "$HPC_MCP_URL/chat" \
  -H "Content-Type: application/json" \
  --data-raw '{"prompt":"Predict training time on Vista. Answer with microseconds and seconds.","max_new_tokens":120}'
```

Expected response shape:

```json
{
  "response": "System: vista\nConfig: /app/vendor/distributed_training_estimator/Estimator/target_config/llemma_7b_4_2_2_V.yml\nEstimated timecost: 4764606.22498043 us (4.765 seconds)."
}
```

The `/chat` endpoint loads the configured LLM backend. On CPU-only pods, responses can be slower than direct calls to `/predict-gpu-time`.

---

# How-To Guides

## Connect an MCP Client

Claude Code or another MCP client can connect through HTTP transport:

```bash
claude mcp add hpc-mcp-server --transport http https://hpcmcpservercpu.pods.icicleai.tapis.io/mcp
```

The service exposes two MCP tools:

- `chat`
- `predict_gpu_time`

To verify MCP initialization and tool discovery with MCP Inspector:

```bash
npx --yes --package @modelcontextprotocol/inspector -- \
  mcp-inspector --cli \
  https://hpcmcpservercpu.pods.icicleai.tapis.io/mcp \
  --transport http \
  --method tools/list
```

Set `MCP_PUBLIC_HOST` when deploying the service under a different public hostname.

## Use the HTTP API

Main endpoints:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Service health check |
| `GET` | `/` | Service metadata |
| `GET` | `/about` | HTML service overview |
| `POST` | `/chat` | Natural-language chat with tool routing |
| `POST` | `/predict-gpu-time` | Direct distributed training-time prediction |
| `POST` | `/mcp` | MCP Streamable HTTP endpoint |

### `POST /predict-gpu-time`

Request:

```json
{
  "system": "vista"
}
```

Response:

```json
{
  "response": "...",
  "system": "vista",
  "config_path": "/app/vendor/distributed_training_estimator/Estimator/target_config/llemma_7b_4_2_2_V.yml"
}
```

Optional request field:

```json
{
  "system": "vista",
  "config_name": "llemma_7b_4_2_2_V.yml"
}
```

### `POST /chat`

Request:

```json
{
  "prompt": "Predict training time on Vista. Answer with microseconds and seconds.",
  "max_new_tokens": 120
}
```

Response:

```json
{
  "response": "..."
}
```

## Deploy with Tapis Pods

A working Tapis Pod configuration example is provided in:

```text
tapis-pod.example.json
```

Current deployment:

| Field | Value |
| --- | --- |
| Pod ID | `hpcmcpservercpu` |
| Base URL | `https://hpcmcpservercpu.pods.icicleai.tapis.io` |
| Container image | `ghcr.io/icicle-ai/hpc-mcp-server-cpu:0.1.0` |
| Release format | Service-only Tapis Pod deployment with source code in GitHub |
| ICICLEaaS category | AI-as-a-Service |

Environment variables:

```text
PORT=8000
LLM_BACKEND=transformers
MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
HF_HOME=/models/.cache/huggingface
HF_LOCAL_FILES_ONLY=false
TORCH_DTYPE=float32
ESTIMATOR_DIR=/app/vendor/distributed_training_estimator/Estimator
GPU_MODEL_TIMEOUT_SECONDS=300
```

Optional CPU demo setting:

```text
TOOL_DECISION_MAX_NEW_TOKENS=8
```

If `MODEL_ID` is a local path, download the model to that path inside the pod before calling `/chat`. For production deployments, use a persistent Tapis volume for model storage.

## Run Locally for Development

Local development prerequisites:

- Python 3.11.
- `uv`.
- Enough disk space for Python dependencies, model files, and estimator assets.

Install dependencies:

```bash
uv sync --locked
```

Run the service locally:

```bash
export LLM_BACKEND=transformers
export MODEL_ID=Qwen/Qwen2.5-0.5B-Instruct
export TORCH_DTYPE=float32
export HF_HOME=/tmp/huggingface
export ESTIMATOR_DIR=vendor/distributed_training_estimator/Estimator

uv run uvicorn hpc_mcp_server_cpu.main:app --host 127.0.0.1 --port 8000
```

Run tests:

```bash
uv run python -m unittest tests/test_mcp_transport.py
```

## Troubleshooting

If `/health` works but `/chat` fails with a model path error, check whether the configured model path exists inside the pod.

If `/chat` is slow on a CPU-only pod, use `/predict-gpu-time` for direct estimator calls or reduce `max_new_tokens` in the chat request.

If MCP clients reject the connection after redeployment under a new hostname, set `MCP_PUBLIC_HOST` to the public host and configure `MCP_ALLOWED_HOSTS` and `MCP_ALLOWED_ORIGINS` as needed.

If `curl` examples fail, make sure the JSON field is written as `max_new_tokens`, not `max\_new\_tokens`.

---

# Explanation

## What HPC-MCP Provides

HPC-MCP packages an LLM-assisted HPC utility workflow as an accessible service. Instead of requiring users to run the estimator manually inside an HPC software environment, the service exposes a small HTTP and MCP interface that can be used from command-line tools, web-facing workflows, or MCP-compatible agents.

The current release focuses on distributed training-time prediction for configurable model training workloads on Vista and Perlmutter. Future releases can add additional HPC tools, including command generation and user-guide question answering.

## How Prediction Requests Work

Direct prediction requests sent to `/predict-gpu-time` call the backend estimator with the selected system configuration. Natural-language requests sent to `/chat` first pass through the configured LLM backend for request understanding and tool routing. When the user asks for a supported training-time prediction, the service routes the request to the estimator and returns a summarized result.

The backend estimator is the Distributed Training Estimator of LLMs:

```text
https://github.com/ICICLE-ai/distributed_training_estimator_of_LLM
```

## Catalog and Release Notes

The component metadata file for catalog review is:

```text
component.yaml
```

The ICICLE software team should review the README, OpenAPI specification, and component metadata before catalog publication. Service name, ICICLEaaS category, and Tapis UI placement should be finalized through the ICICLE SDD/release process.
