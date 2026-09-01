# HPC-MCP: LLM-Assisted HPC Utility Server

HPC-MCP is a web-accessible MCP service for HPC users. It connects an LLM interface with backend HPC utilities so users can ask natural-language questions and call HPC tools through HTTP or MCP clients.

The current release focuses on distributed training-time prediction for configurable model training workloads on Vista and Perlmutter. Future releases can add additional HPC tools, including command generation and user-guide question answering.

## Training Catalog Summary

Audience:

- HPC users who want a natural-language interface to HPC utilities
- Developers building MCP clients or agents for HPC workflows
- ICICLE users testing service-based AI utilities through Tapis Pods

Use cases:

- Ask normal questions through `/chat`
- Request distributed training-time predictions for Vista and Perlmutter
- Connect an MCP client to the service and call exposed tools

Prerequisites:

- Network access to the deployed Tapis Pod endpoint
- A terminal with `curl` for HTTP examples
- Optional: Claude Code or another MCP client for MCP access

Estimated setup time:

- 5 minutes for HTTP examples
- 10-15 minutes for MCP client setup and tool discovery

## Service Information

Current deployment:

- Pod ID: `hpcmcpservercpu`
- Base URL: `https://hpcmcpservercpu.pods.icicleai.tapis.io`
- MCP endpoint: `https://hpcmcpservercpu.pods.icicleai.tapis.io/mcp`
- Container image: `ghcr.io/icicle-ai/hpc-mcp-server-cpu:0.1.0`
- Release format: service-only Tapis Pod deployment with source code in GitHub

Upstream estimator tool:

- Distributed Training Estimator of LLMs: `https://github.com/ICICLE-ai/distributed_training_estimator_of_LLM`

## Features

- LLM chat through `/chat`
- Training-time prediction through `/chat`
- Direct distributed training-time prediction through `/predict-gpu-time`
- MCP access over Streamable HTTP at `/mcp`
- Service health checking through `/health`

The LLM handles request understanding and tool routing. Numerical training-time prediction is computed by the backend estimator.

Example question:

```text
Predict training time for the LLaMA 7B training configuration on Vista. Answer with microseconds and seconds.
```

## Quick Start

Check that the service is running:

```bash
curl https://hpcmcpservercpu.pods.icicleai.tapis.io/health
```

Expected result:

```json
{
  "status": "ok",
  "service": "hpc-mcp-server-cpu",
  "version": "0.1.0"
}
```

Ask a short question:

```bash
curl --http1.1 --max-time 180 https://hpcmcpservercpu.pods.icicleai.tapis.io/chat \
  -H "Content-Type: application/json" \
  --data-raw '{"prompt":"Say hello in one short sentence.","max_new_tokens":30}'
```

Ask for a training-time prediction:

```bash
curl --http1.1 --max-time 180 https://hpcmcpservercpu.pods.icicleai.tapis.io/chat \
  -H "Content-Type: application/json" \
  --data-raw '{"prompt":"Predict training time on Vista. Answer with microseconds and seconds.","max_new_tokens":120}'
```

Example:

```json
{
  "response": "System: vista\nConfig: /app/vendor/distributed_training_estimator/Estimator/target_config/llemma_7b_4_2_2_V.yml\nEstimated timecost: 4764606.22498043 us (4.765 seconds)."
}
```

Call the estimator directly:

```bash
curl --http1.1 --max-time 180 https://hpcmcpservercpu.pods.icicleai.tapis.io/predict-gpu-time \
  -H "Content-Type: application/json" \
  --data-raw '{"system":"perlmutter"}'
```

Supported `system` values:

- `vista`
- `perlmutter`

Common misspellings such as `permutter` and `permultter` are also accepted.

## API Reference

The static OpenAPI specification is saved at:

```text
docs/openapi.json
```

Live OpenAPI URL:

```text
https://hpcmcpservercpu.pods.icicleai.tapis.io/openapi.json
```

Main HTTP endpoints:

```text
GET  /health
GET  /
GET  /about
POST /chat
POST /predict-gpu-time
```

### POST /chat

Request body:

```json
{
  "prompt": "Predict training time on Vista. Answer with microseconds and seconds.",
  "max_new_tokens": 120
}
```

Response body:

```json
{
  "response": "..."
}
```

### POST /predict-gpu-time

Request body:

```json
{
  "system": "vista"
}
```

Response body:

```json
{
  "response": "...",
  "system": "vista",
  "config_path": "/app/vendor/distributed_training_estimator/Estimator/target_config/llemma_7b_4_2_2_V.yml"
}
```

## MCP Client Access

Claude Code or another MCP client can connect through HTTP transport:

```bash
claude mcp add hpc-mcp-server --transport http https://hpcmcpservercpu.pods.icicleai.tapis.io/mcp
```

After connecting, an agent can call the exposed MCP tools:

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

Set `MCP_PUBLIC_HOST` when deploying under a different hostname.

## Tapis Pod Deployment

A working Tapis Pod configuration example is provided in:

```text
tapis-pod.example.json
```

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

If `MODEL_ID` is a local path, download the model to that path inside the pod before calling `/chat`.

## Local Development

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

If `/health` works but `/chat` fails with a model path error, check whether the configured model path exists inside the pod:

```bash
ls /tmp/huggingface/Qwen2.5-0.5B-Instruct
```

If the directory is missing, download the model again or use a persistent Tapis volume for model storage.

If `/chat` returns slowly on CPU, reduce the number of requested tokens:

```json
{
  "prompt": "Say hello.",
  "max_new_tokens": 5
}
```

For demos on CPU-only pods, setting `TOOL_DECISION_MAX_NEW_TOKENS=8` can reduce routing latency.

If `curl` examples fail, make sure the JSON field is written as `max_new_tokens`, not `max\_new\_tokens`.

## Maintainers and Support

Maintainer:

- Molang Wu

For release, deployment, or ICICLE Tapis UI questions, contact the ICICLE software team.

For bugs or documentation issues, open an issue in the GitHub repository:

```text
https://github.com/ICICLE-ai/hpc-mcp-server/issues
```

## Acknowledgements

This work is part of the National Science Foundation funded AI Institute for Intelligent Cyberinfrastructure with Computational Learning in the Environment (ICICLE), OAC 2112606.

## License

License information should be finalized with the project PI and ICICLE software release team before public source-code release.

## Tags

- Software
- AI-as-a-Service
- HPC
- MCP
- Tapis Pods
- ICICLE
