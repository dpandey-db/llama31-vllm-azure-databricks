# Serving a 4-bit Llama 3.1 8B with vLLM on Azure Databricks (Canada Central)

This repo shows two ways to serve a 4-bit (bitsandbytes) Llama 3.1 8B model with **vLLM** on Azure Databricks Model Serving. Both give you an endpoint with an OpenAI-compatible chat API.

| | Custom LLM Serving | GPU Custom Model Serving (pyfunc wrapper) |
|---|---|---|
| Notebook | `notebooks/deploy_llama31_8b_bnb4.py` | `notebooks/deploy_llama31_8b_pyfunc.py` |
| Job definition | `jobs/job_custom_llm_serving.json` | `jobs/job_pyfunc_canada_central.json` |
| Needs AI Runtime (serverless GPU) | Yes, to log and register the model | No. Logs from serverless CPU or a classic cluster |
| Available in Canada Central | No (US regions only) | Yes |

**In Canada Central, use the pyfunc wrapper.** Custom LLM Serving requires the model to be logged from AI Runtime, and it's only available in US regions. GPU Custom Model Serving is supported in Canada Central, and the pyfunc wrapper runs vLLM on it without AI Runtime.

Both notebooks use `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`, which is ungated on Hugging Face, so no token is needed. To serve a different Hugging Face model, change the `hf_repo` parameter. If your fine-tuned model is already registered in Unity Catalog, use [the Unity Catalog variant](#starting-from-a-fine-tuned-model-in-unity-catalog) instead.

## Option 1 (Canada Central): GPU Custom Model Serving with a pyfunc wrapper

### How it works

The model is a standard MLflow `ChatModel`. When the endpoint loads it, `load_context` starts a vLLM OpenAI-compatible server inside the serving container, and `predict` / `predict_stream` pass chat requests to it. The model is logged with explicit `pip_requirements` (vLLM and bitsandbytes), so the serving container installs them when the endpoint is built. Nothing in this setup needs a GPU or AI Runtime until serving.

The notebook:
1. Downloads the model weights.
2. Logs the `ChatModel` wrapper with the weights as an artifact, and registers it to Unity Catalog. It doesn't use `env_pack`.
3. Creates or updates a GPU serving endpoint.
4. Queries the endpoint and runs a short throughput benchmark.

### Required endpoint setting: `GUNICORN_CMD_ARGS`

By default, the serving container runs one single-threaded web server process, which handles one request at a time. Because of that, vLLM never sees concurrent requests and can't batch them. The notebook sets this environment variable on the served model so requests are handled in parallel:

```python
environment_vars={"GUNICORN_CMD_ARGS": "--threads 16"}
```

This isn't a documented Model Serving setting, so load-test it with your own traffic before relying on it in production.

### Differences from Custom LLM Serving

- **Structured output:** `response_format` isn't a `ChatModel` parameter, so the wrapper reads it from `custom_inputs` (`{"custom_inputs": {"response_format": ...}}` in the request body). This path hasn't been tested yet.
- **Build time:** endpoint builds take longer, because vLLM is installed when the endpoint is built.
- **Log-time check:** MLflow loads and queries a `ChatModel` while logging it. On CPU, the wrapper detects that vLLM isn't installed and skips starting the server. On a GPU cluster, set `local_vllm=true` to run a real vLLM test before registering.

### Compute

- **Logging and registering:** CPU compute is enough. Serverless CPU (Standard v6 environment) has been tested, and it's what `jobs/job_pyfunc_canada_central.json` uses. A classic CPU cluster should also work. Either way, the node needs enough local disk for the weights, which are downloaded before upload: about 6 GB for this 4-bit model, or about 16 GB for a 16-bit 8B fine-tune.
- **Optional GPU test:** on a classic GPU cluster, for example `Standard_NV36ads_A10_v5` (1× A10 24 GB, which needs *Standard NVADSA10v5 Family vCPUs* quota: 36 per cluster), set `local_vllm=true` to run a real vLLM test while logging. This catches bad settings before an endpoint is built, but it isn't required.
- **Serving:** `GPU_LARGE` (1× A100 80 GB), `Small` workload size.

### Starting from a fine-tuned model in Unity Catalog

`notebooks/deploy_llama31_8b_pyfunc_from_uc.py` (job: `jobs/job_pyfunc_from_uc.json`) is the same pyfunc wrapper, for when your fine-tuned model is already registered in Unity Catalog. Only the start of the notebook changes. From the wrapper onward it's identical, including `GUNICORN_CMD_ARGS`.

| | Hugging Face version | Unity Catalog version |
|---|---|---|
| Source parameters | `hf_repo` | `source_model` (`catalog.schema.name`), `source_version`, and optional `chat_template_path` |
| Getting the weights | `snapshot_download` | `mlflow.artifacts.download_artifacts("models:/...")`, then one folder is assembled with the config, weights, and tokenizer |
| Check before logging | none | Loads the config and tokenizer on CPU and renders the chat template, so a bad layout fails in minutes |
| vLLM bitsandbytes flags | always on | on only when the model's `config.json` says the weights are bnb-quantized |
| Permissions | none | `EXECUTE` on the source model, plus `USE CATALOG` and `USE SCHEMA` |

The serving model is registered as a new model (`model_name`), separate from your source model, because it's a different kind of artifact.

**What the source model must contain:**
- **Full weights** (`*.safetensors`) with `config.json`, either merged 16-bit or merged bnb-4bit. A LoRA adapter on its own must be merged into the base model first. A model logged with `mlflow.transformers` and `save_pretrained=False` has no weights, only a Hugging Face reference, so it can't be used.
- **A tokenizer with a chat template.** If it has none, set `chat_template_path` to a `.jinja` file, for example in a UC Volume.

The notebook finds the weights and tokenizer in any of these layouts: `mlflow.transformers` (weights in `model/`, tokenizer in `components/tokenizer/`), a pyfunc with an artifacts folder, or a plain Hugging Face folder. It downloads the weights and then copies them, so the node needs local disk for two copies: about 12 GB for 4-bit, or about 32 GB for 16-bit.

## Option 2 (US regions): Custom LLM Serving

`notebooks/deploy_llama31_8b_bnb4.py` follows the official [Serve custom LLMs](https://learn.microsoft.com/azure/databricks/machine-learning/model-serving/serve-custom-llms) starter notebooks, adapted for a bnb-4bit model. It:

1. Downloads the model weights.
2. Starts vLLM inside the notebook and sends a test request, so a bad setting or out-of-memory error shows up in minutes, before any endpoint is built.
3. Logs the model to MLflow with its vLLM `entrypoint` command, and registers it to Unity Catalog as an express deployment (`env_pack="databricks_model_serving"`). This packs the notebook's Python environment together with the model.
4. Creates or updates the GPU serving endpoint.
5. Queries the endpoint and runs a short throughput benchmark.

Requirements:
- **Custom LLM Serving** enabled: Admin Settings → Previews → *Custom LLM Serving*.
- A workspace in a US region with **AI Runtime** (serverless GPU).
- The notebook must run on **serverless GPU (1× A10), Standard v6 environment**. A model registered from CPU or classic compute fails at endpoint creation with *"entrypoint is only supported with express deployments"*.

On Azure, Custom LLM Serving offers `GPU_SMALL` (T4), `GPU_LARGE` (A100), and `GPU_LARGE_RTX` (RTX PRO 6000). There's no A10 serving option.

## Getting started

The notebooks take their settings as parameters (widgets): `catalog`, `schema`, `model_name`, `endpoint`, `workload_type`, and `scale_to_zero`, plus `hf_repo` (or `source_model` and `source_version` in the Unity Catalog variant). You need `CREATE MODEL` on the target schema.

1. Import the notebooks into your workspace:
   ```bash
   databricks workspace import-dir notebooks /Workspace/Users/<your-user>/llama31-vllm --profile <profile>
   ```
2. In the job definition you want, replace `<your-user>`, `<catalog>`, and `<schema>`. Then create and run the job:
   ```bash
   databricks jobs create --json @jobs/job_pyfunc_canada_central.json --profile <profile>
   databricks jobs run-now <job-id> --profile <profile>
   ```
   You can also run the notebook interactively on the same compute.
3. Query the endpoint with any OpenAI client:
   ```python
   from openai import OpenAI
   client = OpenAI(api_key="<databricks-token>", base_url="https://<workspace-host>/serving-endpoints")
   client.chat.completions.create(model="llama31-8b-bnb4-pyfunc",
                                  messages=[{"role": "user", "content": "Hello"}])
   ```

The first endpoint build can take a while. The notebooks poll until the endpoint is ready.

## Recommendations

- **Serve on `GPU_LARGE` (A100).** An 8B 4-bit model uses only a small part of an A100, but the T4 is much slower to generate, especially with long prompts, which makes it a poor fit for an interactive app.
- **Start with one `Small` replica.** For heavier traffic, or long retrieval-style prompts, increase `workload_size`.
- **Stream responses** in the app, and **use `response_format` with a JSON schema** wherever the app parses model output. Plain JSON mode doesn't enforce the structure the app expects.
- **Cost control:** keep scale-to-zero off in production and size for peak traffic. In development, use **Stop** and **Start** on the endpoint to pause GPU billing.

## Adapting to your own model

- **Weights:** merged 16-bit or merged 4-bit weights work as-is. LoRA adapters must be merged into the base model before serving.
- **Chat template:** vLLM uses the chat template in the model's tokenizer config. If your model doesn't include one, add it, or pass `--chat-template` to vLLM.
- **dtype:** A100 and RTX PRO 6000 use bf16. The T4 doesn't support bf16, so the notebooks switch to fp16 on `GPU_SMALL`.
