# Databricks notebook source
# MAGIC %md
# MAGIC # Serve a Llama 3.1 8B from Unity Catalog on vLLM with an OpenAI *completions* API
# MAGIC
# MAGIC Variant of `deploy_llama31_8b_pyfunc_from_uc` for apps that send a **raw, pre-formatted prompt** to
# MAGIC `/v1/completions` (`client.completions.create(prompt=...)`, reading `choices[0].text`) instead of chat
# MAGIC `messages`. The weights, vLLM server, `GUNICORN_CMD_ARGS` setting, and deploy steps are unchanged; only
# MAGIC the wrapper differs.
# MAGIC
# MAGIC ### What changes compared with `deploy_llama31_8b_pyfunc_from_uc`
# MAGIC | | Chat version | This notebook |
# MAGIC |---|---|---|
# MAGIC | Wrapper | `mlflow.pyfunc.ChatModel` | `mlflow.pyfunc.PythonModel` with an explicit completions signature |
# MAGIC | Request | `{"messages": [...], ...}` | `{"prompt": "...", "max_tokens": ..., "temperature": ..., "stop": [...]}` |
# MAGIC | Response | `choices[].message.content` | `choices[].text` (OpenAI `text_completion` object) |
# MAGIC | vLLM route | `/v1/chat/completions` | `/v1/completions` |
# MAGIC | Model metadata | `task: agent/v1/chat` (set by `ChatModel`) | `task: llm/v1/completions` |
# MAGIC | Chat template | required | optional: the caller formats the prompt |
# MAGIC
# MAGIC A request with a top-level `prompt` key is treated by the MLflow scoring server as an LLM request, so the
# MAGIC endpoint returns the completion object as-is, with no `{"predictions": ...}` wrapper.
# MAGIC
# MAGIC ### Request limits
# MAGIC - `stop` must be a list of strings (`["<|eot_id|>"]`), not a single string.
# MAGIC - Streaming (`stream=true`) isn't supported; requests with it are rejected.
# MAGIC - Supported sampling fields: `max_tokens`, `temperature`, `top_p`, `top_k`, `stop`, `n`, `seed`,
# MAGIC   `frequency_penalty`, `presence_penalty`. Other fields are ignored.
# MAGIC
# MAGIC ### What the source model must contain
# MAGIC **Full weights** (`*.safetensors`) with `config.json`: merged 16-bit or merged bnb-4bit, plus the tokenizer.
# MAGIC A LoRA adapter on its own must be merged first. A `mlflow.transformers` model logged with
# MAGIC `save_pretrained=False` has no weights and can't be used. Supported layouts, found automatically:
# MAGIC `mlflow.transformers` (weights in `model/`, tokenizer in `components/tokenizer/`), a pyfunc with an
# MAGIC artifacts folder, or a plain Hugging Face folder.
# MAGIC
# MAGIC ### Prerequisites
# MAGIC CPU compute is enough (tested on serverless CPU). The node needs local disk for the weights twice over
# MAGIC (downloaded copy + assembled folder): ~12 GB for 4-bit, ~32 GB for 16-bit. You need `EXECUTE` on the
# MAGIC source model (plus `USE CATALOG` / `USE SCHEMA`) and `CREATE MODEL` on the target schema.

# COMMAND ----------

# MAGIC %pip install -q mlflow==3.13.0 transformers==5.8.1 openai==2.37.0 "databricks-sdk>=0.102.0"
# MAGIC %restart_python

# COMMAND ----------

# On a GPU cluster, install the serving stack here too so MLflow's log-time load runs vLLM for real.
dbutils.widgets.dropdown("local_vllm", "false", ["true", "false"], "Install vLLM locally (GPU compute only)")
dbutils.widgets.text("vllm_version", "0.24.0", "vLLM version installed in the serving env")
if dbutils.widgets.get("local_vllm") == "true":
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           f"vllm=={dbutils.widgets.get('vllm_version')}", "bitsandbytes>=0.48.1"])
    dbutils.library.restartPython()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Config (widgets)

# COMMAND ----------

dbutils.widgets.text("source_model", "main.llm_serving.llama31_8b_finetuned", "Source UC model (catalog.schema.name)")
dbutils.widgets.text("source_version", "1", "Source UC model version")
dbutils.widgets.text("catalog", "main", "UC catalog (serving model)")
dbutils.widgets.text("schema", "llm_serving", "UC schema (serving model)")
dbutils.widgets.text("model_name", "llama31_8b_completions_pyfunc", "UC registered model name (serving model)")
dbutils.widgets.text("endpoint", "llama31-8b-completions", "Serving endpoint name")
dbutils.widgets.dropdown("workload_type", "GPU_LARGE", ["GPU_LARGE", "GPU_LARGE_RTX", "GPU_SMALL"], "Serving GPU")
dbutils.widgets.dropdown("scale_to_zero", "false", ["true", "false"], "Scale-to-zero")
dbutils.widgets.dropdown("deploy", "true", ["true", "false"], "Create/update the endpoint")

SOURCE_MODEL = dbutils.widgets.get("source_model")
SOURCE_VERSION = dbutils.widgets.get("source_version")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
MODEL_NAME = dbutils.widgets.get("model_name")
ENDPOINT_NAME = dbutils.widgets.get("endpoint")
WORKLOAD_TYPE_STR = dbutils.widgets.get("workload_type")
VLLM_VERSION = dbutils.widgets.get("vllm_version")
SCALE_TO_ZERO = dbutils.widgets.get("scale_to_zero") == "true"
DEPLOY = dbutils.widgets.get("deploy") == "true"

UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"
if UC_MODEL_NAME == SOURCE_MODEL:
    raise ValueError("Register the serving model under a different name than the source model.")
ARTIFACTS_PATH = "model"                  # local weights dir; logged as the `model_dir` artifact

# vLLM tuning, passed to the wrapper through model_config. `bnb` is set in step 1 from the model's config.json.
# A100/RTX are Ampere+ -> bf16; T4 (GPU_SMALL) is Turing with no bf16 -> fp16.
MODEL_CONFIG = {
    "dtype": "float16" if WORKLOAD_TYPE_STR == "GPU_SMALL" else "bfloat16",
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.85,
    "port": 8100,                         # internal to the container; the scoring server owns 8080
    "startup_timeout_s": 1500,
}

print(f"source={SOURCE_MODEL} v{SOURCE_VERSION} -> uc_model={UC_MODEL_NAME} | endpoint={ENDPOINT_NAME} | workload={WORKLOAD_TYPE_STR}")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import AlreadyExists, ResourceAlreadyExists

# Create the UC model up front so a permission problem fails in seconds, not after the download.
try:
    WorkspaceClient().registered_models.create(catalog_name=CATALOG, schema_name=SCHEMA, name=MODEL_NAME)
except (AlreadyExists, ResourceAlreadyExists):
    pass


# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Get the weights from the source UC model
# MAGIC Downloads the source model version, then assembles one folder with `config.json`, the weights, and the
# MAGIC tokenizer, which is what vLLM's `--model` expects. `mlflow.transformers` keeps the tokenizer in a separate
# MAGIC `components/tokenizer/` folder, so its files are merged in.

# COMMAND ----------

import glob, json, os, shutil, tempfile
import mlflow

os.chdir(tempfile.mkdtemp())                         # /Workspace can't hold large weights
os.environ["MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC"] = "true"
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "600"
os.environ["MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT"] = "600"
mlflow.set_registry_uri("databricks-uc")

src = mlflow.artifacts.download_artifacts(f"models:/{SOURCE_MODEL}/{SOURCE_VERSION}", dst_path="source")

def dirs_with(pattern):
    return sorted({os.path.dirname(p) for p in glob.glob(f"{src}/**/{pattern}", recursive=True)})

if dirs_with("adapter_config.json") and not dirs_with("*.safetensors"):
    raise ValueError("The source model is a LoRA adapter. Merge it into the base model and register the merged weights.")
weight_dirs = [d for d in dirs_with("config.json") if glob.glob(f"{d}/*.safetensors")]
if not weight_dirs:
    raise ValueError("No *.safetensors next to a config.json in the source model. If it was logged with "
                     "mlflow.transformers(save_pretrained=False), it holds only a Hugging Face reference: re-log it with weights.")
if len(weight_dirs) > 1:
    raise ValueError(f"Found more than one weights folder, pick one: {weight_dirs}")

shutil.rmtree(ARTIFACTS_PATH, ignore_errors=True)
shutil.copytree(weight_dirs[0], ARTIFACTS_PATH)
# Merge the tokenizer in if it lives elsewhere (mlflow.transformers: components/tokenizer/). Never overwrite.
for tok_dir in dirs_with("tokenizer_config.json"):
    for f in os.listdir(tok_dir):
        if not os.path.exists(f"{ARTIFACTS_PATH}/{f}") and os.path.isfile(f"{tok_dir}/{f}"):
            shutil.copy2(f"{tok_dir}/{f}", ARTIFACTS_PATH)
shutil.rmtree("source")                              # free the disk before logging

with open(f"{ARTIFACTS_PATH}/config.json") as f:
    quant = json.load(f).get("quantization_config") or {}
MODEL_CONFIG["bnb"] = quant.get("quant_method") == "bitsandbytes"
print(f"from {os.path.relpath(weight_dirs[0], src)}/ | bnb={MODEL_CONFIG['bnb']}")
print("model folder:", sorted(os.listdir(ARTIFACTS_PATH)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1b. Check the folder before logging
# MAGIC Loads the config and tokenizer the way vLLM will. Runs on CPU in seconds; without it, a missing tokenizer
# MAGIC only shows up after a long endpoint build. If the tokenizer has a chat template, it's used in step 6 to
# MAGIC build a correctly formatted test prompt.

# COMMAND ----------

from transformers import AutoConfig, AutoTokenizer

missing = [f for f in ("config.json", "tokenizer_config.json") if not os.path.exists(f"{ARTIFACTS_PATH}/{f}")]
if missing:
    raise ValueError(f"Model folder is missing {missing}.")
hf_config = AutoConfig.from_pretrained(ARTIFACTS_PATH)
tok = AutoTokenizer.from_pretrained(ARTIFACTS_PATH)
print(f"architecture={hf_config.architectures} | vocab={len(tok)} | chat template: {bool(tok.chat_template)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Completions wrapper around a vLLM OpenAI server
# MAGIC `load_context` starts vLLM once per container (a file lock stops extra scoring workers from starting a
# MAGIC second copy) and waits for `/health`. `predict` forwards each request to `/v1/completions` and returns
# MAGIC vLLM's response unchanged.

# COMMAND ----------

import mlflow
from mlflow.models import ModelSignature
from mlflow.types.llm import COMPLETIONS_MODEL_OUTPUT_SCHEMA
from mlflow.types.schema import Array, ColSpec, DataType, Schema

class VLLMCompletionsModel(mlflow.pyfunc.PythonModel):
    """Raw-prompt completions (OpenAI /v1/completions) backed by a local vLLM server."""

    SAMPLING_PARAMS = ("max_tokens", "temperature", "top_p", "top_k", "stop", "n", "seed",
                       "frequency_penalty", "presence_penalty")

    def load_context(self, context):
        import fcntl, importlib.util, os, subprocess, sys, time
        import requests

        cfg = context.model_config
        self.base_url = f"http://127.0.0.1:{cfg['port']}"
        self.session = requests.Session()
        # No vLLM installed (e.g. a CPU notebook loading the model) and no server already running:
        # answer with a placeholder instead of failing. The serving env installs vLLM from pip_requirements.
        self.validate_only = importlib.util.find_spec("vllm") is None and not self._healthy()
        if self.validate_only:
            return
        with open("/tmp/vllm.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)         # held until this worker's server is up (or found)
            if not self._healthy():
                cmd = [sys.executable, "-u", "-m", "vllm.entrypoints.openai.api_server",
                       "--model", context.artifacts["model_dir"],
                       "--served-model-name", "llm",
                       "--host", "127.0.0.1",
                       "--port", str(cfg["port"]),
                       "--dtype", cfg["dtype"],
                       "--max-model-len", str(cfg["max_model_len"]),
                       "--gpu-memory-utilization", str(cfg["gpu_memory_utilization"])]
                if cfg["bnb"]:                       # 4-bit bnb weights; merged 16-bit weights need no flags
                    cmd += ["--quantization", "bitsandbytes", "--load-format", "bitsandbytes"]
                env = {**os.environ, "VLLM_USE_FLASHINFER_SAMPLER": "0"}
                # Inherit stdout/stderr so vLLM's log shows in the endpoint's Logs tab.
                server = subprocess.Popen(cmd, env=env, start_new_session=True)
                deadline = time.time() + cfg["startup_timeout_s"]
                while not self._healthy():
                    if server.poll() is not None:
                        raise RuntimeError(f"vLLM exited with code {server.returncode}; see the endpoint logs.")
                    if time.time() > deadline:
                        raise RuntimeError("vLLM did not become healthy in time; see the endpoint logs.")
                    time.sleep(5)

    def _healthy(self):
        import requests
        try:
            return requests.get(f"{self.base_url}/health", timeout=2).ok
        except requests.ConnectionError:
            return False

    @staticmethod
    def _rows(model_input):
        """One dict per request. Unified LLM requests ({"prompt": ...}) arrive as a one-row DataFrame or a dict."""
        import math
        import pandas as pd
        if isinstance(model_input, pd.DataFrame):
            rows = model_input.to_dict(orient="records")
        elif isinstance(model_input, dict):
            rows = [model_input]
        else:
            rows = list(model_input)
        clean = []
        for row in rows:
            out = {}
            for k, v in row.items():
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    continue                        # optional field the caller left out
                out[k] = v.tolist() if hasattr(v, "tolist") else v
            clean.append(out)
        return clean

    def _complete(self, row):
        if row.get("stream"):
            raise ValueError("stream=true isn't supported by this endpoint; send a non-streaming request.")
        body = {"model": "llm", "prompt": row["prompt"],
                **{k: row[k] for k in self.SAMPLING_PARAMS if k in row}}
        if self.validate_only:
            return {"id": "validation", "object": "text_completion", "created": 0, "model": "llm",
                    "choices": [{"index": 0, "text": "validation placeholder", "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
        r = self.session.post(f"{self.base_url}/v1/completions", json=body, timeout=290)
        r.raise_for_status()
        return r.json()

    def predict(self, context, model_input, params=None):
        outs = [self._complete(row) for row in self._rows(model_input)]
        # A single unified-LLM request returns the bare OpenAI completion object (no "predictions" wrapper).
        return outs[0] if len(outs) == 1 else outs


# Request schema: OpenAI completions fields. All optional except `prompt`; unknown fields (e.g. `model`) are dropped.
SIGNATURE = ModelSignature(
    inputs=Schema([
        ColSpec(DataType.string, "prompt"),
        ColSpec(DataType.double, "temperature", required=False),
        ColSpec(DataType.long, "max_tokens", required=False),
        ColSpec(Array(DataType.string), "stop", required=False),
        ColSpec(DataType.long, "n", required=False),
        ColSpec(DataType.double, "top_p", required=False),
        ColSpec(DataType.long, "top_k", required=False),
        ColSpec(DataType.long, "seed", required=False),
        ColSpec(DataType.double, "frequency_penalty", required=False),
        ColSpec(DataType.double, "presence_penalty", required=False),
        ColSpec(DataType.boolean, "stream", required=False),
    ]),
    outputs=COMPLETIONS_MODEL_OUTPUT_SCHEMA,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Log → register (no `env_pack`; the serving container installs `pip_requirements`)

# COMMAND ----------

import cloudpickle, requests, subprocess
# The SDK uploader and longer timeouts for multi-GB files were set in step 1.

# Explicit requirements: inference from this CPU notebook would miss vLLM entirely.
SERVING_REQS = [
    f"mlflow=={mlflow.__version__}",
    f"cloudpickle=={cloudpickle.__version__}",   # must match the pickler of python_model
    f"requests=={requests.__version__}",
    f"vllm=={VLLM_VERSION}",
    "bitsandbytes>=0.48.1",
]

with mlflow.start_run(run_name=f"deploy_{MODEL_NAME}"):
    model_info = mlflow.pyfunc.log_model(
        name=MODEL_NAME,
        python_model=VLLMCompletionsModel(),
        artifacts={"model_dir": ARTIFACTS_PATH},
        model_config=MODEL_CONFIG,
        signature=SIGNATURE,
        metadata={"task": "llm/v1/completions"},
        pip_requirements=SERVING_REQS,
    )
print("logged:", model_info.model_uri)
subprocess.run(["pkill", "-f", "vllm.entrypoints.openai.api_server"])   # free the GPU after the local load

model_version = mlflow.register_model(model_info.model_uri, UC_MODEL_NAME)
print("registered:", UC_MODEL_NAME, "v", model_version.version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Deploy the GPU endpoint
# MAGIC The build installs vLLM and then loads the model, so expect a long first deployment. We poll below.

# COMMAND ----------

import time
from datetime import timedelta
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput, ServedEntityInput, ServingModelWorkloadType)

w = WorkspaceClient()
entities = [ServedEntityInput(
    entity_name=UC_MODEL_NAME, entity_version=str(model_version.version),
    workload_type=ServingModelWorkloadType(WORKLOAD_TYPE_STR), workload_size="Small",
    scale_to_zero_enabled=SCALE_TO_ZERO,
    # The scoring server runs one sync gunicorn worker, which serializes requests so vLLM never batches.
    # Threads let concurrent requests reach the shared vLLM server (~93 -> ~440 gen tok/s at 16 users on A100).
    environment_vars={"GUNICORN_CMD_ARGS": "--threads 16"},
)]
if not DEPLOY:
    print("deploy=false -> registered only; skipping the endpoint")
elif ENDPOINT_NAME in [e.name for e in w.serving_endpoints.list()]:
    w.serving_endpoints.update_config(name=ENDPOINT_NAME, served_entities=entities)
else:
    w.serving_endpoints.create(
        name=ENDPOINT_NAME, config=EndpointCoreConfigInput(name=ENDPOINT_NAME, served_entities=entities))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Wait for the new config (polls up to ~90 min)

# COMMAND ----------

deadline = time.time() + 90 * 60
ready = False
while DEPLOY and time.time() < deadline:
    e = w.serving_endpoints.get(ENDPOINT_NAME)
    ready_v = getattr(e.state.ready, "value", "?")
    cu_v = getattr(e.state.config_update, "value", "?")
    print(time.strftime("%T"), "ready=", ready_v, "config_update=", cu_v)
    if cu_v == "UPDATE_FAILED":
        raise RuntimeError(f"Endpoint update failed; check build and service logs for {ENDPOINT_NAME}.")
    if ready_v == "READY" and cu_v == "NOT_UPDATING":
        ready = True; break
    time.sleep(30)
print("endpoint READY:", ready)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Query the endpoint (OpenAI-compatible completions)
# MAGIC The caller formats the prompt. `fmt` uses the tokenizer's chat template when it has one, and falls back to
# MAGIC the Llama 3 prompt format. The raw REST call checks that the response is the bare completion object.

# COMMAND ----------

import requests
from openai import OpenAI

TOKEN = w.config.authenticate()["Authorization"].removeprefix("Bearer ")
client = OpenAI(api_key=TOKEN, base_url=f"{w.config.host}/serving-endpoints")
SYSTEM = "You are a concise technical assistant."

def fmt(user, system=SYSTEM):
    """Pre-formatted prompt string, ending with the assistant header so the model answers."""
    if tok.chat_template:
        return tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                       tokenize=False, add_generation_prompt=True)
    return (f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n{user}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n")

if ready:
    prompt = fmt("Give me three bullet points on why 4-bit quantization helps LLM serving.")
    print(prompt)
    resp = client.completions.create(model=ENDPOINT_NAME, prompt=prompt, max_tokens=256, temperature=0.3,
                                     stop=["<|eot_id|>"])
    print(resp.choices[0].text)
    print("\nfinish_reason:", resp.choices[0].finish_reason, "| usage:", resp.usage)

    raw = requests.post(f"{w.config.host}/serving-endpoints/{ENDPOINT_NAME}/invocations",
                        headers={"Authorization": f"Bearer {TOKEN}"},
                        json={"prompt": fmt("Say hello in five words."), "max_tokens": 20, "temperature": 0}, timeout=300)
    raw.raise_for_status()
    body = raw.json()
    print("\nREST response keys:", sorted(body))
    assert "predictions" not in body and body["object"] == "text_completion", body
else:
    print("endpoint not READY in the wait window -- poll the UI, then re-run from here.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Light tokens/s benchmark

# COMMAND ----------

import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor

GEN_TOKENS = 256
PROMPTS = [
    "Explain vector databases to a data engineer in 4 sentences.",
    "Write a short Python function that reverses a linked list.",
    "Summarize the tradeoffs of 4-bit vs 8-bit LLM quantization.",
    "Describe how vLLM paged attention improves throughput.",
    "List five best practices for prompt engineering.",
    "What is Unity Catalog and why does governance matter?",
    "Draft a two-sentence product description for a GPU serving platform.",
    "Explain the difference between latency and throughput for LLMs.",
]

def one_call(prompt):
    t0 = time.perf_counter()
    r = client.completions.create(model=ENDPOINT_NAME, prompt=fmt(prompt), max_tokens=GEN_TOKENS, temperature=0.7)
    return time.perf_counter() - t0, r.usage.completion_tokens

def run_bench(concurrency, n_requests=16):
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(n_requests)]
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        out = list(ex.map(one_call, prompts))
    wall = time.perf_counter() - t0
    lat = np.array([o[0] for o in out])
    gen = int(sum(o[1] for o in out))
    return {"concurrency": concurrency, "requests": n_requests, "gen_tokens": gen,
            "wall_s": round(wall, 2), "gen_tokens_per_s": round(gen / wall, 1),
            "req_per_s": round(n_requests / wall, 2),
            "p50_ms": round(float(np.percentile(lat, 50)) * 1000, 1),
            "p95_ms": round(float(np.percentile(lat, 95)) * 1000, 1)}

results = []
if ready:
    print("warmup:", one_call(PROMPTS[0]))
    for conc in [1, 4, 8]:
        r = run_bench(conc)
        print(r)
        results.append(r)
    display(pd.DataFrame(results).sort_values("gen_tokens_per_s", ascending=False))
else:
    print("skipping benchmark -- endpoint not READY")

# COMMAND ----------

import json
summary = {"source": f"{SOURCE_MODEL}/{SOURCE_VERSION}", "bnb": MODEL_CONFIG["bnb"],
           "model_files": {f: os.path.getsize(f"{ARTIFACTS_PATH}/{f}") for f in sorted(os.listdir(ARTIFACTS_PATH))},
           "uc_model": UC_MODEL_NAME, "version": str(model_version.version),
           "endpoint": ENDPOINT_NAME, "workload": WORKLOAD_TYPE_STR, "ready": ready}
if results:
    best = max(results, key=lambda r: r["gen_tokens_per_s"])
    summary["peak_gen_tokens_per_s"] = best["gen_tokens_per_s"]
    summary["peak_at_concurrency"] = best["concurrency"]
dbutils.notebook.exit(json.dumps(summary))
