# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` on vLLM (classic GPU Custom Model Serving)
# MAGIC
# MAGIC Variant of the Custom LLM Serving notebook for workspaces **without AI Runtime** (e.g. Canada Central).
# MAGIC No `entrypoint` and no `env_pack`: the model is a regular MLflow `ChatModel` whose `load_context`
# MAGIC starts a vLLM OpenAI server inside the serving container and whose `predict` / `predict_stream`
# MAGIC forward chat requests to it. The serving container builds the environment from `pip_requirements`.
# MAGIC
# MAGIC ### What it does
# MAGIC 1. Downloads the HF weights to local disk.
# MAGIC 2. Logs the `ChatModel` wrapper with the weights as an artifact and pinned serving requirements.
# MAGIC 3. Registers to Unity Catalog (no `env_pack`).
# MAGIC 4. Creates/updates a GPU serving endpoint.
# MAGIC 5. Queries it (OpenAI-compatible chat) and runs a light tokens/s benchmark.
# MAGIC
# MAGIC ### Prerequisites
# MAGIC - **CPU compute is enough** to log and register this model (`local_vllm=false`, the default). Tested on
# MAGIC   **serverless CPU**; a classic CPU cluster should also work. The node needs enough local disk for the
# MAGIC   weights (~6 GB for this 4-bit model, ~16 GB for a 16-bit 8B fine-tune), since they're downloaded before upload.
# MAGIC - Optional: on a **classic GPU cluster** (e.g. ML runtime on `Standard_NV36ads_A10_v5`), set `local_vllm=true`
# MAGIC   to install vLLM and run a real vLLM test while logging, before any endpoint is built. Not required.
# MAGIC - GPU Custom Model Serving available in the workspace region.
# MAGIC
# MAGIC ### Differences from the entrypoint route
# MAGIC - MLflow loads and queries a `ChatModel` while logging it. With `local_vllm=true` that's a real vLLM
# MAGIC   smoke test on the cluster's GPU; on CPU it's skipped and the first real load happens on the endpoint.
# MAGIC - `response_format` isn't a `ChatModel` param: pass it as `custom_inputs={"response_format": ...}`.
# MAGIC - The endpoint build installs vLLM server-side, so builds are slower.

# COMMAND ----------

# MAGIC %pip install -q mlflow==3.13.0 "huggingface_hub>=0.27" hf_transfer==0.1.9 openai==2.37.0 "databricks-sdk>=0.102.0"
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

dbutils.widgets.text("catalog", "main", "UC catalog")
dbutils.widgets.text("schema", "llm_serving", "UC schema")
dbutils.widgets.text("model_name", "llama31_8b_bnb4_pyfunc", "UC registered model name")
dbutils.widgets.text("endpoint", "llama31-8b-bnb4-pyfunc", "Serving endpoint name")
dbutils.widgets.text("hf_repo", "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", "HuggingFace repo")
dbutils.widgets.dropdown("workload_type", "GPU_LARGE", ["GPU_LARGE", "GPU_LARGE_RTX", "GPU_SMALL"], "Serving GPU")
dbutils.widgets.dropdown("scale_to_zero", "false", ["true", "false"], "Scale-to-zero")
dbutils.widgets.dropdown("deploy", "true", ["true", "false"], "Create/update the endpoint")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
MODEL_NAME = dbutils.widgets.get("model_name")
ENDPOINT_NAME = dbutils.widgets.get("endpoint")
HF_REPO = dbutils.widgets.get("hf_repo")
WORKLOAD_TYPE_STR = dbutils.widgets.get("workload_type")
VLLM_VERSION = dbutils.widgets.get("vllm_version")
SCALE_TO_ZERO = dbutils.widgets.get("scale_to_zero") == "true"
DEPLOY = dbutils.widgets.get("deploy") == "true"

UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"
ARTIFACTS_PATH = "model"                  # local weights dir; logged as the `model_dir` artifact

# vLLM tuning, passed to the wrapper through model_config.
# A100/RTX are Ampere+ -> bf16; T4 (GPU_SMALL) is Turing with no bf16 -> fp16.
MODEL_CONFIG = {
    "dtype": "float16" if WORKLOAD_TYPE_STR == "GPU_SMALL" else "bfloat16",
    "max_model_len": 8192,
    "gpu_memory_utilization": 0.85,
    "port": 8100,                         # internal to the container; the scoring server owns 8080
    "startup_timeout_s": 1500,
}

print(f"uc_model={UC_MODEL_NAME} | endpoint={ENDPOINT_NAME} | workload={WORKLOAD_TYPE_STR} | vllm={VLLM_VERSION}")
print(f"model_config={MODEL_CONFIG}")

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
# MAGIC ## 1. Download weights to local disk

# COMMAND ----------

import os, shutil, tempfile
os.environ["HF_HOME"] = tempfile.mkdtemp()           # serverless default cache (~/.cache) isn't writable
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.chdir(tempfile.mkdtemp())                         # /Workspace can't hold large weights

from huggingface_hub import snapshot_download

if not os.path.exists(f"{ARTIFACTS_PATH}/config.json"):
    snapshot_download(repo_id=HF_REPO, local_dir=ARTIFACTS_PATH)
shutil.rmtree(f"{ARTIFACTS_PATH}/.cache", ignore_errors=True)   # HF download metadata; don't log it
print("local weights:", sorted(os.listdir(ARTIFACTS_PATH))[:8], "...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. ChatModel wrapper around a vLLM OpenAI server
# MAGIC `load_context` starts vLLM once per container (a file lock stops extra scoring workers from starting a
# MAGIC second copy) and waits for `/health`. `predict` and `predict_stream` forward to `/v1/chat/completions`.

# COMMAND ----------

import mlflow
from mlflow.pyfunc.model import ChatModel
from mlflow.types.llm import ChatCompletionChunk, ChatCompletionResponse

class VLLMChatModel(ChatModel):
    SAMPLING_PARAMS = ("temperature", "max_tokens", "stop", "n", "top_p", "top_k",
                       "frequency_penalty", "presence_penalty")

    def load_context(self, context):
        import fcntl, importlib.util, os, subprocess, sys, time
        import requests

        cfg = context.model_config
        self.base_url = f"http://127.0.0.1:{cfg['port']}"
        self.session = requests.Session()
        # MLflow calls load_context + predict when logging a ChatModel. The CPU logging notebook has no
        # vLLM, so skip the server there; the serving env installs vLLM from pip_requirements.
        self.validate_only = importlib.util.find_spec("vllm") is None
        if self.validate_only:
            return
        with open("/tmp/vllm.lock", "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)         # held until this worker's server is up (or found)
            if not self._healthy():
                cmd = [sys.executable, "-u", "-m", "vllm.entrypoints.openai.api_server",
                       "--model", context.artifacts["model_dir"],
                       "--served-model-name", "llm",
                       "--quantization", "bitsandbytes",     # 4-bit bnb weights
                       "--load-format", "bitsandbytes",
                       "--host", "127.0.0.1",
                       "--port", str(cfg["port"]),
                       "--dtype", cfg["dtype"],
                       "--max-model-len", str(cfg["max_model_len"]),
                       "--gpu-memory-utilization", str(cfg["gpu_memory_utilization"])]
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

    def _body(self, messages, params, stream):
        body = {"model": "llm", "messages": [m.to_dict() for m in messages], "stream": stream}
        p = params.to_dict()
        body.update({k: p[k] for k in self.SAMPLING_PARAMS if p.get(k) is not None})
        if params.tools:
            body["tools"] = [t.to_dict() for t in params.tools]
        body.update(params.custom_inputs or {})        # e.g. {"response_format": {...}}
        return body

    def predict(self, context, messages, params):
        if self.validate_only:
            return ChatCompletionResponse.from_dict(
                {"choices": [{"message": {"role": "assistant", "content": "validation placeholder"}}]})
        r = self.session.post(f"{self.base_url}/v1/chat/completions",
                              json=self._body(messages, params, stream=False), timeout=290)
        r.raise_for_status()
        return ChatCompletionResponse.from_dict(r.json())

    def predict_stream(self, context, messages, params):
        import json
        with self.session.post(f"{self.base_url}/v1/chat/completions",
                               json=self._body(messages, params, stream=True), stream=True, timeout=290) as r:
            r.raise_for_status()
            for line in r.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ")
                if data == "[DONE]":
                    break
                yield ChatCompletionChunk.from_dict(json.loads(data))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Log → register (no `env_pack`; the serving container installs `pip_requirements`)

# COMMAND ----------

import cloudpickle, requests, subprocess

# Upload the weights with the Databricks SDK and longer timeouts: MLflow's default uploader often fails on multi-GB files.
os.environ["MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC"] = "true"
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "600"
os.environ["MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT"] = "600"

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
        python_model=VLLMChatModel(),
        artifacts={"model_dir": ARTIFACTS_PATH},
        model_config=MODEL_CONFIG,
        pip_requirements=SERVING_REQS,
    )
print("logged:", model_info.model_uri)
subprocess.run(["pkill", "-f", "vllm.entrypoints.openai.api_server"])   # free the GPU after the local load

mlflow.set_registry_uri("databricks-uc")
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
    # Threads let concurrent requests reach the shared vLLM server so it can batch them.
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
# MAGIC ## 6. Query the endpoint (OpenAI-compatible chat)

# COMMAND ----------

from openai import OpenAI

TOKEN = w.config.authenticate()["Authorization"].removeprefix("Bearer ")
client = OpenAI(api_key=TOKEN, base_url=f"{w.config.host}/serving-endpoints")

if ready:
    resp = client.chat.completions.create(
        model=ENDPOINT_NAME,
        messages=[{"role": "user", "content": "Give me three bullet points on why 4-bit quantization helps LLM serving."}],
        max_tokens=256, temperature=0.3,
    )
    print(resp.choices[0].message.content)
    print("\nusage:", resp.usage)
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
    r = client.chat.completions.create(
        model=ENDPOINT_NAME,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=GEN_TOKENS, temperature=0.7,
    )
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
summary = {"uc_model": UC_MODEL_NAME, "version": str(model_version.version),
           "endpoint": ENDPOINT_NAME, "workload": WORKLOAD_TYPE_STR, "ready": ready}
if results:
    best = max(results, key=lambda r: r["gen_tokens_per_s"])
    summary["peak_gen_tokens_per_s"] = best["gen_tokens_per_s"]
    summary["peak_at_concurrency"] = best["concurrency"]
dbutils.notebook.exit(json.dumps(summary))
