# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` on vLLM (Custom LLM Serving)
# MAGIC
# MAGIC Single, self-contained notebook that **tests → logs → deploys → queries → benchmarks** the 4-bit
# MAGIC bitsandbytes Llama-3.1-8B checkpoint on Databricks Model Serving via **Custom LLM Serving**
# MAGIC (`task=llm/v1/chat` + vLLM `entrypoint`, registered as an **express deployment** with
# MAGIC `env_pack="databricks_model_serving"`). Follows the official
# MAGIC [Serve custom LLMs](https://learn.microsoft.com/azure/databricks/machine-learning/model-serving/serve-custom-llms)
# MAGIC starter notebooks, adapted for a bnb-4bit model.
# MAGIC
# MAGIC ### What it does
# MAGIC 1. Downloads the HF weights to local disk (unsloth re-uploads are **ungated** — no HF token needed).
# MAGIC 2. Starts vLLM in the notebook on the A10 and sends a test request (catches bad flags / OOM in minutes).
# MAGIC 3. Logs an MLflow `ChatModel` placeholder + the vLLM `entrypoint` command as metadata.
# MAGIC 4. Registers to Unity Catalog with `env_pack="databricks_model_serving"` (packs this notebook's env).
# MAGIC 5. Creates/updates a **GPU_LARGE (A100 80 GB)** serving endpoint.
# MAGIC 6. Queries it (OpenAI-compatible chat) and runs a **light tokens/s benchmark** across concurrency.
# MAGIC
# MAGIC ### Prerequisites
# MAGIC - **Custom LLM Serving** enabled: Admin Settings → Previews → *Custom LLM Serving* → On.
# MAGIC - Workspace in a **US region** with **AI Runtime** (serverless GPU).
# MAGIC - Run on **Serverless GPU (A10)** with environment **Standard v6** (vLLM is installed below, as in the
# MAGIC   official `qwen3-5-4b-standard` starter). The model must be logged from GPU
# MAGIC   compute — a model logged from CPU packs CPU deps and the GPU endpoint fails with
# MAGIC   *"entrypoint is only supported with express deployments"*.
# MAGIC
# MAGIC > ⚠ First A100 endpoint build is slow (often 20–40+ min) and can exceed the SDK `create_and_wait`
# MAGIC > timeout while still `DEPLOYMENT_CREATING` — not an error. We poll.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install vLLM + bitsandbytes
# MAGIC Pins match the official Standard v6 starter notebook, plus bitsandbytes for the 4-bit weights.
# MAGIC Installed in the notebook so `env_pack` packs them into the serving environment.

# COMMAND ----------

# MAGIC %pip install -q vllm==0.24.0 transformers==5.8.1 mlflow==3.13.0 openai==2.37.0 hf_transfer==0.1.9 "bitsandbytes>=0.48.1"
# MAGIC %pip install -q --force-reinstall --no-deps opencv-python-headless==4.12.0.88
# MAGIC %restart_python

# COMMAND ----------

import ctypes, importlib.util, os

# Serverless v6 reports a client.6.x runtime; vLLM comes from the %pip cell above.
runtime = os.environ.get("DATABRICKS_RUNTIME_VERSION", "")
if not runtime.startswith("client.6.") or not importlib.util.find_spec("vllm"):
    raise RuntimeError(f"Select Serverless GPU with environment Standard v6 (found {runtime or 'none'}).")
try:
    ctypes.CDLL("libcuda.so.1")              # NVIDIA driver is only present on GPU compute
except OSError:
    raise RuntimeError("No GPU found. Attach Serverless GPU compute with an A10.") from None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Config (widgets)

# COMMAND ----------

dbutils.widgets.text("catalog", "main", "UC catalog")
dbutils.widgets.text("schema", "llm_serving", "UC schema")
dbutils.widgets.text("model_name", "llama31_8b_bnb4", "UC registered model name")
dbutils.widgets.text("endpoint", "llama31-8b-bnb4", "Serving endpoint name")
dbutils.widgets.text("hf_repo", "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit", "HuggingFace repo")
dbutils.widgets.dropdown("workload_type", "GPU_LARGE", ["GPU_LARGE", "GPU_LARGE_RTX", "GPU_SMALL"], "Serving GPU (A100 / RTX PRO 6000 / T4)")
dbutils.widgets.dropdown("dtype", "auto", ["auto", "bfloat16", "float16"], "vLLM dtype (auto: fp16 on T4, else bf16)")
dbutils.widgets.dropdown("smoke_test", "true", ["true", "false"], "Local vLLM smoke test")
dbutils.widgets.dropdown("scale_to_zero", "false", ["true", "false"], "Scale-to-zero")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
MODEL_NAME = dbutils.widgets.get("model_name")
ENDPOINT_NAME = dbutils.widgets.get("endpoint")
HF_REPO = dbutils.widgets.get("hf_repo")
WORKLOAD_TYPE_STR = dbutils.widgets.get("workload_type")
SMOKE_TEST = dbutils.widgets.get("smoke_test") == "true"
SCALE_TO_ZERO = dbutils.widgets.get("scale_to_zero") == "true"

UC_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"
SERVED_MODEL_NAME = MODEL_NAME            # vLLM --served-model-name
ARTIFACTS_PATH = "model"                  # local weights dir; logged as the artifacts folder `--model` names

# vLLM / serving tuning
# The entrypoint (and so the dtype) is baked into the model version, so it must suit the *serving* GPU:
# A100/RTX are Ampere+ -> bf16; T4 (GPU_SMALL) is Turing with no bf16 -> fp16.
DTYPE = dbutils.widgets.get("dtype")
if DTYPE == "auto":
    DTYPE = "float16" if WORKLOAD_TYPE_STR == "GPU_SMALL" else "bfloat16"
MAX_MODEL_LEN = 8192
GPU_MEMORY_UTILIZATION = 0.85
LOCAL_PORT = 3080                         # serverless GPU notebooks allow ports 3000-3999
SERVING_PORT = 8080                       # Model Serving requires 8080

print(f"uc_model={UC_MODEL_NAME} | endpoint={ENDPOINT_NAME} | workload={WORKLOAD_TYPE_STR} | dtype={DTYPE}")
print(f"hf_repo={HF_REPO} | smoke_test={SMOKE_TEST} | scale_to_zero={SCALE_TO_ZERO}")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import AlreadyExists, ResourceAlreadyExists

# Create the UC model up front so a permission problem fails in seconds, not after download + test.
try:
    WorkspaceClient().registered_models.create(catalog_name=CATALOG, schema_name=SCHEMA, name=MODEL_NAME)
except (AlreadyExists, ResourceAlreadyExists):
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Download weights to local disk
# MAGIC `/Workspace` can't hold large weight files; download to a temp dir on the node.

# COMMAND ----------

import os, shutil, tempfile
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.chdir(tempfile.mkdtemp())                         # /Workspace can't hold large weights

from huggingface_hub import snapshot_download

# Download straight into the artifacts dir (unsloth bnb-4bit is ~5.6 GB, ungated -> no HF token).
if not os.path.exists(f"{ARTIFACTS_PATH}/config.json"):
    snapshot_download(repo_id=HF_REPO, local_dir=ARTIFACTS_PATH)
shutil.rmtree(f"{ARTIFACTS_PATH}/.cache", ignore_errors=True)   # HF download metadata; don't log it
print("local weights:", sorted(os.listdir(ARTIFACTS_PATH))[:8], "...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. vLLM chat entrypoint
# MAGIC bitsandbytes pre-quantized checkpoint → tell vLLM to load/serve it in bnb format.

# COMMAND ----------

def entrypoint(port: int) -> str:
    args = [
        "VLLM_USE_FLASHINFER_SAMPLER=0",       # FlashInfer's sampler fails at startup on Serverless GPU
        "python", "-u", "-m", "vllm.entrypoints.openai.api_server",
        "--model", ARTIFACTS_PATH,
        "--served-model-name", SERVED_MODEL_NAME,
        "--quantization", "bitsandbytes",      # 4-bit bnb weights
        "--load-format", "bitsandbytes",
        "--host", "0.0.0.0",
        "--port", str(port),
        "--dtype", DTYPE,
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION),
    ]
    return " ".join(args)

print(entrypoint(SERVING_PORT))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Local smoke test (standalone vLLM OpenAI server on the A10)
# MAGIC Verifies the entrypoint and bnb weights load before we pay for an endpoint build. Hits
# MAGIC `/invocations`, the same route Model Serving calls.

# COMMAND ----------

import signal, subprocess, requests, time

def log_tail():
    return open("vllm.log").read()[-5000:]

def is_ready():
    try:
        return requests.get(f"http://localhost:{LOCAL_PORT}/health", timeout=2).ok
    except requests.ConnectionError:
        return False

if SMOKE_TEST:
    with open("vllm.log", "w") as log:
        server = subprocess.Popen(entrypoint(LOCAL_PORT), shell=True, stdout=log, stderr=log,
                                  start_new_session=True)
    try:
        deadline = time.time() + 30 * 60
        while not is_ready():
            if server.poll() is not None:
                raise RuntimeError(f"vLLM exited with code {server.returncode}. Log tail:\n{log_tail()}")
            if time.time() > deadline:
                raise RuntimeError(f"vLLM did not start within 30 min. Log tail:\n{log_tail()}")
            time.sleep(5)
        r = requests.post(
            f"http://localhost:{LOCAL_PORT}/invocations",
            json={"messages": [{"role": "user", "content": "In one sentence, what is Databricks?"}],
                  "max_tokens": 64, "temperature": 0.2},
            timeout=180)
        if not r.ok:
            raise RuntimeError(f"vLLM test request failed with HTTP {r.status_code}: {r.text[:500]}")
        print(r.json()["choices"][0]["message"]["content"][:400])
    finally:
        if server.poll() is None:                    # free the GPU before logging
            os.killpg(server.pid, signal.SIGTERM)
else:
    print("smoke_test=false -> skipping local vLLM load")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Log (ChatModel placeholder + entrypoint metadata) → register as express deployment

# COMMAND ----------

import mlflow
from mlflow.pyfunc.model import ChatModel, ChatCompletionResponse

# Upload the weights with the Databricks SDK and longer timeouts: MLflow's default uploader often fails on multi-GB files.
os.environ["MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC"] = "true"
os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = "600"
os.environ["MLFLOW_ARTIFACT_UPLOAD_DOWNLOAD_TIMEOUT"] = "600"

class LLMModel(ChatModel):
    # Placeholder: serving runs the entrypoint, not predict. ChatModel gives the chat signature UC needs.
    def predict(self, context, messages, params):
        return ChatCompletionResponse.from_dict({"choices": []})

with mlflow.start_run(run_name=f"deploy_{MODEL_NAME}"):
    model_info = mlflow.pyfunc.log_model(
        name=SERVED_MODEL_NAME,
        python_model=LLMModel(),
        artifacts={"model_dir": ARTIFACTS_PATH},
        metadata={"task": "llm/v1/chat", "entrypoint": entrypoint(SERVING_PORT)},
    )
print("logged:", model_info.model_uri)

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")

# env_pack packs this notebook's GPU env (vLLM + bitsandbytes) with the model -> express deployment,
# which is required for entrypoint models.
model_version = mlflow.register_model(model_info.model_uri, UC_MODEL_NAME, env_pack="databricks_model_serving")
print("registered:", UC_MODEL_NAME, "v", model_version.version)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Deploy the A100 (GPU_LARGE) endpoint
# MAGIC Custom LLM endpoints don't autoscale and docs recommend scale-to-zero off; if it's requested and
# MAGIC rejected we fall back to fixed concurrency. First build can exceed the SDK wait while still
# MAGIC `DEPLOYMENT_CREATING` (not an error).

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput, ServedEntityInput, ServingModelWorkloadType)
from datetime import timedelta

# Azure custom LLM GPUs: GPU_SMALL = 1x T4 16 GB, GPU_LARGE = 1x A100 80 GB,
# GPU_LARGE_RTX = 1x RTX PRO 6000 96 GB (only in some regions)
WORKLOAD_TYPE = ServingModelWorkloadType(WORKLOAD_TYPE_STR)

w = WorkspaceClient()
existing = [e.name for e in w.serving_endpoints.list()]

def deploy(scale_to_zero: bool):
    entities = [ServedEntityInput(
        entity_name=UC_MODEL_NAME, entity_version=str(model_version.version),
        workload_type=WORKLOAD_TYPE, workload_size="Small",
        scale_to_zero_enabled=scale_to_zero,
    )]
    if ENDPOINT_NAME in existing:
        w.serving_endpoints.update_config_and_wait(
            name=ENDPOINT_NAME, served_entities=entities, timeout=timedelta(minutes=50))
    else:
        w.serving_endpoints.create_and_wait(
            name=ENDPOINT_NAME,
            config=EndpointCoreConfigInput(name=ENDPOINT_NAME, served_entities=entities),
            timeout=timedelta(minutes=50))

scale_to_zero = SCALE_TO_ZERO
try:
    deploy(scale_to_zero)
    print(f"endpoint READY: {ENDPOINT_NAME} (scale_to_zero={scale_to_zero})")
except TimeoutError:
    print(f"still deploying (poll below/UI): {ENDPOINT_NAME} state={w.serving_endpoints.get(ENDPOINT_NAME).state}")
except Exception as e:
    msg = str(e).lower()
    if scale_to_zero and ("autoscal" in msg or "scale_to_zero" in msg or "scale to zero" in msg
                          or "entrypoint" in msg):
        print(f"scale-to-zero rejected -> retrying with fixed concurrency.\n  ({e})")
        scale_to_zero = False
        try:
            deploy(scale_to_zero)
            print(f"endpoint READY: {ENDPOINT_NAME} (scale_to_zero=False, fallback)")
        except TimeoutError:
            print(f"still deploying (poll below/UI): {ENDPOINT_NAME} state={w.serving_endpoints.get(ENDPOINT_NAME).state}")
    else:
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Wait for READY (decoupled from create — polls up to ~45 min)

# COMMAND ----------

deadline = time.time() + 45 * 60
ready = False
while time.time() < deadline:
    s = w.serving_endpoints.get(ENDPOINT_NAME).state
    ready_v = getattr(s.ready, "value", str(s.ready)) if s and s.ready else "?"
    cu_v = getattr(s.config_update, "value", str(s.config_update)) if s and s.config_update else "?"
    print(time.strftime("%T"), "ready=", ready_v, "config_update=", cu_v)
    if ready_v == "READY":
        ready = True; break
    time.sleep(30)
print("endpoint READY:", ready)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Query the endpoint (OpenAI-compatible chat)

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
# MAGIC ## 8. Light tokens/s benchmark
# MAGIC Measures **generated (completion) tokens/s** against the live endpoint, sweeping client concurrency.
# MAGIC Each request generates up to `GEN_TOKENS`; throughput = total completion tokens / wall time.

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
    dt = time.perf_counter() - t0
    return dt, r.usage.completion_tokens

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
    bench_df = pd.DataFrame(results).sort_values("gen_tokens_per_s", ascending=False)
    display(bench_df)
else:
    print("skipping benchmark -- endpoint not READY")

# COMMAND ----------

import json
summary = {"uc_model": UC_MODEL_NAME, "version": str(model_version.version),
           "endpoint": ENDPOINT_NAME, "workload": WORKLOAD_TYPE_STR,
           "ready": ready, "scale_to_zero": scale_to_zero}
if results:
    best = max(results, key=lambda r: r["gen_tokens_per_s"])
    summary["peak_gen_tokens_per_s"] = best["gen_tokens_per_s"]
    summary["peak_at_concurrency"] = best["concurrency"]
    print(f"peak {best['gen_tokens_per_s']:,.0f} gen-tokens/s at concurrency={best['concurrency']}; "
          f"p50 {best['p50_ms']}ms p95 {best['p95_ms']}ms")
dbutils.notebook.exit(json.dumps(summary))
