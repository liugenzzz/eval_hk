#!/usr/bin/env bash

set -euo pipefail

cd "$(dirname "$0")"
#MMLongBench_DOC   MMLB-Doc
#ERQA
#RealWorldQA
#MMMU_DEV_VAL
#MathVision  数学推理需要耗时10个小时评估
#MathVista_MINI
#Video-MME  
#Design2Code
#AIME2025
#LCBenchV6

DATASET="${DATASET:-Video-MME}"
MODEL_NAME="${MODEL_NAME:-qwen3_vl_8b_instruct}"
MODEL_PATH="${MODEL_PATH:-/code/llamafactory/trainspace/model/Qwen3-VL-8B-Instruct}"
WORK_DIR="${WORK_DIR:-./results}"
MODE="${MODE:-all}"
JUDGE_MODEL="${JUDGE_MODEL:-fx-q3-235}"
JUDGE_API_BASE="${JUDGE_API_BASE:-https://mcc-pre.3xmt.com/gateway/ai-service/v1/chat/completions}"
JUDGE_API_KEY="${JUDGE_API_KEY:-sk-2h1RwMjqYcdh6F5Fs5}"
API_NPROC="${API_NPROC:-8}"
REUSE="${REUSE:-1}"

# Prediction file format (xlsx may truncate very long fields like LCBench private tests).
PRED_FORMAT="${PRED_FORMAT:-}"
# Optional local override for Video-MME dataset root.
# Expected structure under this dir: Video-MME.tsv and ./video/*.mp4
VIDEO_MME_LOCAL_PATH="${VIDEO_MME_LOCAL_PATH:-}"

# Online download knobs for HuggingFace
FORCE_ONLINE_DOWNLOAD="${FORCE_ONLINE_DOWNLOAD:-0}"
HF_ENDPOINT="${HF_ENDPOINT:-}"
HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"

# Fast defaults for long-document evaluation (overridable by env).
if [[ "${DATASET}" == "MMLongBench_DOC" ]]; then
  # FAST_MODE levels:
  # 0: close to original behavior
  # 1: balanced speed/quality (default)
  # 2: aggressive speed-first profile
  FAST_MODE="${FAST_MODE:-1}"
  if [[ "${FAST_MODE}" == "2" ]]; then
    export MMLONGBENCH_MAX_PAGES="${MMLONGBENCH_MAX_PAGES:-32}"
    export MMLONGBENCH_PDF_DPI="${MMLONGBENCH_PDF_DPI:-72}"
    export MMLONGBENCH_CONCAT_EDGE="${MMLONGBENCH_CONCAT_EDGE:-768}"
    export MMLONGBENCH_MAX_COLUMN_NUM="${MMLONGBENCH_MAX_COLUMN_NUM:-8}"
    export QWEN3VL_MAX_NEW_TOKENS="${QWEN3VL_MAX_NEW_TOKENS:-1024}"
    export QWEN3VL_MAX_PIXELS="${QWEN3VL_MAX_PIXELS:-602112}"
    export QWEN3VL_MIN_PIXELS="${QWEN3VL_MIN_PIXELS:-100352}"
  elif [[ "${FAST_MODE}" == "1" ]]; then
    export MMLONGBENCH_MAX_PAGES="${MMLONGBENCH_MAX_PAGES:-48}"
    export MMLONGBENCH_PDF_DPI="${MMLONGBENCH_PDF_DPI:-96}"
    export MMLONGBENCH_CONCAT_EDGE="${MMLONGBENCH_CONCAT_EDGE:-896}"
    export MMLONGBENCH_MAX_COLUMN_NUM="${MMLONGBENCH_MAX_COLUMN_NUM:-12}"
    export QWEN3VL_MAX_NEW_TOKENS="${QWEN3VL_MAX_NEW_TOKENS:-2048}"
    export QWEN3VL_MAX_PIXELS="${QWEN3VL_MAX_PIXELS:-1003520}"
    export QWEN3VL_MIN_PIXELS="${QWEN3VL_MIN_PIXELS:-200704}"
  fi
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,4}"
export MASTER_PORT="${MASTER_PORT:-29503}"

# Inference concurrency strategy:
# - MODEL_CONCURRENT=1: prioritize concurrent inference for the tested model.
# - INFER_NPROC / NPROC_PER_NODE: number of inference worker processes.
#   (NPROC_PER_NODE keeps backward compatibility.)
MODEL_CONCURRENT="${MODEL_CONCURRENT:-1}"
INFER_NPROC="${INFER_NPROC:-auto}"
NPROC_PER_NODE="${NPROC_PER_NODE:-${INFER_NPROC}}"
MIN_FREE_MEM_GB="${MIN_FREE_MEM_GB:-24}"

# In low /dev/shm containers, default to gloo for multi-process sync.
export DIST_BACKEND="${DIST_BACKEND:-gloo}"
export DIST_TIMEOUT="${DIST_TIMEOUT:-7200}"

# NCCL fallbacks (useful when DIST_BACKEND=nccl is manually enabled)
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"
export NCCL_CUMEM_HOST_ENABLE="${NCCL_CUMEM_HOST_ENABLE:-0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

# Optional HuggingFace endpoint / transfer settings
if [[ -n "${HF_ENDPOINT}" ]]; then
  export HF_ENDPOINT
fi
export HF_HUB_ENABLE_HF_TRANSFER

# Preflight for Video-MME to avoid distributed cascaded failures in offline environments.
if [[ "${DATASET}" == "Video-MME" ]]; then
  CORRUPTED_HF_REPO="${HOME}/.cache/huggingface/hub/datasets--lmms-lab--Video-MME"
  if [[ -e "${CORRUPTED_HF_REPO}" && ! -d "${CORRUPTED_HF_REPO}" ]]; then
    echo "[Fix] Removing corrupted HuggingFace cache entry: ${CORRUPTED_HF_REPO}"
    rm -f "${CORRUPTED_HF_REPO}"
  fi

  PRECHECK_OUTPUT="$({ python - <<'PY'
import os
import os.path as osp
import socket
import sys
from pathlib import Path

local_override = os.environ.get("VIDEO_MME_LOCAL_PATH", "").strip()
candidates = []
if local_override:
    candidates.append(local_override)

home = os.path.expanduser("~")
hf_root = osp.join(home, ".cache", "huggingface", "hub")
repo_root = osp.join(hf_root, "datasets--lmms-lab--Video-MME")
if osp.isdir(repo_root):
    snapshots = osp.join(repo_root, "snapshots")
    if osp.isdir(snapshots):
        for sub in sorted(Path(snapshots).glob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
            candidates.append(str(sub))
common_candidates = [
    "/code/dataset/Video-MME",
    "/code/data/Video-MME",
]
candidates.extend(common_candidates)

def looks_like_videomme(path: str) -> bool:
    return osp.isfile(osp.join(path, "Video-MME.tsv")) and osp.isdir(osp.join(path, "video"))

seen = set()
for p in candidates:
    if not p:
        continue
    rp = osp.realpath(p)
    if rp in seen:
        continue
    seen.add(rp)
    if looks_like_videomme(rp):
        print(f"LOCAL_OK|{rp}")
        sys.exit(0)

try:
    socket.create_connection(("huggingface.co", 443), timeout=3).close()
    print("ONLINE_OK|")
except OSError:
    print("OFFLINE_NO_LOCAL|")
PY
  } )"
  PRECHECK_STATUS="${PRECHECK_OUTPUT%%|*}"
  PRECHECK_PATH="${PRECHECK_OUTPUT#*|}"

  if [[ "${PRECHECK_STATUS}" == "LOCAL_OK" ]]; then
    export VIDEO_MME_LOCAL_PATH="${PRECHECK_PATH}"
    echo "[Precheck] Video-MME local dataset detected: ${VIDEO_MME_LOCAL_PATH}"
  elif [[ "${PRECHECK_STATUS}" == "ONLINE_OK" ]]; then
    echo "[Precheck] No local Video-MME dataset found, will download from HuggingFace."
  else
    if [[ "${FORCE_ONLINE_DOWNLOAD}" == "1" ]]; then
      echo "[Warn] Precheck cannot reach huggingface.co, but FORCE_ONLINE_DOWNLOAD=1 is set."
      echo "[Warn] Will continue and try snapshot_download directly (proxy/HF_ENDPOINT may still make it work)."
    else
      echo "[Error] Video-MME requires either internet access to huggingface.co or a local dataset copy."
      echo "[Error] Current environment is offline and no valid local Video-MME dataset was found."
      echo "[Hint] Set VIDEO_MME_LOCAL_PATH to a directory containing Video-MME.tsv and video/*.mp4"
      echo "[Hint] Or set FORCE_ONLINE_DOWNLOAD=1 and configure HTTPS_PROXY/HF_ENDPOINT for online download."
      exit 1
    fi
  fi
fi

if [[ "${DATASET}" == "LCBenchV6" ]]; then
  if [[ -z "${PRED_FORMAT}" ]]; then
    export PRED_FORMAT="tsv"
    echo "[Precheck] DATASET=LCBenchV6, default PRED_FORMAT=tsv to avoid xlsx cell truncation."
  else
    export PRED_FORMAT
  fi
fi

IFS=',' read -r -a GPU_LIST <<< "${CUDA_VISIBLE_DEVICES}"
GPU_COUNT="${#GPU_LIST[@]}"

if [[ "${GPU_COUNT}" -lt 1 ]]; then
  echo "[Error] No visible GPUs from CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  exit 1
fi

if [[ "${NPROC_PER_NODE}" == "auto" ]]; then
  if [[ "${MODEL_CONCURRENT}" == "1" ]]; then
    NPROC_PER_NODE="${GPU_COUNT}"
    echo "[AutoNPROC] MODEL_CONCURRENT=1, use full visible GPUs: nproc_per_node=${NPROC_PER_NODE}"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    AUTO_PICK_RESULT="$({ CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" MIN_FREE_MEM_GB="${MIN_FREE_MEM_GB}" python - <<'PY'
import os
import subprocess

visible = [int(x) for x in os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',') if x.strip()]
min_free = float(os.environ.get('MIN_FREE_MEM_GB', '24'))

if not visible:
    print('1|unknown|0')
    raise SystemExit(0)

try:
    out = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
        text=True,
    )
    free_map = {}
    for line in out.strip().splitlines():
        idx_s, free_s = [x.strip() for x in line.split(',')]
        free_map[int(idx_s)] = float(free_s) / 1024.0

    selected = [free_map[i] for i in visible if i in free_map]
    if not selected:
        print('1|unknown|0')
        raise SystemExit(0)

    fit = sum(1 for x in selected if x >= min_free)
    nproc = max(1, fit)
    min_visible_free = min(selected)
    print(f'{nproc}|{min_visible_free:.1f}|{fit}')
except Exception:
    print('1|unknown|0')
PY
    } )"
    IFS='|' read -r NPROC_PER_NODE AUTO_MIN_FREE_GB AUTO_FIT_COUNT <<< "${AUTO_PICK_RESULT}"
    if [[ "${AUTO_MIN_FREE_GB}" == "unknown" ]]; then
      NPROC_PER_NODE="${GPU_COUNT}"
      echo "[AutoNPROC] nvidia-smi query failed, fallback to full visible GPUs: nproc_per_node=${NPROC_PER_NODE}"
    else
      echo "[AutoNPROC] min_visible_free_mem=${AUTO_MIN_FREE_GB}GB, min_free_mem_threshold=${MIN_FREE_MEM_GB}GB, fit_gpu_count=${AUTO_FIT_COUNT}, nproc_per_node=${NPROC_PER_NODE}"
    fi
  else
    NPROC_PER_NODE="${GPU_COUNT}"
    echo "[AutoNPROC] nvidia-smi not found, fallback nproc_per_node=${NPROC_PER_NODE}"
  fi
fi

if [[ ! "${NPROC_PER_NODE}" =~ ^[0-9]+$ ]] || [[ "${NPROC_PER_NODE}" -lt 1 ]]; then
  echo "[Warn] NPROC_PER_NODE=${NPROC_PER_NODE} is invalid, fallback to 1"
  NPROC_PER_NODE=1
fi

if [[ "${NPROC_PER_NODE}" -gt "${GPU_COUNT}" ]]; then
  echo "[Warn] NPROC_PER_NODE=${NPROC_PER_NODE} > visible GPU count=${GPU_COUNT}, fallback to ${GPU_COUNT}"
  NPROC_PER_NODE="${GPU_COUNT}"
fi

CMD_ARGS=(
  --data "${DATASET}"
  --model "${MODEL_NAME}"
  --model-path "${MODEL_PATH}"
  --work-dir "${WORK_DIR}"
  --mode "${MODE}"
  --judge-model "${JUDGE_MODEL}"
  --judge-api-base "${JUDGE_API_BASE}"
  --judge-api-key "${JUDGE_API_KEY}"
  --api-nproc "${API_NPROC}"
)

if [[ "${REUSE}" == "1" ]]; then
  CMD_ARGS+=(--reuse)
fi

echo "[Launch] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[Launch] NPROC_PER_NODE=${NPROC_PER_NODE}, MASTER_PORT=${MASTER_PORT}, DIST_BACKEND=${DIST_BACKEND}, MODEL_CONCURRENT=${MODEL_CONCURRENT}"
echo "[Launch] NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE}, NCCL_CUMEM_HOST_ENABLE=${NCCL_CUMEM_HOST_ENABLE}"
if [[ "${DATASET}" == "MMLongBench_DOC" ]]; then
  echo "[FastMode] FAST_MODE=${FAST_MODE:-0}, REUSE=${REUSE}, MAX_PAGES=${MMLONGBENCH_MAX_PAGES:-na}, PDF_DPI=${MMLONGBENCH_PDF_DPI:-na}, CONCAT_EDGE=${MMLONGBENCH_CONCAT_EDGE:-na}, MAX_COLUMN_NUM=${MMLONGBENCH_MAX_COLUMN_NUM:-na}, QWEN3VL_MAX_NEW_TOKENS=${QWEN3VL_MAX_NEW_TOKENS:-na}, QWEN3VL_MIN_PIXELS=${QWEN3VL_MIN_PIXELS:-na}, QWEN3VL_MAX_PIXELS=${QWEN3VL_MAX_PIXELS:-na}"
fi

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
  torchrun --nnodes=1 --nproc-per-node "${NPROC_PER_NODE}" --master-port "${MASTER_PORT}" run.py "${CMD_ARGS[@]}"
else
  python run.py "${CMD_ARGS[@]}"
fi
