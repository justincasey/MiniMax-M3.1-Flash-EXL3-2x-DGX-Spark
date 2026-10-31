# Serving image for MiniMax-M3.1-Flash EXL3 SAGE 2.15bpw at
# TP=2 on two DGX Sparks. Built by `./m31.sh build` on rank 0 from the engine
# commits pinned in config/pins.sh (fetched and staged by scripts/prepare_src.sh);
# the Docker build itself downloads nothing.
#
#   base : vllm/vllm-openai:minimax-m31-flash, pinned by the arm64 manifest
#          list digest in config/pins.sh.
#   +    : exllamav3 at EXL3_REV (vcruz305's fork with the act-3 batched-MoE
#          fix) plus patches/exllamav3/0001-0003 (coop MoE decode, dense
#          decode GEMV), with the plugin's aarch64 CPU-path patch.
#   +    : vllm-exl3 at PLUGIN_REV (vcruz305) plus patches/0001-0019: TP=2
#          enablement, weight snapshots, prefill and decode kernels.
#          PATCH_SHA256 covers both series.

ARG BASE_IMAGE_DIGEST=sha256:86f582dae25ea92802353727164ae4eb03b37b18baf9119b805ffaec2a7ef07c
FROM vllm/vllm-openai@${BASE_IMAGE_DIGEST}

ARG BASE_IMAGE_DIGEST
ARG EXL3_REV=unknown
ARG PLUGIN_REV=unknown
ARG PATCH_SHA256=unknown

# GB10 unifies CPU and GPU memory: a larger nvcc fan-out OOMs the host.
ENV CUDA_HOME=/usr/local/cuda \
    TORCH_CUDA_ARCH_LIST=12.1a \
    MAX_JOBS=8 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY exllamav3-src /opt/exllamav3-local
COPY vllm-exl3-src /opt/vllm-exl3-local
COPY PINS /opt/m31-pins.txt

RUN set -eux; \
    python3 /opt/vllm-exl3-local/tools/patch_exllamav3_aarch64.py \
        /opt/exllamav3-local/exllamav3/exllamav3_ext; \
    CUSPARSE_DIR="$(find /usr/local/lib/python3*/dist-packages/nvidia -name cusparse.h -exec dirname {} \; 2>/dev/null | head -1)"; \
    [ -n "$CUSPARSE_DIR" ] || { echo "ERROR: cusparse.h not found under NVIDIA Python packages" >&2; exit 2; }; \
    export CPATH="$CUSPARSE_DIR${CPATH:+:$CPATH}"; \
    python3 -m pip install --no-cache-dir --no-build-isolation --no-deps /opt/exllamav3-local; \
    export EXL3_EXT_INCLUDE=/opt/exllamav3-local/exllamav3/exllamav3_ext; \
    python3 -m pip install --no-cache-dir --no-build-isolation --no-deps /opt/vllm-exl3-local; \
    test -f /opt/vllm-exl3-local/src/vllm_exl3/exl3.py; \
    grep -q "_lm_head_full_vocab" /opt/vllm-exl3-local/src/vllm_exl3/exl3.py; \
    grep -q "_indexer_rows_for_rank" /opt/vllm-exl3-local/src/vllm_exl3/exl3.py; \
    grep -qF 'getattr(layer, "use_ep", None) is False' /opt/vllm-exl3-local/src/vllm_exl3/exl3.py; \
    grep -q "def _narrow_lm_head" /opt/vllm-exl3-local/src/vllm_exl3/exl3.py; \
    grep -q "elif moe_act == MOE_ACT_SWIGLU_OAI:" /opt/vllm-exl3-local/src/vllm_exl3/exl3.py; \
    test -f /opt/vllm-exl3-local/src/vllm_exl3/weight_snapshot.py; \
    grep -q EXL3_FAT_GEMM_MAX_K /opt/vllm-exl3-local/csrc/bindings.cpp; \
    grep -q exl3_fat_gemm_grouped /opt/vllm-exl3-local/csrc/bindings.cpp; \
    grep -q "def _apply_wide_prefill" /opt/vllm-exl3-local/src/vllm_exl3/exl3.py; \
    grep -q "def _mixedk_m_tile" /opt/vllm-exl3-local/src/vllm_exl3/exl3.py; \
    grep -q "_COOP_SPLIT_MAX_TOKENS" /opt/vllm-exl3-local/src/vllm_exl3/exl3.py; \
    test -f /opt/vllm-exl3-local/src/vllm_exl3/sparse_attn_tiled.py; \
    grep -q FAT_THRESHOLD_PREFILL /opt/vllm-exl3-local/src/vllm_exl3/exl3.py

LABEL local.m31.base-digest="${BASE_IMAGE_DIGEST}" \
      local.m31.exl3="${EXL3_REV}" \
      local.m31.plugin="${PLUGIN_REV}" \
      local.m31.patch-sha256="${PATCH_SHA256}"

ENTRYPOINT []
