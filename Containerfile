# =============================================================================
# LiteLLM with fastText language detection (image quay.io/sovereign-selfheal/router).
# -----------------------------------------------------------------------------
# The privacy gate (litellm/privacy_scoring.py) runs in the LiteLLM pod and detects
# the language of each prompt with fastText. fastText is a compiled library plus a
# model file, so a ConfigMap cannot carry it: this image adds both.
# The hook code (*.py) is NOT in the image: the gitops repo mounts it from a
# ConfigMap at /app/litellm, at the same version as this image (see AGENTS.md).
#
# SUPPORT STATUS: LiteLLM is community software, operated by the customer, NOT
# supported by Red Hat. The fastText model lid.176.ftz is licensed CC BY-SA 3.0.
#
# Quay builds this file on every git tag v* (build trigger): see README.md.
# =============================================================================
# litellm-non_root v1.102.0 (multi-arch index), resolved on ghcr.io on 2026-09-25
FROM ghcr.io/berriai/litellm-non_root:v1.102.0@sha256:0fc63424aab32e62185948b7a59180c8ab483cc3e706fcfdf7bf46262297247f

USER 0

# fastText runtime. The base image (Wolfi + uv venv, Python 3.13) has no pip, so
# ensurepip installs it first. fasttext-predict (predict only, MIT) has prebuilt cp313
# wheels; `fasttext` and `fasttext-wheel` would need a C++ compiler.
RUN python -m ensurepip && python -m pip install --no-cache-dir fasttext-predict==0.9.2.4

# fastText language-id model (176 languages, 938,013 bytes). It is downloaded at build
# time and checked by sha256: the running pod has no egress. The path must match
# ner.lang_detect.model_path in the gitops policy privacy-plus.yaml.
# lid.176.ftz, resolved on dl.fbaipublicfiles.com on 2026-09-25
ARG LID_URL=https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz
ARG LID_SHA256=8f3472cfe8738a7b6099e8e999c3cbfae0dcd15696aac7d7738a8039db603e83
RUN python -c "import hashlib, os, sys, urllib.request; \
data = urllib.request.urlopen(sys.argv[1], timeout=120).read(); \
digest = hashlib.sha256(data).hexdigest(); \
digest == sys.argv[2] or sys.exit('lid.176.ftz: sha256 ' + digest + ', expected ' + sys.argv[2]); \
os.makedirs('/opt/models', exist_ok=True); \
open('/opt/models/lid.176.ftz', 'wb').write(data); \
os.chmod('/opt/models/lid.176.ftz', 0o644)" "${LID_URL}" "${LID_SHA256}"

# Back to the non-root user of the base image (65534 / nobody).
USER 65534
