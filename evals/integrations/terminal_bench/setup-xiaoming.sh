#!/usr/bin/env bash
set -euo pipefail

export GIT_TERMINAL_PROMPT=0

if [[ -n "${XIAOMING_PIP_SPEC:-}" ]]; then
  spec="${XIAOMING_PIP_SPEC}"
else
  spec="https://github.com/FogXcG/XiaoMing/archive/refs/heads/master.zip"
fi

ensure_runtime_deps() {
  venv_probe="$(mktemp -d)"
  if python3 -m venv "${venv_probe}" >/dev/null 2>&1 && "${venv_probe}/bin/python" -m pip --version >/dev/null 2>&1; then
    rm -rf "${venv_probe}"
    if [[ "${spec}" != git+* ]] || command -v git >/dev/null 2>&1; then
      return
    fi
  else
    rm -rf "${venv_probe}"
  fi

  packages=(ca-certificates python3-venv)
  if [[ "${spec}" == git+* ]]; then
    packages+=(git)
  fi

  if command -v apt-get >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get update
    apt-get install -y --no-install-recommends "${packages[@]}"
    return
  fi

  echo "Missing python3 venv support or git for ${spec}, and apt-get is unavailable." >&2
  exit 1
}

ensure_runtime_deps

if command -v git >/dev/null 2>&1; then
  git config --global http.version HTTP/1.1
fi

venv_dir="${XIAOMING_VENV_DIR:-/opt/xiaoming-venv}"
python3 -m venv "${venv_dir}"
"${venv_dir}/bin/python" -m pip install --upgrade pip

for attempt in 1 2 3; do
  pip_args=(--no-cache-dir --timeout "${XIAOMING_PIP_TIMEOUT:-60}")
  if [[ "${XIAOMING_PIP_NO_INDEX:-}" == "1" ]]; then
    pip_args+=(--no-index)
  fi
  if [[ -n "${XIAOMING_PIP_FIND_LINKS:-}" ]]; then
    pip_args+=(--find-links "${XIAOMING_PIP_FIND_LINKS}")
  fi
  if [[ -n "${XIAOMING_PIP_TRUSTED_HOST:-}" ]]; then
    for host in ${XIAOMING_PIP_TRUSTED_HOST}; do
      pip_args+=(--trusted-host "${host}")
    done
  fi
  if "${venv_dir}/bin/python" -m pip install "${pip_args[@]}" "${spec}"; then
    break
  fi
  if [[ "${attempt}" == "3" ]]; then
    exit 1
  fi
  sleep $((attempt * 3))
done

ln -sf "${venv_dir}/bin/xiaoming-cli" /usr/local/bin/xiaoming-cli
xiaoming-cli --help >/dev/null
