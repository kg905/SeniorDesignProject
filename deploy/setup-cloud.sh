#!/usr/bin/env bash
# Reproduce the tested Debian cloud setup without privileged filesystem writes.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_root=/workspace/cribwatch-runtime
mkdir -p "$runtime_root/cache" "$runtime_root/matplotlib"
export PIP_CACHE_DIR="$runtime_root/cache/pip"
export XDG_CACHE_HOME="$runtime_root/cache"
export MPLCONFIGDIR="$runtime_root/matplotlib"
python3 -m venv "$project_root/.venv"
"$project_root/.venv/bin/python" -m pip install -e "$project_root[vision,notebook,test]"

# MediaPipe's native library links GLES even when CPU inference is selected.
if ldd "$project_root"/.venv/lib/python*/site-packages/mediapipe/tasks/c/libmediapipe.so | rg -q 'libGLESv2.so.2 => not found'; then
    apt_root="$runtime_root/apt"
    mkdir -p "$apt_root/lists/partial" "$apt_root/cache/archives/partial" "$apt_root/downloads" "$runtime_root/native"
    # This helper targets the inspected Debian 13 cloud image; Pi uses its OS packages.
    . /etc/os-release
    if [[ "${ID:-}" != debian || "${VERSION_CODENAME:-}" != trixie ]]; then
        echo 'Install the OS libgles2 package, then rerun setup.' >&2
        exit 1
    fi
    cat > "$apt_root/sources.list" <<'EOF'
deb [signed-by=/usr/share/keyrings/debian-archive-keyring.gpg] https://deb.debian.org/debian trixie main
EOF
    # Set configuration before apt initializes, so host post-install hooks do not run.
    cat > "$apt_root/apt.conf" <<EOF
Dir::Etc::parts "-";
Dir::Etc::main "-";
Dir::Etc::sourcelist "$apt_root/sources.list";
Dir::Etc::sourceparts "-";
Dir::State::lists "$apt_root/lists";
Dir::Cache "$apt_root/cache";
Debug::NoLocking "true";
EOF
    APT_CONFIG="$apt_root/apt.conf" apt-get update
    (cd "$apt_root/downloads" && APT_CONFIG="$apt_root/apt.conf" apt-get download libgles2)
    for package in "$apt_root"/downloads/libgles2_*.deb; do
        dpkg-deb -x "$package" "$runtime_root/native"
    done
fi
export LD_LIBRARY_PATH="$runtime_root/native/usr/lib/$(dpkg-architecture -qDEB_HOST_MULTIARCH)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$project_root"
.venv/bin/cribwatch download-models --output models
.venv/bin/python -m pytest -q
.venv/bin/python - <<'PY'
import numpy as np
from cribwatch.vision import Vision
with Vision('models') as vision:
    result = vision.observe(np.zeros((480, 640, 3), dtype=np.uint8))
    assert not result.pose_ok and result.reason == 'no_pose'
print('Actual MediaPipe inference verified; no infant accuracy claim.')
PY
