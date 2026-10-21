# Source this file in a cloud shell: source deploy/activate-cloud.sh
export PIP_CACHE_DIR=/workspace/cribwatch-runtime/cache/pip
export XDG_CACHE_HOME=/workspace/cribwatch-runtime/cache
export MPLCONFIGDIR=/workspace/cribwatch-runtime/matplotlib
export LD_LIBRARY_PATH="/workspace/cribwatch-runtime/native/usr/lib/$(dpkg-architecture -qDEB_HOST_MULTIARCH)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
source /workspace/SeniorDesignProject/.venv/bin/activate
