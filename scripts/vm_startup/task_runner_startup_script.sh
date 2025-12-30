#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

export GCP_LOGGING_ENABLED="TRUE"
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCHELASTIC_ERROR_FILE=/tmp/torch_elastic_error.json
export CUDA_LAUNCH_BLOCKING=1
export NCCL_DEBUG=INFO

# -----------------------------------------------------------------------------
# NVIDIA driver install bazen modülleri hemen yüklemiyor.
# Bu yüzden ilk boot'ta driver kur -> tek sefer reboot -> ikinci boot'ta devam et.
# Reboot loop'u engellemek için sentinel dosyası kullanıyoruz.
# -----------------------------------------------------------------------------
NVIDIA_REBOOT_SENTINEL="/var/run/nvidia_driver_rebooted"

INSTANCE_GROUP_NAME=$(curl --silent --fail http://metadata.google.internal/computeMetadata/v1/instance/attributes/instance_group_name -H "Metadata-Flavor: Google")
DOCKER_IMAGE=$(curl --silent --fail http://metadata.google.internal/computeMetadata/v1/instance/attributes/docker_image -H "Metadata-Flavor: Google")
ZONE=$(curl --silent http://metadata.google.internal/computeMetadata/v1/instance/attributes/zone -H "Metadata-Flavor: Google")
PYTHON_HASH_SEED=$(curl --silent --fail http://metadata.google.internal/computeMetadata/v1/instance/attributes/python_hash_seed -H "Metadata-Flavor: Google" || echo "42")
MLFLOW_TRACKING_URI=$(curl --silent http://metadata.google.internal/computeMetadata/v1/instance/attributes/mlflow_tracking_uri -H "Metadata-Flavor: Google")
NODE_COUNT=$(curl --silent http://metadata.google.internal/computeMetadata/v1/instance/attributes/node_count -H "Metadata-Flavor: Google")
DISKS=$(curl --silent http://metadata.google.internal/computeMetadata/v1/instance/attributes/disks -H "Metadata-Flavor: Google")

INSTANCE_GROUP_NAME=$(echo ${INSTANCE_GROUP_NAME} | tr '[:upper:]' '[:lower:]')

echo -e "TRAINING: instance group name: ${INSTANCE_GROUP_NAME}, docker image: ${DOCKER_IMAGE}, node count: ${NODE_COUNT}, python hash seed: ${PYTHON_HASH_SEED}"

echo "============================ Installing Nvidia Drivers ============================"
apt-get update || echo "apt-get update failed (Debian buster EOL?), continuing..."
/opt/deeplearning/install-driver.sh 2>&1 | tee /var/log/install-driver.log || true
echo "install-driver last lines:"
tail -n 80 /var/log/install-driver.log || true

# GPU cihazı var mı? (attach edilmemişse burada net fail verelim)
echo "============================ GPU sanity checks (pre-reboot) ============================"
lspci | grep -i nvidia || { echo "NO NVIDIA GPU DETECTED (accelerator attach edilmemis olabilir)"; exit 1; }

# Eğer daha önce reboot yapmadıysak ve nvidia-smi hâlâ yoksa: tek sefer reboot.
if [[ ! -f "${NVIDIA_REBOOT_SENTINEL}" ]]; then
  echo "============================ Checking NVIDIA readiness (may reboot once) ============================"
  if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not ready after driver install. Rebooting ONCE to load kernel modules..."
    touch "${NVIDIA_REBOOT_SENTINEL}"
    sync
    reboot
  fi
fi

# Buraya geldiysek ya reboot sonrası 2. boot’tayız ya da reboot gerekmemiştir.
echo "============================ GPU sanity checks (post-reboot) ============================"
echo "Waiting for nvidia-smi to become available..."
for i in {1..60}; do
  if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
    echo "NVIDIA driver ready"
    nvidia-smi
    break
  fi
  echo "Waiting for nvidia-smi... ($i/60)"
  sleep 5
done

# Son kez garanti
nvidia-smi || { echo "Driver never became ready (nvidia-smi failed)"; exit 1; }

# NVML library var mı? (senin önceki hatan: libnvidia-ml.so.1)
ldconfig -p | grep -q "libnvidia-ml.so.1" || { echo "NVML missing: libnvidia-ml.so.1 not found"; ldconfig -p | grep -i nvidia || true; exit 1; }

echo "============================ Downloading Docker Image ============================"
gcloud auth configure-docker --quiet europe-west4-docker.pkg.dev
time docker pull "${DOCKER_IMAGE}"

echo "============================ Running Training Container (DEBUG MODE: python) ============================"
# IMPORTANT: VIRTUAL_ENV is an env var inside the container image, not on the VM host.
# So we must run via bash -lc and use absolute paths INSIDE the container.
docker run --init --rm --gpus all --ipc host --user root --hostname "$(hostname)" --privileged \
  --log-driver=gcplogs \
  -e PYTHONHASHSEED="${PYTHON_HASH_SEED}" \
  -e MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI}" \
  -e TOKENIZERS_PARALLELISM=false \
  "${DOCKER_IMAGE}" \
  bash -lc 'set -euxo pipefail; ls -la /app | head; ls -la /app/cybulde/run_tasks.py; /home/caner/venv/bin/python /app/cybulde/run_tasks.py' \
  || echo '============================ TRAINING: job failed ============================'

echo '============================ Cleaning Up ============================'
gcloud compute instance-groups managed delete --quiet "${INSTANCE_GROUP_NAME}" --zone "${ZONE}"
