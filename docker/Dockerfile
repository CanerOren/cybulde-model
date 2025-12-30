FROM python:3.10-bullseye

ARG USER_ID
ARG USER_NAME
ENV HOME=/home/${USER_NAME} \
    VIRTUAL_ENV=/home/${USER_NAME}/venv
ENV \
  PYTHONUNBUFFERED=1 \
  DEBIAN_FRONTEND=noninteractive \
  TZ=Europe/Istanbul \
  PATH="/usr/local/gcloud/google-cloud-sdk/bin:${HOME}/.local/bin:${VIRTUAL_ENV}/bin:${PATH}" \
  PYTHONPATH="/app:${PYTHONPATH}" \
  BUILD_POETRY_LOCK="${HOME}/poetry.lock.build"

# ------------------------------------------------------------
# PyTorch CUDA compatibility (quick-success line)
# common-cu113 VM image typically yields ~510.x driver; cu116 wheels work reliably.
# ------------------------------------------------------------
ARG TORCH_CUDA_TAG=cu116
ARG TORCH_VERSION=1.13.1
ARG TORCHVISION_VERSION=0.14.1
ARG TORCHAUDIO_VERSION=0.13.1

RUN apt-get -qq update \
    && apt-get -qq -y install vim gcc curl git build-essential libb64-dev software-properties-common tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get -qq -y clean

RUN curl https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-426.0.0-linux-x86_64.tar.gz > /tmp/google-cloud-sdk.tar.gz
RUN mkdir -p /usr/local/gcloud \
    && tar -C /usr/local/gcloud -xf /tmp/google-cloud-sdk.tar.gz \
    && /usr/local/gcloud/google-cloud-sdk/install.sh --usage-reporting false --command-completion true --bash-completion true --path-update true --quiet

RUN addgroup --system --gid ${USER_ID} ${USER_NAME} \
    && useradd --system -m --no-log-init --home-dir ${HOME} --uid ${USER_ID} --gid ${USER_NAME} --groups ${USER_NAME} ${USER_NAME}

RUN chown -R ${USER_NAME}:${USER_NAME} ${HOME}
RUN mkdir -p /app /mlflow-artifact-store && chown -R ${USER_NAME}:${USER_NAME} /app /tmp /mlflow-artifact-store

RUN curl -sSL https://install.python-poetry.org | python3 - --version 2.2.1

USER ${USER_NAME}

COPY pyproject.toml *.lock /app/
WORKDIR /app

RUN poetry config virtualenvs.create false \
    && python -m venv ${VIRTUAL_ENV} \
    && pip install --upgrade pip setuptools wheel \
    # Preinstall CUDA wheels (helps caching + avoids transient CPU torch)
    && pip install \
        torch==${TORCH_VERSION}+${TORCH_CUDA_TAG} \
        torchvision==${TORCHVISION_VERSION}+${TORCH_CUDA_TAG} \
        torchaudio==${TORCHAUDIO_VERSION}+${TORCH_CUDA_TAG} \
        --extra-index-url https://download.pytorch.org/whl/${TORCH_CUDA_TAG} \
    && poetry install \
    && cp poetry.lock ${BUILD_POETRY_LOCK} \
    # Force CUDA wheels again AFTER poetry (poetry may reinstall CPU torch==1.13.1)
    && pip install --no-deps --force-reinstall \
        torch==${TORCH_VERSION}+${TORCH_CUDA_TAG} \
        torchvision==${TORCHVISION_VERSION}+${TORCH_CUDA_TAG} \
        torchaudio==${TORCHAUDIO_VERSION}+${TORCH_CUDA_TAG} \
        --extra-index-url https://download.pytorch.org/whl/${TORCH_CUDA_TAG} \
    && rm -rf ${HOME}/.cache/*

USER root
COPY ./docker/scripts/* /
RUN chown -R ${USER_NAME} /*.sh && chmod +x /*.sh
USER ${USER_NAME}

COPY . /app/
CMD ["/startup-script.sh"]
