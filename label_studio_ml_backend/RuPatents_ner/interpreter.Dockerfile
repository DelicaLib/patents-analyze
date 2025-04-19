# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS python-base
ARG TEST_ENV

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_CACHE_DIR=/.cache \
    WORKERS=1 \
    THREADS=8

# Update the base OS
RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    set -eux; \
    apt-get update; \
    apt-get upgrade -y; \
    apt install --no-install-recommends -y  \
        git; \
    apt-get autoremove -y

RUN python -m venv .venv

COPY ./.venv/lib/python3.12/site-packages /app/.venv/lib/python3.12/site-packages

COPY requirements.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    if /app/.venv/bin/pip show "ru_patents_ner" > /dev/null 2>&1; then \
        echo "ru_patents_ner already exist"; \
    else \
        . /app/.venv/bin/activate && \
        pip install -r requirements.txt; \
    fi

# install base requirements
COPY requirements-base.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    . /app/.venv/bin/activate && \
    pip install -r requirements-base.txt

# install custom requirements
COPY requirements-model.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    . /app/.venv/bin/activate && \
    pip install -r requirements-model.txt

# install test requirements if needed
COPY requirements-test.txt .
# build only when TEST_ENV="true"
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    if [ "$TEST_ENV" = "true" ]; then \
      . /app/.venv/bin/activate && \
      pip install -r requirements-test.txt; \
    fi

ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

CMD ["/app/.venv/bin/python"]