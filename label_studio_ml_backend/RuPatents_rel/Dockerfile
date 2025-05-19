# syntax=docker/dockerfile:1
ARG PYTHON_VERSION=3.12

FROM python:${PYTHON_VERSION}-slim AS python-base
ARG TEST_ENV
ARG MODEL_NER
ARG MODEL_REL
ARG CLICKHOUSE_MIGRATE

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=${PORT:-9090} \
    PIP_CACHE_DIR=/.cache \
    WORKERS=1 \
    THREADS=8 \
    MODEL_NER=${MODEL_NER} \
    MODEL_REL=${MODEL_REL} \
    CLICKHOUSE_MIGRATE=${CLICKHOUSE_MIGRATE}

# Update the base OS
RUN --mount=type=cache,target="/var/cache/apt",sharing=locked \
    --mount=type=cache,target="/var/lib/apt/lists",sharing=locked \
    set -eux; \
    apt-get update; \
    apt-get upgrade -y; \
    apt install --no-install-recommends -y  \
        git; \
    apt-get install -y libgl1; \
    apt-get install -y libglib2.0-0; \
    apt-get autoremove -y;

# install base requirements
COPY requirements-base.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements-base.txt

# install custom requirements
COPY requirements.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip install -r requirements.txt

RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    if [ "$MODEL_NER" = "ru_patents_ner_tiny" ]; then \
      pip install "ru_patents_ner_tiny @ https://huggingface.co/Delicalib/ru_patents_ner-tiny/resolve/main/ru_patents_ner_tiny-any-py3-none-any.whl"; \
    elif [ "$MODEL_NER" = "ru_patents_ner" ]; then \
      pip install "ru_patents_ner @ https://huggingface.co/Delicalib/ru_patents_ner/resolve/main/ru_patents_ner-any-py3-none-any.whl"; \
    else \
      echo "$MODEL_NER"; \
    fi

RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    if [ "$MODEL_REL" = "ru_patents_rel_tiny" ]; then \
      pip install "ru_patents_rel_tiny @ https://huggingface.co/Delicalib/ru_patents_rel_tiny/resolve/main/ru_patents_rel_tiny-any-py3-none-any.whl"; \
    elif [ "$MODEL_REL" = "ru_patents_rel" ]; then \
      pip install "ru_patents_rel @ https://huggingface.co/Delicalib/ru_patents_rel/resolve/main/ru_patents_rel-any-py3-none-any.whl"; \
    else \
      echo "$MODEL_REL"; \
    fi

RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    if ["$CLICKHOUSE_MIGRATE" = "true"]; then \
      pip install "clickhouse_driver"; \
      python /app/clickhouse_migrations/main.py; \
    fi


# install test requirements if needed
COPY requirements-test.txt .
# build only when TEST_ENV="true"
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    if [ "$TEST_ENV" = "true" ]; then \
      pip install -r requirements-test.txt; \
    fi

COPY .. .

EXPOSE 9090

CMD gunicorn --preload --reload --bind :$PORT --workers $WORKERS --threads $THREADS --timeout 0 RuPatents_rel._wsgi:app
