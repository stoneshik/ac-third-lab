FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
  PIP_NO_CACHE_DIR=off \
  POETRY_VERSION=1.6.1

WORKDIR /code
COPY src/poetry.lock src/pyproject.toml /code

RUN pip install --no-cache-dir --progress-bar off "poetry==$POETRY_VERSION"

RUN poetry config virtualenvs.create false \
  && poetry install --no-cache --no-interaction --no-ansi