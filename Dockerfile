# Stdlib only, so the image stays small and there is nothing to audit in a
# supply chain. Python 3.14 matches what this was developed against.
FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src

WORKDIR /app

# Copy dependency metadata first so the layer caches independently of source.
COPY pyproject.toml ./
COPY src ./src
COPY tests ./tests

# Run as a non-root user. Nothing here needs privileges.
RUN useradd --create-home --shell /usr/sbin/nologin agent \
    && chown -R agent:agent /app
USER agent

ENTRYPOINT ["python", "-m", "hive.cli"]
CMD ["--population", "16", "--generations", "12", "--episodes", "6"]
