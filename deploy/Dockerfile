# The bot has no dependencies, so the stock slim image is the whole runtime.
FROM python:3.12-slim

WORKDIR /app
COPY smcbot/ ./smcbot/
COPY scripts/ ./scripts/
COPY tests/ ./tests/
COPY README.md pyproject.toml ./

# Results and downloaded candles live on a mounted volume, not in the image.
VOLUME ["/app/data", "/app/runs"]

ENV PYTHONUNBUFFERED=1

# Fail the build if the engine is broken -- a container that starts but
# miscalculates is worse than one that refuses to build.
RUN python -m unittest tests.test_engines tests.test_ml tests.test_fetch -v

ENTRYPOINT ["python", "-m", "smcbot"]
CMD ["backtest", "--synthetic", "--days", "60"]
