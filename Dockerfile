# Builds one image that runs the prediction API and the dashboard together.
#
# Stage 1 downloads NASA's data and trains the model. Stage 2 copies only the
# trained model and the files the app reads, so the final image stays small
# and never contains MLflow or the training data.

# ---- Stage 1: train the model ----
FROM python:3.12-slim AS train

# curl and unzip fetch the data; libgomp1 is the OpenMP library LightGBM needs
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl unzip libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt mlflow==2.22.5

# NASA's public C-MAPSS data, the same download as the README's Quick Start
RUN mkdir -p data/raw \
    && curl -fsSL -o cmapss.zip \
       "https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip" \
    && unzip -j cmapss.zip "*/CMAPSSData.zip" \
    && unzip -j CMAPSSData.zip train_FD001.txt test_FD001.txt RUL_FD001.txt -d data/raw \
    && rm cmapss.zip CMAPSSData.zip

COPY src ./src
# Seeded, so every build trains the same model
RUN python -m src.models.train

# ---- Stage 2: run the app ----
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as user 1000, so files belong to that user
RUN useradd --create-home --uid 1000 user
WORKDIR /app

COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt

COPY --chown=user src ./src
COPY --chown=user .streamlit ./.streamlit
COPY --chown=user reports ./reports
COPY --chown=user scripts/start.sh ./scripts/start.sh
COPY --from=train --chown=user /app/models ./models
COPY --from=train --chown=user /app/data/raw/test_FD001.txt /app/data/raw/RUL_FD001.txt ./data/raw/

USER user
ENV HOME=/home/user \
    PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    RUL_API_URL=http://127.0.0.1:8000 \
    STREAMLIT_CLIENT_TOOLBAR_MODE=viewer \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# The port Hugging Face Spaces sends visitors to
EXPOSE 7860
CMD ["bash", "scripts/start.sh"]
