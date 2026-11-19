FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    VALUATION_API_URL=http://127.0.0.1:8000

WORKDIR /app

RUN pip install --no-cache-dir uv==0.11.14

COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY app ./app
COPY data ./data
COPY artifacts ./artifacts
COPY docs ./docs
COPY .streamlit ./.streamlit

RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:${PATH}"

EXPOSE 7860

CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port ${PORT:-7860} --server.headless true"]
