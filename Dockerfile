FROM python:3.12-slim-bookworm@sha256:4766d8b510c428e595d74b9cc5bbb2fae8e26316fffb4adc89908d79aacd58a2 AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_PORT=8501

WORKDIR /app

RUN groupadd --system datalens && useradd --system --gid datalens --create-home datalens

COPY requirements.txt .
RUN python -m pip install \
    --disable-pip-version-check \
    --retries 5 \
    --timeout 120 \
    --requirement requirements.txt

COPY --chown=datalens:datalens app.py ./
COPY --chown=datalens:datalens src ./src
COPY --chown=datalens:datalens sample_data ./sample_data
COPY --chown=datalens:datalens .streamlit ./.streamlit

USER datalens
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3)"

CMD ["streamlit", "run", "app.py"]
