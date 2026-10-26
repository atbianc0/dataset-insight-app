FROM python:3.15-slim-bookworm@sha256:575f8bc12ec26542e0678bc6b60e385218248019d433157ec327264a78c3a853 AS runtime

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
COPY --chown=datalens:datalens LICENSE DATASETS.md ./

USER datalens
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health', timeout=3)"

CMD ["streamlit", "run", "app.py"]
