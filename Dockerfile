FROM python:3.13.7-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HEART_MODEL_DIR=/opt/bundle/model \
    HEART_DISEASE_PROJECT_ROOT=/opt/empty \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 app \
    && useradd --uid 10001 --gid app --no-create-home --shell /usr/sbin/nologin app \
    && mkdir -p /opt/empty

WORKDIR /opt/app
COPY requirements-api.txt requirements-runtime.txt /opt/requirements/
RUN python -m pip install --no-cache-dir -r /opt/requirements/requirements-api.txt

# Build context is generated from a verified model bundle, never from training data.
COPY bundle/ /opt/bundle/
RUN python -m pip install --no-cache-dir --no-deps /opt/bundle/wheels/*.whl \
    && python -m pip check

USER 10001:10001
RUN python -m heart_disease.verify_bundle --bundle /opt/bundle

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=3).close()"]
CMD ["heart-api", "--host", "0.0.0.0", "--port", "8000", "--model-dir", "/opt/bundle/model"]
