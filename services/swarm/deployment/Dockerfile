# CPU worker image
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# tini for clean signal handling
RUN apt-get update && apt-get install -y --no-install-recommends tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# install "light" deps 
COPY requirements.txt ./
RUN pip install -r requirements.txt

# copy only the worker's code into the image
COPY . /app/

# if the worker needs shared libs/configs from the repo root, copy them here too:
# COPY libs/common /app/libs/common

# default start command (override at deploy time if your module path is different)
ENTRYPOINT ["tini","--"]
CMD ["python","-m","services.swarm_worker.main"]