FROM node:22-alpine AS frontend-build

WORKDIR /build
COPY web/package.json web/package-lock.json ./
RUN npm ci
COPY web/ ./
RUN npm run build

FROM python:3.13-slim AS production

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY core/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt gunicorn

COPY core/app.py core/data.py core/util.py ./core/
COPY data/processed_with_images.csv ./data/processed_with_images.csv
COPY model/ ./model/

COPY --from=frontend-build /build/dist ./core/static/

EXPOSE 8000

CMD ["sh", "-c", "cd /app/core && gunicorn --bind 0.0.0.0:${PORT:-8000} --workers 2 --timeout 120 --preload app:app"]
