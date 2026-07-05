FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY apps/dashboard/backend/requirements.txt /tmp/dashboard-requirements.txt
RUN pip install --no-cache-dir -r /tmp/dashboard-requirements.txt

COPY src ./src
COPY apps ./apps
COPY README.md ./README.md

EXPOSE 8000

CMD ["uvicorn", "apps.dashboard.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
