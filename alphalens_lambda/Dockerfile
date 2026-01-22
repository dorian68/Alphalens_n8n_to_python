FROM public.ecr.aws/lambda/python:3.12

COPY app.py ${LAMBDA_TASK_ROOT}
COPY requirements.txt .
COPY .env .

# 🔑 Upgrade pip AVANT install
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

CMD ["app.handler"]


FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1

CMD [ "gunicorn", "app:app", \
  "-k", "uvicorn.workers.UvicornWorker", "-w", "4", "-b", "0.0.0.0:9000", \
  "--log-level", "debug", "--access-logfile", "-", "--error-logfile", "-" \
]

