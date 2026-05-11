FROM apache/airflow:2.9.0

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

USER airflow

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Yasindu's app code, DAGs, scripts, and pre-trained models
COPY --chown=airflow:root app/     /opt/airflow/app/
COPY --chown=airflow:root dags/    /opt/airflow/dags/
COPY --chown=airflow:root scripts/ /opt/airflow/scripts/
COPY --chown=airflow:root models/  /opt/airflow/models/
COPY --chown=airflow:root config/  /opt/airflow/config/
COPY --chown=airflow:root .env.local /opt/airflow/.env.local

# Also expose as .env so load_dotenv() with no args finds it (used by batch_features.py)
RUN cp /opt/airflow/.env.local /opt/airflow/.env

# Thilaksan's batch features DAG and job (copied in by CI — see docker-build.yml)
COPY --chown=airflow:root batch_dags/ /opt/airflow/dags/
COPY --chown=airflow:root batch_jobs/ /opt/airflow/spark/jobs/

ENV PYTHONPATH=/opt/airflow
