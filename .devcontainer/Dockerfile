# Use official Microsoft Python devcontainer base image
FROM mcr.microsoft.com/devcontainers/python:1-3.11-bullseye

# Prevent Python from writing .pyc files & enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy requirements into container
COPY requirements.txt /tmp/pip-tmp/

# Upgrade pip and pre-install dependencies
RUN pip --disable-pip-version-check --no-cache-dir install -r /tmp/pip-tmp/requirements.txt \
    && rm -rf /tmp/pip-tmp
