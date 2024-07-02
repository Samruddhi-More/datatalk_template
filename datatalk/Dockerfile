# syntax=docker/dockerfile:1
FROM python:3.10

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 

WORKDIR /code

ENV PYTHONPATH=/code:$PYTHONPATH

ENV TZ="Asia/Kolkata"

COPY requirements.txt /code/

RUN apt update -y \
    && apt upgrade -y \
    && pip install -r requirements.txt

COPY . /code/