FROM python:3.9

RUN adduser --disabled-password --gecos '' ml-api-user

WORKDIR /app

COPY ./streamlit_app /app/streamlit_app
COPY ./ktp_classification /app/ktp_classification
COPY pyproject.toml /app
COPY run.py /app

RUN pip3 install poetry
RUN poetry config virtualenvs.create false
RUN poetry install

RUN chown -R ml-api-user:ml-api-user ./

EXPOSE 8001

CMD poetry run python run.py