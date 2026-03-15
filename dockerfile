#base image
FROM python:3.11-slim

#wordir
WORKDIR /app

#copy
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /app

EXPOSE 8501


CMD ["streamlit", "run", "streamlit_frontend_database.py", "--server.port=8501", "--server.address=0.0.0.0"]