FROM python:3.9
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
COPY models/ models/
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]