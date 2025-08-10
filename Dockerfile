FROM python:3.9

WORKDIR /app
COPY . .

# Install app deps + sqlite-web
RUN pip install -r requirements.txt && \
    pip install sqlite-web

# Add a small entrypoint that runs sqlite-web + uvicorn
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose both ports
EXPOSE 8000 8080

CMD ["/app/entrypoint.sh"]