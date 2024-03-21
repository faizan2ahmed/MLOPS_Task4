FROM python:3.8-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
CMD ["python", "Main.py"]
EXPOSE 5000
ENV NAME World
CMD ["python", "Main.py"]
