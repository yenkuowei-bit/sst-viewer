FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY sst_viewer.py .
COPY data/ ./data/

EXPOSE 8501

CMD ["sh", "-c", "streamlit run sst_viewer.py --server.port $PORT --server.address 0.0.0.0"]
