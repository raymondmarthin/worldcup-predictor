FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Model must already be trained (models/model.joblib present) before building
# the image, or run `python -m src.train` inside the container once at build
# time if you want a self-contained image:
# RUN python -m src.clean && python -m src.features && python -m src.train

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
