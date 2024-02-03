FROM python:3.10

WORKDIR /app

COPY ./data/ /app/data
COPY ./src/train /app/src/train
COPY ./src/text_processor.py /app/src

COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Command to run when the container starts
CMD ["python3", "src/train/train.py"]
