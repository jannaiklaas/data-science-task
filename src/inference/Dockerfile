FROM python:3.10

ARG model_name=model_1.pkl
ARG processor_name=processor_1.pkl

WORKDIR /app

COPY ./data/ /app/data
COPY ./outputs/models/${model_name} /app/outputs/models/${model_name}
COPY ./outputs/processors/${processor_name} /app/outputs/models/${processor_name}
COPY ./src/inference /app/src/inference
COPY ./src/train /app/src/train
COPY ./src/text_processor.py /app/src

COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Command to run when the container starts
CMD ["python3", "src/inference/run_inference.py"]