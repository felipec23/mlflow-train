FROM python:3.9

# Copy local code to the container image.
WORKDIR /

COPY training.py ./
COPY requirements.txt ./
COPY bert_wrapper.py ./

# Install dependencies.
RUN pip install -r requirements.txt

# Set up the entry point to invoke the trainer.
ENTRYPOINT ["python", "-m", "training.py"]