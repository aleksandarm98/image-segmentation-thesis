FROM pytorch/pytorch:latest

RUN apt-get update && apt-get install -y \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*


WORKDIR .

COPY requirements.txt .
RUN pip install --no-cache-dir --timeout=1500 -r requirements.txt

ENV AWS_ACCESS_KEY_ID $AWS_ACCESS_KEY_ID
ENV AWS_SECRET_ACCESS_KEY $AWS_SECRET_ACCESS_KEY
ENV AWS_REGION_NAME $AWS_REGION_NAME
ENV AWS_BUCKET_NAME $AWS_BUCKET_NAME
ENV COMET_SECRET_KEY $COMET_SECRET_KEY
ENV CONFIG_PATH ${CONFIG_PATH}

COPY . .


RUN ls -la

CMD ["python", "app/app.py"]
