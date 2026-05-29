FROM python:3.11-slim

# Install system media dependencies for video analysis and frame extraction
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . .

# Grant execution permissions to our orchestration script
RUN chmod +x start.sh

# Run the unified stack
CMD ["./start.sh"]