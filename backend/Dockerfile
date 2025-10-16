
FROM node:18-slim

# Install Python and pip
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-venv \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment
RUN python3 -m venv /opt/venv

# Activate virtual environment and install Python dependencies
ENV PATH="/opt/venv/bin:$PATH"
COPY ./models/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

WORKDIR /app

COPY ./backend/package*.json ./

RUN npm install

# Copy models directory
COPY ./models ./models

COPY . .

EXPOSE 3000

CMD ["npm", "start"]
