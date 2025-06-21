# Use a lightweight Python base image
FROM python:3.10-slim

# Install OS-level dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the entire app code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Launch Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
