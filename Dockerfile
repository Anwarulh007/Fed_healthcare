# Use official Python 3.11 image
FROM python:3.11-slim

# Set environment vars to reduce Python warnings
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory inside container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy app code into container
COPY fl-healthcare-app/ .

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Expose Streamlit default port
EXPOSE 8501

# Command to run your Streamlit app
CMD ["streamlit", "run", "streamlit_app.py", "--server.enableStaticServing=true", "--server.port=8501", "--server.address=0.0.0.0"]

