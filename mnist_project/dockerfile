# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Create and set the working directory
WORKDIR /app

# Set the PYTHONPATH so that /app (your project root) is on it
ENV PYTHONPATH=/app

# Install system dependencies required for psycopg2
RUN apt-get update && apt-get install -y gcc libpq-dev

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the code
COPY . .

# Expose the port that Streamlit runs on (default is 8501)
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "src/web/app.py", "--server.address=0.0.0.0", "--server.enableCORS", "false"]
