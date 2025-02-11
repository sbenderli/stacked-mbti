# Use a minimal Python image based on Alpine Linux
FROM python:3.9-alpine

# Prevent Python from writing pyc files to disc and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Install build dependencies needed to compile some Python packages
RUN apk update && apk add --no-cache gcc musl-dev

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Upgrade pip and install the Python dependencies from requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the main application file and prompts file into the container
COPY mbti_test.py .
COPY prompts.json .

# Expose port 5000 so the container can communicate with the host
EXPOSE 5000

# Command to run the Flask app when the container starts
CMD ["python", "mbti_test.py"]
