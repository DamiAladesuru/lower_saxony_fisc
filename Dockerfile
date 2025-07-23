FROM python:3

# Create and set working directory
WORKDIR /lower_saxony_fisc

# Copy everything in the current directory (where the Dockerfile is) into the container
COPY . .

# Install dependencies
RUN pip install -r requirements.txt