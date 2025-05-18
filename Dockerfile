FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create responses directory
RUN mkdir -p responses

# Expose the port the MCP server will run on
EXPOSE 8050

# Run the MCP server with SSE transport
ENV TRANSPORT=sse
CMD ["python", "truth.py"]
