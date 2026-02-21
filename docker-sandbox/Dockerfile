FROM python:3.12-slim

# Create non-root user
RUN useradd -m -s /bin/bash sandbox

# Install dependencies
RUN pip install pytest pytest-timeout numpy

# Switch to non-root user
USER sandbox
WORKDIR /workspace

CMD ["bash"]
