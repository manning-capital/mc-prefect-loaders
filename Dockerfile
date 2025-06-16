# Use a Python image with uv pre-installed
FROM python:3.12.11-slim-bookworm

# Install the project into `/app`
WORKDIR /app

# Install uv tool
RUN pip install uv

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Copy dependency files first for better layer caching
COPY pyproject.toml uv.lock ./

# Install the project's dependencies using the lockfile and settings
RUN uv sync --locked --no-install-project --no-dev

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
COPY . .
RUN uv sync --locked --no-dev

# Place executables in the environment at the front of the path
ENV PATH="/app/.venv/bin:$PATH"