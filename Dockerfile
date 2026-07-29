FROM ghcr.io/astral-sh/uv:python3.12-bookworm

WORKDIR /workspace


# Install common system dependencies (OpenCV, git, build tools, etc.)
RUN apt-get update && apt-get install --no-install-recommends -y \
    ca-certificates \
    build-essential

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Create a non-root user (optional but recommended for dev containers)
ARG USERNAME=guest-user
ARG USER_UID=1000
ARG USER_GID=$USER_UID
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y --no-install-recommends sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

USER $USERNAME

# Register the project's .venv as a Jupyter kernel so notebooks see uv-managed packages
RUN uv run python -m ipykernel install --user --name=workspace --display-name="Python (workspace .venv)"

CMD [ "/bin/bash" ]