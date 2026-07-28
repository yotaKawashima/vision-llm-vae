FROM ghcr.io/astral-sh/uv:python3.12-bookworm

WORKDIR /workspace


# Use system Python for uv installs (single global env in container)
ENV UV_SYSTEM_PYTHON=1


# Install common system dependencies (OpenCV, git, build tools, etc.)
RUN apt-get update && apt-get install --no-install-recommends -y \
    ca-certificates \
    build-essential

#COPY pyproject.toml uv.lock ./
#RUN uv sync --frozen

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

CMD [ "/bin/bash" ]