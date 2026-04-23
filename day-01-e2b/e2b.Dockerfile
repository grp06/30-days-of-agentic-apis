FROM e2bdev/base

USER root

ARG NODE_VERSION=22.12.0
ENV PNPM_HOME=/home/user/.local/share/pnpm
ENV PATH=${PNPM_HOME}:${PATH}

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
      ca-certificates \
      curl \
      git \
      gnupg \
      unzip \
      xdg-utils \
    && rm -rf /var/lib/apt/lists/*

RUN arch="$(dpkg --print-architecture)" \
    && case "$arch" in \
      amd64) node_arch="x64" ;; \
      arm64) node_arch="arm64" ;; \
      *) echo "unsupported architecture: $arch" >&2; exit 1 ;; \
    esac \
    && curl -fsSL "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-${node_arch}.tar.gz" \
      | tar -xz -C /usr/local --strip-components=1 \
    && node --version \
    && npm --version

USER user
WORKDIR /home/user

RUN curl -fsSL https://get.pnpm.io/install.sh | env PNPM_VERSION=10.9.0 SHELL=/bin/bash sh - \
    && "${PNPM_HOME}/pnpm" --version

USER root
RUN ln -sf /home/user/.local/share/pnpm/pnpm /usr/local/bin/pnpm \
    && ln -sf /home/user/.local/share/pnpm/pnpx /usr/local/bin/pnpx

USER user
