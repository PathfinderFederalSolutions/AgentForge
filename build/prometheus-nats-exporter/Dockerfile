# Multi-stage build for prometheus-nats-exporter (JetStream metrics)
# Local build to bypass upstream registry auth issues.
# VERSION must match an existing upstream tag (list with: git ls-remote --tags https://github.com/nats-io/prometheus-nats-exporter.git)

ARG VERSION=v0.17.3

# Updated to Go 1.24.x because upstream go.mod requires >=1.24.1
# NOTE: golang:1.24.1-alpine may present base CVEs; for production consider a Chainguard/Distroless builder once digest pinned.
FROM golang:1.24.1-alpine AS builder
ARG VERSION
ENV CGO_ENABLED=0 \
    GOOS=linux \
    GOARCH=amd64 \
    # Ensure we do not implicitly fetch a different toolchain later
    GOTOOLCHAIN=local
WORKDIR /workspace
RUN apk add --no-cache git ca-certificates && \
    git clone https://github.com/nats-io/prometheus-nats-exporter.git . && \
    git fetch --tags && \
    (git rev-parse "$VERSION" >/dev/null 2>&1 && git checkout "$VERSION" || echo "Tag $VERSION not found, using default branch") && \
    go mod download && \
    go build -trimpath -ldflags "-s -w -X main.Version=${VERSION}" -o /out/prometheus-nats-exporter .

# (Optional) Produce SBOM (uncomment if syft available / acceptable):
# RUN wget -qO /usr/local/bin/syft https://raw.githubusercontent.com/anchore/syft/main/install.sh && \
#     chmod +x /usr/local/bin/syft && syft packages dir:/workspace -o cyclonedx-json > /out/sbom.json

FROM scratch AS final
# Copy CA certs for HTTPS (if exporter ever needs to scrape TLS endpoints in future flags)
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/ca-certificates.crt
COPY --from=builder /out/prometheus-nats-exporter /prometheus-nats-exporter
# COPY --from=builder /out/sbom.json /sbom.json  # (optional)
USER 65532:65532
EXPOSE 7777
ENTRYPOINT ["/prometheus-nats-exporter"]

# Optional fallback (uncomment to use small alpine instead of scratch):
# FROM alpine:3.20 AS final
# RUN adduser -S -u 65532 exporter
# COPY --from=builder /out/prometheus-nats-exporter /prometheus-nats-exporter
# USER 65532
# EXPOSE 7777
# ENTRYPOINT ["/prometheus-nats-exporter"]

# Usage:
# docker build -t prometheus-nats-exporter:local -f build/prometheus-nats-exporter/Dockerfile .
# kind load docker-image prometheus-nats-exporter:local --name <kindCluster>
# kubectl -n agentforge-staging rollout restart deploy/nats-prometheus-exporter
