# Build temporalite from source so we avoid GHCR pull
FROM golang:1.22-alpine AS build
RUN apk add --no-cache git
WORKDIR /src
RUN git clone --depth 1 https://github.com/temporalio/temporalite.git .
RUN go build -o /out/temporalite ./cmd/temporalite

FROM alpine:3.19
RUN apk add --no-cache ca-certificates
COPY --from=build /out/temporalite /usr/local/bin/temporalite
ENTRYPOINT ["temporalite"]