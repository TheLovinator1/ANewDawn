FROM golang:alpine

# Git is required for go mod download
RUN apk update && apk add --no-cache git ca-certificates

ENV USER=anewdawn
ENV UID=10001

# Create anewdawn user
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    "${USER}"

# Set the working directory
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# Download dependencies
RUN go get -d -v

# Build the binary
RUN GOOS=linux GOARCH=amd64 go build -ldflags="-w -s" -o  /usr/local/bin/anewdawn

FROM scratch

COPY --from=0 /etc/passwd /etc/passwd
COPY --from=0 /etc/group /etc/group
COPY --from=0 /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/

# Copy the binary from the first stage
COPY --from=0 /usr/local/bin/anewdawn /usr/local/bin/anewdawn

# Use an unprivileged user.
USER anewdawn:anewdawn

# Command to run the executable
ENTRYPOINT ["/usr/local/bin/anewdawn"]
