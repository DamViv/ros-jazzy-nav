#!/bin/bash
set -e  # Exit on error

ABSOLUTE_DOCKER=$(pwd)

# Stop and remove all containers defined in compose file (if any)
echo "Stopping and removing containers..."
docker compose down

# Remove leftover container if exists
docker rm -f ros-jazzy-gz-simulator || true
docker rm -f ros-jazzy-nav || true

# Create ccache directory if it doesn't exist
[ -d ~/.ccache ] || mkdir -p ~/.ccache

# Allow X server connections from local root (needed for GUI apps in container)
if command -v xhost >/dev/null; then
    xhost +local:root
else
    echo "WARNING xhost not found. GUI applications may not work."
fi

# Start containers in background (detached)
echo "Launching containers..."
docker compose up -d --remove-orphans #--build

echo "All containers are up and running!"
docker compose ps

# Wait a bit for containers to initialize
sleep 3

# Launch Terminator from each container
echo "Launching GUI terminals..."
docker exec -e DISPLAY=$DISPLAY -u root ros-jazzy-nav terminator --title="ROS2 Navigation" --layout=navLayout &
