#!/bin/bash
# Start a virtual framebuffer so SFML can open a display in headless environments.
Xvfb :0 -screen 0 1280x1024x24 &
export DISPLAY=:0

# Hand off to the user's command (default: bash)
exec "$@"