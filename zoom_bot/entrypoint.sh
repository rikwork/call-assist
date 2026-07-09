#!/bin/bash

# Start dbus daemon
export $(dbus-launch)

# Start Xvfb (Virtual Frame Buffer)
echo "Starting Xvfb..."
Xvfb :99 -screen 0 1280x720x24 -ac +extension GLX +render -noreset &
sleep 2

# Start PulseAudio daemon
echo "Starting PulseAudio..."
pulseaudio -D --exit-idle-time=-1 --system=false --fail=true
sleep 2

# Load virtual null sink for capturing page audio output
echo "Setting up virtual null sink..."
pactl load-module module-null-sink sink_name=VirtualSink sink_properties=device.description=VirtualSink
pactl set-default-sink VirtualSink

# Run Node.js bot script passing command line arguments
echo "Launching Zoom Bot..."
exec node zoom_bot.js "$@"
