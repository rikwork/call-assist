FROM mcr.microsoft.com/playwright:v1.45.0-jammy

# Install audio utilities and Virtual Frame Buffer
RUN apt-get update && apt-get install -y \
    xvfb \
    pulseaudio \
    dbus-x11 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy package dependencies and install
COPY package.json ./
RUN npm install

# Copy application files
COPY . .

# Set up environment variables for xvfb and pulseaudio
ENV DISPLAY=:99
ENV PULSE_SERVER=unix:/tmp/pulseaudio.socket

# Make entrypoint executable
RUN chmod +x entrypoint.sh

# Run startup script
CMD ["./entrypoint.sh"]
