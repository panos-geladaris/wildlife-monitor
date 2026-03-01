#!/usr/bin/env bash
#
# Deploy wildlife-monitor to a Raspberry Pi.
#
# Usage:
#   ./scripts/deploy.sh              # uses PI_HOST env var or defaults to "pi"
#   ./scripts/deploy.sh mypi.local   # explicit host
#   PI_USER=admin ./scripts/deploy.sh # custom user
#
set -euo pipefail

PI_HOST="${1:-${PI_HOST:-pi}}"
PI_USER="${PI_USER:-pi}"
REMOTE_DIR="/home/${PI_USER}/wildlife-monitor"
SERVICE_NAME="wildlife-monitor"

echo "==> Deploying to ${PI_USER}@${PI_HOST}:${REMOTE_DIR}"

# Pull latest code
echo "==> Pulling latest code..."
ssh "${PI_USER}@${PI_HOST}" "cd ${REMOTE_DIR} && git pull"

# Install/update dependencies
echo "==> Installing dependencies..."
ssh "${PI_USER}@${PI_HOST}" "cd ${REMOTE_DIR} && venv/bin/pip install -q -r requirements.txt -r requirements-pi.txt"

# Install/reload systemd service if needed
echo "==> Updating systemd service..."
ssh "${PI_USER}@${PI_HOST}" "sudo cp ${REMOTE_DIR}/scripts/wildlife-monitor.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable ${SERVICE_NAME}"

# Restart the service
echo "==> Restarting ${SERVICE_NAME}..."
ssh "${PI_USER}@${PI_HOST}" "sudo systemctl restart ${SERVICE_NAME}"

# Show status
echo "==> Service status:"
ssh "${PI_USER}@${PI_HOST}" "sudo systemctl status ${SERVICE_NAME} --no-pager" || true

echo "==> Deploy complete!"
