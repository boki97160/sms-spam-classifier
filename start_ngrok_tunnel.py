
from pyngrok import ngrok
import os
import time

# Install ngrok to the default location if not already present
ngrok.install_ngrok()

# Terminate any existing ngrok tunnels
ngrok.kill()

# Open a tunnel to the Streamlit port (usually 8501)
# The authtoken should already be configured from the previous step
public_url = ngrok.connect(8501)
print(f"Streamlit app is live at: {public_url}")

# Keep the tunnel alive for a while, or until the script is manually stopped
try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("Shutting down ngrok tunnel.")
    ngrok.kill()
