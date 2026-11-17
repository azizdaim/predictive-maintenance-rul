"""Run the prediction API inside the dashboard's own process.

Some hosts, such as Streamlit Community Cloud, run a single Streamlit app and
nothing else. There, the FastAPI service starts in a background thread on
localhost, and the dashboard still talks to it over HTTP exactly as it would to
a separate server. Turn it on by setting RUL_EMBEDDED_API=1.
"""

import os
import threading
import time

import streamlit as st
import uvicorn

from src.api.main import app

ENV_FLAG = "RUL_EMBEDDED_API"


def embedded_api_enabled() -> bool:
    """Return whether this deployment should run the API in-process."""
    return os.environ.get(ENV_FLAG) == "1"


@st.cache_resource(show_spinner="Starting the prediction service...")
def start_embedded_api(port: int = 8000, timeout: float = 10.0) -> uvicorn.Server:
    """Start the API in a background thread, once per server process.

    st.cache_resource makes every visitor share the same server, so it starts
    only once no matter how many sessions open.

    Args:
        port: Localhost port to serve on; matches the dashboard's default API URL.
        timeout: Seconds to wait for the server to accept requests.

    Returns:
        uvicorn.Server: The running server.

    Raises:
        RuntimeError: If the server does not start within the timeout.
    """
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    # A daemon thread stops with the app instead of keeping the process alive
    threading.Thread(target=server.run, name="prediction-api", daemon=True).start()

    deadline = time.monotonic() + timeout
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError(f"The prediction API did not start on port {port}")
        time.sleep(0.05)
    return server
