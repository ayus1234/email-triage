# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the My Env Environment.
"""

try:
    from openenv.core.env_server.http_server import create_app
except ImportError as e:
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

from fastapi.responses import HTMLResponse

try:
    from ..models import EmailAction, EmailObservation
    from .my_env_environment import MyEnvironment
except (ImportError, ValueError):
    try:
        from models import EmailAction, EmailObservation
        from server.my_env_environment import MyEnvironment
    except ImportError:
        import sys
        import os
        sys.path.append(os.getcwd())
        from models import EmailAction, EmailObservation
        from server.my_env_environment import MyEnvironment


# Create the app with web interface
app = create_app(
    MyEnvironment,
    EmailAction,
    EmailObservation,
    env_name="my_env",
    max_concurrent_envs=1,
)

@app.get("/", response_class=HTMLResponse)
async def root_page():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Email Triage Environment</title>
        <style>
            body { 
                font-family: 'Inter', system-ui, -apple-system, sans-serif; 
                background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
                color: white; 
                display: flex; 
                flex-direction: column; 
                align-items: center; 
                justify-content: center; 
                height: 100vh; 
                margin: 0;
            }
            .container {
                background: rgba(255, 255, 255, 0.05);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.1);
                padding: 3rem;
                border-radius: 24px;
                text-align: center;
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
            }
            h1 { font-size: 2.5rem; margin-bottom: 1rem; background: linear-gradient(to right, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
            p { color: #94a3b8; font-size: 1.1rem; }
            .status { display: inline-flex; align-items: center; background: rgba(34, 197, 94, 0.1); color: #4ade80; padding: 0.5rem 1rem; border-radius: 9999px; font-weight: 500; margin-top: 1rem; }
            .status-dot { width: 8px; height: 8px; background: #22c55e; border-radius: 50%; margin-right: 8px; box-shadow: 0 0 10px #22c55e; }
            .links { margin-top: 2rem; display: flex; gap: 1rem; justify-content: center; }
            .link { color: white; text-decoration: none; background: rgba(255, 255, 255, 0.1); padding: 0.75rem 1.5rem; border-radius: 12px; transition: all 0.2s; border: 1px solid rgba(255, 255, 255, 0.1); }
            .link:hover { background: rgba(255, 255, 255, 0.2); transform: translateY(-2px); }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Email Triage AI Agent</h1>
            <p>The Environment is successfully running and ready for inference.</p>
            <div class="status"><span class="status-dot"></span> System Online</div>
            <div class="links">
                <a href="/health" class="link">Health Check</a>
                <a href="/docs" class="link">API Documentation</a>
            </div>
        </div>
    </body>
    </html>
    """


def main(host: str = "0.0.0.0", port: int = 7860):
    """
    Entry point for direct execution via uv run or python -m.
    """
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main(port=args.port) # main()
