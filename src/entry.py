import argparse
import json
import uvicorn

from app import App

CONFIG_PATH = "settings/meta.json"
parser = argparse.ArgumentParser(description="Run the FastAPI application.")
parser.add_argument(
    "-p",
    "--port",
    type=int,
    default=8003,
    help="Port to run the FastAPI application on(default: 8003)",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Enable debug mode",
)
args = parser.parse_args()

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

app = App(config, args.debug).get_app()

if __name__ == "__main__":
    uvicorn.run("entry:app", host="0.0.0.0", port=args.port)
