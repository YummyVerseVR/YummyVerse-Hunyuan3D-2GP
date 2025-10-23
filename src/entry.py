import argparse
import json
import uvicorn

from app import App

parser = argparse.ArgumentParser(description="Run the FastAPI application.")
parser.add_argument(
    "-p",
    "--port",
    type=int,
    default=8003,
    help="Port to run the FastAPI application on(default: 8003)",
)
parser.add_argument(
    "-c",
    "--config",
    type=str,
    default="./settings/config.json",
    help="Config file path",
)
parser.add_argument(
    "-d",
    "--debug",
    action="store_true",
    help="Enable debug mode",
)
parser.add_argument(
    "-l",
    "--logging",
    action="store_true",
    help="Enable network logging",
)

args = parser.parse_args()

with open(args.config, "r") as f:
    config = json.load(f)

app = App(config, args.debug, args.logging).get_app()

if __name__ == "__main__":
    uvicorn.run("entry:app", host="0.0.0.0", port=args.port)
