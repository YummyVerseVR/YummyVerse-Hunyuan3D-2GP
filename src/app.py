import requests

from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pylognet.client import LoggingClient, LogLevel

from controller import Hunyuan3DController


class UserRequest(BaseModel):
    user_id: str
    prompt: str


class App:
    def __init__(self, config: dict, debug_mode: bool = False, logging: bool = False):
        self.__debug = debug_mode
        self.__config = config
        self.__endpoints = self.__config.get("endpoints", {})
        self.__control_endpoint = self.__endpoints.get(
            "control", "http://localhost:8000"
        )
        self.__logger_endpoint = self.__endpoints.get(
            "logger", "http://logger.local:9000"
        )

        self.__logger = LoggingClient(
            "YummyI23DServer",
            self.__logger_endpoint,
            disable=not logging,
        )

        self.__hunyuan3D_controller = Hunyuan3DController(
            config, self.__logger, self.__debug
        )
        self.__executor = ThreadPoolExecutor()
        self.__router = APIRouter()
        self.__app = FastAPI()

        self.__setup_routes()

    def __del__(self):
        self.__executor.shutdown(wait=True)

    def __setup_routes(self):
        self.__router.add_api_route(
            "/generate",
            self.generate,
            methods=["POST"],
            response_class=JSONResponse,
        )
        self.__router.add_api_route(
            "/ping", self.ping, methods=["GET"], response_class=JSONResponse
        )

    def __post(self, *args, **kwargs):
        if self.__debug:
            self.__logger.log(
                f"[DEBUG] POST Request to {args[0]} with {kwargs}",
                LogLevel.DEBUG,
            )
        else:
            requests.post(*args, **kwargs)

    def __save_model(self, user_id: str, path: str) -> None:
        if not path:
            raise HTTPException(status_code=500, detail="Model path is empty.")

        payload = {"user_id": user_id}
        files = {"file": open(path, "rb")}

        try:
            self.__logger.log(
                f"Saving model for user_id: {user_id} from path: {path}",
                LogLevel.INFO,
            )
            self.__post(
                f"{self.__control_endpoint}/save/model", data=payload, files=files
            )
        except Exception as e:
            self.__logger.log(
                f"Error saving model for user_id: {user_id}: {e}",
                LogLevel.ERROR,
            )
            raise HTTPException(status_code=500, detail="Error saving model.")

    def __save_image(self, user_id: str, image: Image.Image) -> None:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)

        data = {"user_id": user_id}
        file = {"file": ("image.png", buffered, "image/png")}

        try:
            self.__logger.log(
                f"Saving image for user_id: {user_id}",
                LogLevel.INFO,
            )
            self.__post(
                f"{self.__control_endpoint}/save/image",
                files=file,
                data=data,
            )
        except Exception as e:
            self.__logger.log(
                f"Error saving image for user_id: {user_id}: {e}",
                LogLevel.ERROR,
            )
            raise HTTPException(status_code=500, detail="Error saving image.")

    def __generate(self, request: UserRequest) -> None:
        path, image = self.__hunyuan3D_controller.generate(caption=request.prompt)

        if image is not None and isinstance(image, dict):
            image = list(image.values())[0]

        if image is None:
            raise HTTPException(status_code=500, detail="Failed to generate image.")

        self.__save_image(request.user_id, image)
        self.__save_model(request.user_id, path)

    def get_app(self):
        self.__app.include_router(self.__router)
        return self.__app

    # /generate
    async def generate(self, request: UserRequest) -> JSONResponse:
        self.__logger.log(
            f"Received generation request for user_id: {request.user_id}",
            LogLevel.INFO,
        )
        self.__executor.submit(self.__generate, request)
        return JSONResponse(
            {"message": "Model and image generation is submitted."},
            status_code=200,
        )

    # /ping
    async def ping(self):
        return JSONResponse({"message": "pong"}, status_code=200)
