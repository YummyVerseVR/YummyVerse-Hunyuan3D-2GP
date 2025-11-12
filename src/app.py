import asyncio
import requests

from PIL import Image
from io import BytesIO
from fastapi import FastAPI, APIRouter, HTTPException, Form, UploadFile, File
from fastapi.responses import JSONResponse
from pylognet.client import LoggingClient, LogLevel
from concurrent.futures import ThreadPoolExecutor

from controller import Hunyuan3DController


class App:
    def __init__(self, config: dict, debug_mode: bool = False, logging: bool = False):
        self.__debug = debug_mode
        self.__config = config
        self.__endpoints = self.__config.get("endpoints", {})
        self.__control_endpoint = self.__endpoints.get(
            "control", "http://localhost:8000"
        )
        self.__logger_endpoint = self.__endpoints.get("logger", "http://localhost:9000")
        self.__mcc_endpoint = self.__endpoints.get("mcc", "http://localhost:7000")

        self.__logger = LoggingClient(
            "YummyI23DServer",
            self.__logger_endpoint,
            disable=not logging,
        )

        self.__queue = asyncio.Queue()
        self.__hunyuan3D_controller = Hunyuan3DController(config, self.__logger)
        self.__router = APIRouter()
        self.__app = FastAPI()

        asyncio.create_task(self.__worker())

        self.__executor = ThreadPoolExecutor()

        self.__setup_routes()
        self.__logger.log(
            "Model Generation Server initialized successfully",
            LogLevel.INFO,
        )

    def __setup_routes(self):
        self.__router.add_api_route(
            "/generate",
            self.generate,
            methods=["POST"],
            response_class=JSONResponse,
        )
        self.__router.add_api_route(
            "/process",
            self.process,
            methods=["GET"],
            response_class=JSONResponse,
        )
        self.__router.add_api_route(
            "/ping", self.ping, methods=["GET"], response_class=JSONResponse
        )

    async def __worker(self):
        while True:
            user_id, byte = await self.__queue.get()
            try:
                self.__generate(user_id, byte)
            except Exception as e:
                self.__logger.log(
                    f"Error processing generation for user ID {user_id}: {e}",
                    LogLevel.ERROR,
                )
            finally:
                self.__queue.task_done()

    def __save_model(self, user_id: str, path: str) -> None:
        if not path:
            raise HTTPException(status_code=500, detail="Model path is empty.")

        try:
            payload = {"user_id": user_id}
            files = {"file": open(path, "rb")}

            response = requests.post(
                f"{self.__control_endpoint}/save/model", data=payload, files=files
            )
            if response.status_code != 200:
                raise HTTPException(
                    status_code=500, detail="Failed to save model to database."
                )

            self.__logger.log(
                f"Saved model of user ID: {user_id}",
                LogLevel.INFO,
            )
        except Exception as e:
            self.__logger.log(
                f"Failed to save model of user ID: {user_id}: {e}",
                LogLevel.ERROR,
            )
            raise HTTPException(status_code=500, detail="Error saving model.")

    def __save_process(self, user_id: str, path: str) -> None:
        if not path:
            raise HTTPException(status_code=500, detail="Model path is empty.")

        try:
            payload = {"user_id": user_id}
            files = {"file": open(path, "rb")}

            response = requests.post(
                f"{self.__mcc_endpoint}/upload/model", data=payload, files=files
            )
            if response.status_code != 200:
                raise HTTPException(
                    status_code=500, detail="Failed to save process to database."
                )

            self.__logger.log(
                f"Saved process of user ID: {user_id}",
                LogLevel.INFO,
            )
        except Exception as e:
            self.__logger.log(
                f"Failed to save process of user ID: {user_id}: {e}",
                LogLevel.ERROR,
            )
            raise HTTPException(status_code=500, detail="Error saving process.")

    def __generate(self, user_id: str, byte: bytes) -> None:
        image = Image.open(BytesIO(byte)).convert("RGB")
        path, _ = self.__hunyuan3D_controller.generate(image=image)

        self.__save_model(user_id=user_id, path=path)

    def __process(self, user_id: str, byte: bytes) -> None:
        image = Image.open(BytesIO(byte)).convert("RGB")
        path, _ = self.__hunyuan3D_controller.generate(image=image)

        self.__save_process(user_id=user_id, path=path)

    def get_app(self):
        self.__app.include_router(self.__router)
        return self.__app

    # /generate
    async def generate(
        self, user_id: str = Form(...), file: UploadFile = File(...)
    ) -> JSONResponse:
        byte = await file.read()
        await self.__queue.put((user_id, byte))
        return JSONResponse(
            {"message": "Model generation is submitted."},
            status_code=200,
        )

    # /process
    async def process(
        self, user_id: str = Form(...), file: UploadFile = File(...)
    ) -> JSONResponse:
        byte = await file.read()
        self.__executor.submit(self.__process, user_id, byte)
        return JSONResponse(
            {"message": "Model generation is submitted."},
            status_code=200,
        )

    # /ping
    async def ping(self):
        return JSONResponse({"message": "pong"}, status_code=200)
