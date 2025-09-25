import requests

from PIL import Image
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from controller import Hunyuan3DController


class UserRequest(BaseModel):
    user_id: str
    prompt: str


class App:
    def __init__(self, config: dict, debug_mode: bool = False):
        self.__debug = debug_mode
        self.__config = config
        self.__endpoints = self.__config.get("endpoints", {})
        self.__control_endpoint = self.__endpoints.get(
            "control", "http://localhost:8000"
        )
        self.__hunyuan3D_controller = Hunyuan3DController(config)

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
            print(f"[DEBUG] POST Request to {args[0]} with {kwargs}")
        else:
            requests.post(*args, **kwargs)

    async def __save_model(self, user_id: str, path: str) -> None:
        if not path:
            raise HTTPException(status_code=500, detail="Model path is empty.")

        payload = {"user_id": user_id}
        files = {"file": open(path, "rb")}

        try:
            print(f"[INFO] Saving model for user_id: {user_id} from path: {path}")
            self.__post(
                f"{self.__control_endpoint}/save/model", data=payload, files=files
            )
        except Exception as e:
            print(f"[ERROR] Exception while saving model: {e}")
            raise HTTPException(status_code=500, detail="Error saving model.")

    async def __save_image(self, user_id: str, image: Image.Image) -> None:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        buffered.seek(0)

        data = {"user_id": user_id}
        file = {"file": ("image.png", buffered, "image/png")}

        try:
            print(f"[INFO] Saving image for user_id: {user_id}")
            self.__post(
                f"{self.__control_endpoint}/save/image",
                files=file,
                data=data,
            )
        except Exception as e:
            print(f"[ERROR] Exception while saving image: {e}")
            raise HTTPException(status_code=500, detail="Error saving image.")

    def get_app(self):
        self.__app.include_router(self.__router)
        return self.__app

    # /generate
    async def generate(self, request: UserRequest) -> JSONResponse:
        path, image = await self.__hunyuan3D_controller.generate(caption=request.prompt)

        if image is not None and isinstance(image, dict):
            image = list(image.values())[0]

        if image is None:
            raise HTTPException(status_code=500, detail="Failed to generate image.")

        self.__executor.submit(self.__save_image, user_id=request.user_id, image=image)
        self.__executor.submit(self.__save_model, user_id=request.user_id, path=path)

        return JSONResponse(
            {"message": "Model and image generation completed."},
            status_code=200,
        )

    # /ping
    async def ping(self):
        return JSONResponse({"message": "pong"}, status_code=200)
