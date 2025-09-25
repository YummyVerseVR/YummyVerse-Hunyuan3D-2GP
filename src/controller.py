import os
import random
import time
import uuid
import numpy
import torch

from PIL import Image
from mmgp import offload, profile_type
from trimesh import Trimesh

from hy3dgen.texgen import Hunyuan3DPaintPipeline
from hy3dgen.shapegen import (
    FaceReducer,
    Hunyuan3DDiTFlowMatchingPipeline,
)
from hy3dgen.shapegen.pipelines import export_to_trimesh
from hy3dgen.rembg import BackgroundRemover
from diffusers.pipelines.auto_pipeline import AutoPipelineForText2Image


class CustomText2ImagePipeline:
    def __init__(self, config: dict):
        torch.set_default_device("cpu")

        self.__config = config.get("text2image", {})
        model_path = self.__config.get(
            "model", "Tencent-Hunyuan/HunyuanDiT-v1.1-Diffusers-Distilled"
        )
        self.__device = self.__config.get("device", "cuda")
        self.__pipe = AutoPipelineForText2Image.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            enable_pag=True,
            pag_applied_layers=["blocks.(16|17|18|19)"],
        )
        self.__pipe.enable_attention_slicing()
        self.__pipe.enable_vae_slicing()
        self.__pipe.enable_sequential_cpu_offload()
        self.__prompt_template = self.__config.get("prompt_template", "{{food}}")
        self.__negative_prompt = self.__config.get("negative_prompt", "")
        self.__inference_steps = int(self.__config.get("inference_steps", 25))
        self.__pag_scale = float(self.__config.get("pag_scale", 1.3))
        self.__width = int(self.__config.get("width", 1024))
        self.__height = int(self.__config.get("height", 1024))

        seed = int(self.__config.get("seed", 0))
        random.seed(seed)
        numpy.random.seed(seed)
        torch.manual_seed(seed)
        os.environ["PL_GLOBAL_SEED"] = str(seed)

    @torch.no_grad()
    def __call__(self, request: str, seed: int = 0) -> Image.Image:
        generator = torch.Generator(device=self.__device)
        generator = generator.manual_seed(int(seed))

        prompt = self.__prompt_template.replace("{{food}}", request)
        out_img = self.__pipe(
            prompt=prompt,
            negative_prompt=self.__negative_prompt,
            num_inference_steps=self.__inference_steps,
            pag_scale=self.__pag_scale,
            width=self.__width,
            height=self.__height,
            generator=generator,
            return_dict=False,
        )[0][0]
        return out_img


class Hunyuan3DController:
    def __init__(self, config: dict):
        self.__config = config.get("hunyuan3d", {})

        self.__seed = int(self.__config.get("seed", 0))
        self.__random_seed = self.__config.get("random_seed", False)
        self.__max_seed = int(self.__config.get("max_seed", 1e7))
        self.__model_path = self.__config.get("model", "tencent/Hunyuan3D-2mini")
        self.__subfolder = self.__config.get("subfolder", "hunyuan3d-dit-v2-mini")
        self.__texgen_model_path = self.__config.get("texgen", "tencent/Hunyuan3D-2")
        self.__device = self.__config.get("device", "cuda")
        self.__mc_algo = self.__config.get("mc_algo", "dmc")
        self.__save_dir = self.__config.get("cache_path", "gradio_cache")
        self.__profile = self.__config.get("profile", "3")
        self.__verbose = self.__config.get("verbose", "1")
        self.__enable_flashvdm = self.__config.get("enable_flashvdm", False)
        self.__disable_tex = self.__config.get("disable_tex", False)
        self.__low_vram_mode = self.__config.get("low_vram_mode", False)
        self.__compile = self.__config.get("compile", False)

        if self.__config.get("mini", False):
            self.__model_path = "tencent/Hunyuan3D-2mini"
            self.__subfolder = "hunyuan3d-dit-v2-mini"
            self.__texgen_model_path = "tencent/Hunyuan3D-2"

        if self.__config.get("mv", False):
            self.__model_path = "tencent/Hunyuan3D-2mv"
            self.__subfolder = "hunyuan3d-dit-v2-mv"
            self.__texgen_model_path = "tencent/Hunyuan3D-2mv"

        if self.__config.get("h2", False):
            self.__model_path = "tencent/Hunyuan3D-2"
            self.__subfolder = "hunyuan3d-dit-v2-0"
            self.__texgen_model_path = "tencent/Hunyuan3D-2"

        if self.__config.get("turbo", False):
            self.__subfolder = self.__subfolder + "-turbo"
            self.__enable_flashvdm = True

        os.makedirs(self.__save_dir, exist_ok=True)

        self.__texturegen = self.__load_texture_generator()
        self.__text2image = CustomText2ImagePipeline(self.__config)
        self.__rmbg_worker = BackgroundRemover()
        self.__i23d_worker = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            self.__model_path,
            subfolder=self.__subfolder,
            torch_dtype=torch.float16,
            use_safetensors=True,
            device=self.__device,
        )
        self.__face_reducer = FaceReducer()

        self.__setup_models()

    def __load_texture_generator(self) -> Hunyuan3DPaintPipeline | None:
        generator = None
        if not self.__disable_tex:
            try:
                generator = Hunyuan3DPaintPipeline.from_pretrained(
                    self.__texgen_model_path
                )
            except Exception as e:
                print(f"[ERROR] Failed to load texture generator: {e}.")
                print(
                    "[ERROR] Please try to install requirements by following README.md"
                )
        return generator

    def __create_kwargs(self) -> dict:
        kwargs = {}
        if self.__profile < 5:
            kwargs["pinnedMemory"] = "i23d_worker/model"
        if self.__profile != 1 and self.__profile != 3:
            kwargs["budgets"] = {"*": 2200}
        return kwargs

    def __setup_models(self) -> None:
        if self.__enable_flashvdm:
            mc_algo = "mc" if self.__device in ["mps", "cpu"] else self.__mc_algo
            self.__i23d_worker.enable_flashvdm(mc_algo=mc_algo)
        if self.__compile:
            self.__i23d_worker.compile()

        property_name = "_execution_device"
        # Get the original class and property
        original_class = type(self.__i23d_worker)
        original_property = getattr(original_class, property_name)

        # Create a custom subclass for this instance
        custom_class = type(f"Custom{original_class.__name__}", (original_class,), {})

        # Create a new property with the new getter but same setter
        new_property = property(lambda _: "cuda", original_property.fset)
        setattr(custom_class, property_name, new_property)

        # Change the instance's class
        self.__i23d_worker.__class__ = custom_class

        pipe = offload.extract_models("i23d_worker", self.__i23d_worker)
        if self.__texturegen is not None:
            pipe.update(offload.extract_models("texturegen_worker", self.__texturegen))
            self.__texturegen.models["multiview_model"].pipeline.vae.use_slicing = True
        if self.__text2image is not None:
            pipe.update(offload.extract_models("t2i_worker", self.__text2image))

        offload.default_verboseLevel = self.__verbose
        kwargs = self.__create_kwargs()
        offload.profile(
            pipe,
            profile_no=profile_type(self.__profile),
            verboseLevel=self.__verbose,
            **kwargs,
        )

        if self.__low_vram_mode:
            torch.cuda.empty_cache()

    def __gen_save_folder(self) -> str:
        # a folder to save the generated files
        folder_name = str(uuid.uuid4())
        save_folder = os.path.join("gradio_cache", folder_name)
        os.makedirs(save_folder, exist_ok=True)
        return save_folder

    def __export(
        self,
        mesh: Trimesh,
        save_folder: str,
        textured: bool = False,
        type: str = "glb",
    ):
        if textured:
            path = os.path.join(save_folder, f"textured_mesh.{type}")
        else:
            path = os.path.join(save_folder, f"white_mesh.{type}")
        if type not in ["glb", "obj"]:
            mesh.export(path)
        else:
            mesh.export(path, include_normals=textured)
        return path

    def __get_seed(self) -> int:
        seed = self.__seed
        if self.__random_seed:
            seed = random.randint(0, self.__max_seed)
        return seed

    async def generate(
        self,
        caption: str,
        steps: int = 50,
    ) -> tuple[str, dict[str, Image.Image | None] | Image.Image | None]:
        seed = int(self.__get_seed())
        octree_resolution = int(self.__config.get("octree_resolution", 256))
        save_folder = self.__gen_save_folder()

        image = self.__text2image(caption)
        if not isinstance(image, Image.Image):
            return "", None

        check_box_rembg = self.__config.get("rembg", False)
        if check_box_rembg or image.mode == "RGB":
            image = self.__rmbg_worker(image.convert("RGB"))

        # image to white model
        start_time = time.time()

        generator = torch.Generator()
        generator = generator.manual_seed(int(seed))
        outputs = self.__i23d_worker(
            image=image,
            num_inference_steps=steps,
            guidance_scale=int(self.__config.get("guidance_scale", 7.5)),
            generator=generator,
            octree_resolution=octree_resolution,
            num_chunks=int(self.__config.get("num_chunks", 200000)),
            output_type="mesh",
        )
        print(f"[INFO] Shape generation takes {time.time() - start_time:.6f} seconds.")

        mesh = export_to_trimesh(outputs)[0]
        mesh = self.__face_reducer(mesh)

        mesh = self.__texturegen(mesh, image) if self.__texturegen else mesh
        path = self.__export(mesh, save_folder, textured=True)

        return path, image
