import yaml
import os, subprocess
import cv2
import typer
import numpy as np
import time
import multiprocessing as mp
from multiprocessing import Value
import asyncio
from furiosa.server.model import FuriosaRTModel, FuriosaRTModelConfig
from utils.postprocess import ObjDetPostProcess
from utils.preprocess import YOLOPreProcessor, letterbox


class WarboyRunner:
    def __init__(self, model, input_path, runner_info, device, idx):
        self.model = FuriosaRTModel(
            FuriosaRTModelConfig(
                name="borde",
                model=model,
                worker_num=8,
                npu_device = device,
            )
        )
        self.preprocessor = YOLOPreProcessor()
        self.postprocessor = ObjDetPostProcess("yolov5l", runner_info)
        self.input_shape = runner_info["input_shape"]
        self.result_path = "result"
        self.proc_idx = idx
        self.input_path = input_path
        self.img_names = os.listdir(str(input_path))

    async def load(self):
        await self.model.load()

    async def process(self, img_name):
        img_path = os.path.join(self.input_path, img_name)
        img_name = img_name.split('.')[0]
        img = cv2.imread(img_path)
        input_, preproc_params = self.preprocessor(img, new_shape=self.input_shape)
        output = await self.model.predict(input_) 
        out = self.postprocessor(output, preproc_params, img)
        cv2.imwrite(os.path.join(self.result_path, img_name+".bmp"),out)
        return

    async def run(self):
        await asyncio.gather(*(self.task(worker_id) for worker_id in range(8)))

    async def task(self, worker_id):
        for i, img_name in enumerate(self.img_names):
            if i % 8 == worker_id:
                if i % 2 == self.proc_idx:
                    await self.process(img_name)

app = typer.Typer(pretty_exceptions_show_locals=False)

def get_params_from_cfg(cfg):
    model_config = open(cfg)
    model_info = yaml.load(model_config, Loader=yaml.FullLoader)
    model_config.close()

    return model_info["model_info"], model_info["runner_info"]

async def startup(runner):
    await runner.load()

async def run_model(runner):
    await runner.run()

def run(runner, inference_time):
    asyncio.run(startup(runner))
    t3 = time.time()
    asyncio.run(run_model(runner))
    t4 = time.time()
    inference_time.value = t4-t3

@app.command()
def main(cfg, input_path):
    model_info, runner_info = get_params_from_cfg(cfg)

    # model_path = "enf_models/borde_model_single_2.enf"
    model_path = "borde_model_i8.onnx"
    
    iterations = 10
    fps_list = []
    for i in range(iterations):
        input_datas = os.listdir(input_path)
        inference_times = [Value('d', 0.0) for i in range(2)]
        runners = [WarboyRunner(model_path, input_path, runner_info, "npu0pe"+str(i), i) for i in range(2)]
        warboy_processes = [mp.Process(target = run, args = (runners[i], inference_times[i],)) for i in range(2)]

        for proc in warboy_processes:
            proc.start()

        for proc in warboy_processes:
            proc.join()
        
        inference_time = max(inference_times[0].value, inference_times[1].value)
        fps = len(input_datas)/inference_time
        print(f"Number of images: {len(input_datas)} FPS: {fps}")
        fps_list.append(fps)

    print(f'Resolution: {runner_info["input_shape"]} Iteration count: {iterations} Avg. FPS: {sum(fps_list)/iterations}')

if __name__ == "__main__":
    app()
