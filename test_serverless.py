import base64
import io
import json
import os
import shlex
import subprocess as sp
from PIL import Image

from mmdeploy.apis.utils import build_task_processor
from mmdeploy.utils import get_input_shape, load_config

import torch

def handler(context, event):
    context.logger.info("Running CGNet model!")
    deploy_cfg = './configs/mmseg/segmentation_torchscript.py'
    model_cfg = './cgnet_tversky_config.py'
    device = 'cuda:0'
    backend_model = ['./end2end.pt']
    image = './tmp.png'

    # save image to tmp.png
    data = event.body
    buf = io.BytesIO(base64.b64decode(data["image"]))
    Image.open(buf).save("./tmp.png")

    # read deploy_cfg and model_cfg
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)

    # build task and backend model
    task_processor = build_task_processor(model_cfg, deploy_cfg, device)
    model = task_processor.build_backend_model(backend_model)

    # process input image
    input_shape = get_input_shape(deploy_cfg)
    model_inputs, _ = task_processor.create_input(image, input_shape)

    # do model inference
    with torch.no_grad():
        result = model.test_step(model_inputs)

    results = []

    data = result[0].pred_sem_seg.data
    # get tensor of positions of predictions for vocal cords
    nonzero = torch.nonzero(result[0].pred_sem_seg.data - 1)

    # remove first column
    nonzero = torch.index_select(nonzero, 1, torch.arange(1, nonzero.size()[1], device="cuda:0"))
    results.append({
        "label": "vocal cord",
        "points": nonzero.tolist(),
        "type": "polygon",
    })

    return context.Response(body=json.dumps(results), headers={},
        content_type='application/json', status_code=200)