import torch
import torch_neuron
from torchvision import models

NEURON_CORES = 1
SIZE = 512
IN_TYPE = torch.float32

model = torch.hub.load('yolov5',
        'custom',
        path='yolov5/weights/yolov5l6_v2.2_2048x2048_30.05.2022_conf0.546.pt',
        source='local',
        force_reload=True)  # local repo

# https://github.com/ultralytics/yolov5/issues/7739
for m in model.modules():
    if hasattr(m, 'inplace'):
        m.inplace = False

fake_image = torch.zeros([1, 3, SIZE, SIZE], dtype=IN_TYPE)

try:
    torch.neuron.analyze_model(model, example_inputs=[fake_image])
except Exception:
    torch.neuron.analyze_model(model, example_inputs=[fake_image])

model_neuron = torch.neuron.trace(model, 
                                example_inputs=[fake_image]
                                )

## Export to saved model
model_neuron.save(f"yolov5/weights/yolov5l6_v2.2_2048x2048_30.05.2022_conf0.546_aws_neuron_size_{SIZE}.pt")
