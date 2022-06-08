import torch
import torch_neuron
from torchvision import models

model = torch.hub.load('yolov5',
        'custom',
        path='yolov5/weights/yolov5l6_v2.2_2048x2048_30.05.2022_conf0.546.pt',
        source='local',
        force_reload=True)  # local repo

fake_image = torch.zeros([1, 3, 2048, 2048], dtype=torch.float32)
#fake_image = (torch.rand(3), torch.rand(3))
try:
    torch.neuron.analyze_model(model, example_inputs=[fake_image])
except Exception:
    torch.neuron.analyze_model(model, example_inputs=[fake_image])

model_neuron = torch.neuron.trace(model, 
                                example_inputs=[fake_image])

## Export to saved model
model_neuron.save("yolov5/weights/yolov5l6_v2.2_2048x2048_30.05.2022_conf0.546_aws_neuron.pt")
