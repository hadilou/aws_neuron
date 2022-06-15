import torch
import torch.neuron
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as dset
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import PIL
import os
import time
import numpy as np
import argparse
n_cores = 4
n_threads = 32

def get_results_as_dict(boxes, scores, labels, image_orig):
    """
    Transforms post-processed output into dictionary output.
    
    This translates the model coordinate bounding boxes (x1, y1, x2, y2) 
    into a rectangular description (x, y, width, height) scaled to the 
    original image size.
    
    Args:
        boxes (torch.Tensor): The Yolo V4 bounding boxes.
        scores (torch.Tensor): The label score for each box.
        labels (torch.Tensor): The label for each box.
        image_orig (torch.Tensor): The image to scale the bounding boxes to.
        
    Returns:
        output (dict): The dictionary of rectangle bounding boxes.
    """
    h_size, w_size = image_orig.shape[-2:]

    x1 = boxes[:, 0] * w_size
    y1 = boxes[:, 1] * h_size
    x2 = boxes[:, 2] * w_size
    y2 = boxes[:, 3] * h_size

    width = x2 - x1
    height = y2 - y1

    boxes = torch.stack([x1, y1, width, height]).T
    return {
        'boxes': boxes.detach().numpy(),
        'labels': labels.detach().numpy(),
        'scores': scores.detach().numpy(),
    }


def get_image_filenames(root=os.getcwd(),work_dir=True):
    """
    Generate paths to the coco dataset image files.
    
    Args:
        root (str): The root folder contains.
    
    Yields:
        filename (str): The path to an image file.
    """
    if not work_dir:
        image_path = "../panoramas/Tallinn"
        for root, dirs, files in os.walk(image_path):
            for filename in files:
                yield os.path.join(image_path, filename)
    
    else:  
        load_path = "../panoramas/Test_UniSteel_Greece/"
        streams = os.listdir(load_path)

        for stream in (streams):
            shoots = os.listdir(os.path.join(load_path,stream))
            for shoot in shoots:
                path_2048 = os.path.join(load_path,stream,shoot,'2048')
                for filename in os.listdir(path_2048):
                    if filename == '3.jpg' or filename == '4.jpg':
                        continue 
                    yield(os.path.join(path_2048,filename))
            
            
def preprocess(path):
    """
    Load an image and convert to the expected Yolo V4 tensor format.
    
    Args:
        path (str): The image file to load from disk.  
        
    Returns:
        result (torch.Tensor): The image for prediction. Shape: [1, 3, 2048, 2048]
    """
    image = PIL.Image.open(path).convert('RGB')
    resized = torchvision.transforms.functional.resize(image, [2048, 2048])
    tensor = torchvision.transforms.functional.to_tensor(resized)
    return tensor.unsqueeze(0).to(torch.float32)



def load_model(filename='../yolov5/weights/yolov5l6_v2.2_2048x2048_30.05.2022_conf0.546_aws_neuron_640_compiled.pt'):
    """
    Load and pre-warm the Yolo V5 model.
    
    Args:
        filename (str): The location to load the model from.
        
    Returns:
        model (torch.nn.Module): The torch model.
    """
    
    # Load model from disk
    model = torch.jit.load(filename)

    # Warm up model on neuron by running a single example image
    filename = next(iter(get_image_filenames()))
    image = preprocess(filename)
    model(image)

    return model


def task(model, filename):
    """
    The thread task to perform prediction.
    
    This does the full end-to-end processing of an image from loading from disk
    all the way to classifying and filtering bounding boxes.
    
    Args:
        model (torch.nn.Module): The model to run processing with
        filename (str): The image file to load from disk.  
    
    Returns:
        boxes (torch.Tensor): The Yolo V4 bounding boxes.
        scores (torch.Tensor): The label score for each box.
        labels (torch.Tensor): The label for each box.        
    """
    image = preprocess(filename)
    begin = time.time()
    result = model(image)
    delta = time.time() - begin
    return result, delta


def benchmark():
    """
    Run a benchmark on the entire dataset against the neuron model.
    """
    start_time = time.time()
    # Load a model into each NeuronCore
    t1_model = time.time()
    models = [load_model() for _ in range(n_cores)]
    t2_model = time.time()
    print(f"Loading models to {n_cores} Neuron Cores took {t2_model - t1_model} s")
    # Create input/output lists
    filenames = list(get_image_filenames())
    results = list()
    latency = list()
    
    # We want to keep track of average completion time per thread
    sum_time = 0.0
    
    # Submit all tasks and wait for them to finish
    with ThreadPoolExecutor(n_threads) as pool:
        for i, filename in enumerate(filenames):
            out = pool.submit(task, models[i % len(models)], filename)
            results.append(out)
        for result in results:
            results, times = result.result() # Note: Outputs unused for benchmark
            latency.append(times)
            sum_time += times
    end_time = time.time()
    print(f"Total filenames :{len(filenames)}")
    print('Prediction Duration in Seconds', sum_time / n_threads)
    print('Images Predicted Per Second:', len(filenames) / (sum_time / n_threads))
    print(f"Total time elapsed {end_time-start_time}")
    # print("Results",results)
    # print("Latency",latency)
    # print("Latency P50: {:.1f}".format(np.percentile(latency, 50)))
    # print("Latency P90: {:.1f}".format(np.percentile(latency, 90)))
    # print("Latency P95: {:.1f}".format(np.percentile(latency, 95)))
    # print("Latency P99: {:.1f}".format(np.percentile(latency, 99)))


benchmark()
