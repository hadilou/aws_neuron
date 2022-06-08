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

n_cores = 4
n_threads = 16
def get_image_filenames(root=os.getcwd()):
    """
    Generate paths to the coco dataset image files.
    
    Args:
        root (str): The root folder contains.
    
    Yields:
        filename (str): The path to an image file.
    """
    # image_path = os.path.join(root, 'val2017')
    image_path = "../panoramas"
    for root, dirs, files in os.walk(image_path):
        for filename in files:
            yield os.path.join(image_path, filename)

def preprocess(path):
    """
    Load an image and convert to the expected Yolo V4 tensor format.
    
    Args:
        path (str): The image file to load from disk.  
        
    Returns:
        result (torch.Tensor): The image for prediction. Shape: [1, 3, 2048, 2048]
    """
    image = PIL.Image.open(path).convert('RGB')
    resized = torchvision.transforms.functional.resize(image, [640, 640])
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
    
    # Load a model into each NeuronCore
    models = [load_model() for _ in range(n_cores)]
    
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
            results, times = result[0],result[1] # Note: Outputs unused for benchmark
            latency.append(times)
            sum_time += times
    
    print('Duration: ', sum_time / n_threads)
    print('Images Per Second:', len(filenames) / (sum_time / n_threads))
    print("Latency P50: {:.1f}".format(np.percentile(latency[1000:], 50)*1000.0))
    print("Latency P90: {:.1f}".format(np.percentile(latency[1000:], 90)*1000.0))
    print("Latency P95: {:.1f}".format(np.percentile(latency[1000:], 95)*1000.0))
    print("Latency P99: {:.1f}".format(np.percentile(latency[1000:], 99)*1000.0))

benchmark()
