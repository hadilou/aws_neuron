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

def postprocess(boxes, scores, score_threshold=0.05, iou_threshold=0.5):
    """
    Classifies and filters bounding boxes from Yolo V4 output.
    
    Performs classification, filtering, and non-maximum suppression to remove
    boxes that are irrelevant. The result is the filtered set of boxes, the 
    associated label confidence score, and the predicted label.
    
    See: https://pytorch.org/docs/stable/torchvision/ops.html#torchvision.ops.nms
    
    Args:
        boxes (torch.Tensor): The Yolo V5 bounding boxes.
        scores (torch.Tensor): The categories scores for each box.
        score_threshold (float): Ignore boxes with scores below threshold.
        iou_threshold (float): Discards boxes with intersection above threshold. 
            
    Returns:
        boxes (torch.Tensor): The filtered Yolo V5 bounding boxes.
        scores (torch.Tensor): The label score for each box.
        labels (torch.Tensor): The label for each box.
    """
    
    # shape: [n_batch, n_boxes, 1, 4] => [n_boxes, 4]  # Assumes n_batch size is 1
    boxes = boxes.squeeze()

    # shape: [n_batch, n_boxes, 80] => [n_boxes, 80]  # Assumes n_batch size is 1
    scores = scores.squeeze()

    # Classify each box according to the maximum category score
    score, column = torch.max(scores, dim=1)

    # Filter out rows for scores which are below threshold
    mask = score > score_threshold

    # Filter model output data
    boxes = boxes[mask]
    score = score[mask]
    idxs = column[mask]

    # Perform non-max suppression on all categories at once. shape: [n_keep,]
    keep = torchvision.ops.batched_nms(
        boxes=boxes, 
        scores=score, 
        idxs=idxs,
        iou_threshold=iou_threshold,
    )

    # The image category id associated with each column
    categories = torch.tensor([
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
        44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56,
        57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72,
        73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85,
        86, 87, 88, 89, 90
    ])
    
    boxes = boxes[keep]       # shape: [n_keep, 4]
    score = score[keep]       # shape: [n_keep,]
    idxs = idxs[keep]
    label = categories[idxs]  # shape: [n_keep,]
    
    return boxes, score, label


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
    start_time = time.time()
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
            results, times = result.result() # Note: Outputs unused for benchmark
            latency.append(times)
            sum_time += times
    end_time = time.time()
    print(f"Total filenames :{len(filenames)}")
    print('Duration in Seconds', sum_time / n_threads)
    print('Images Per Second:', len(filenames) / (sum_time / n_threads))
    print(f"End time - Start time in Seconds {end_time-start_time}")
    # print("Results",results)
    # print("Latency",latency)
    # print("Latency P50: {:.1f}".format(np.percentile(latency, 50)))
    # print("Latency P90: {:.1f}".format(np.percentile(latency, 90)))
    # print("Latency P95: {:.1f}".format(np.percentile(latency, 95)))
    # print("Latency P99: {:.1f}".format(np.percentile(latency, 99)))


benchmark()
