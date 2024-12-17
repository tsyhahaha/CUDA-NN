import os
import time
import logging

import torch
from torch.utils.data import DataLoader

import torch.utils.data

import torch.nn.parallel
import numpy as np
from tqdm import tqdm

from model import PointNet, inplace_relu
from dataset import PointCloudDataset, random_point_dropout, random_scale_point_cloud, shift_point_cloud

cfg = {
    'model_path': '/home/tsyhahaha/default',
    'num_class': 10,
    'batch_size': 1,
}

def inference_pad_collate_fn(batch):
    # padding -> chunk
    # max_size = max([item[0].shape[0] for item in batch])
    max_size = 30000
    
    padded_batch = []
    for points, target in batch:
        points = torch.from_numpy(points)[:max_size,:]
        pad_length = max_size - points.shape[0]
        mask = torch.ones(max_size)

        if pad_length > 0:
            mask[points.shape[0]:] = 0
            points = torch.nn.functional.pad(points, (0, 0, 0, pad_length), mode='constant', value=0)
            mask = mask.to(bool)
        padded_batch.append((points, target, mask))
    
    return torch.utils.data.dataloader.default_collate(padded_batch)


def test(model, loader, num_class=10):
    classifier = model.eval()
    correct_sum = 0
    
    for j, (points, target, mask) in tqdm(enumerate(loader)):
        points, target, mask = points.cuda(), target.cuda(), mask.cuda()

        points = points.transpose(2, 1)
        pred, _ = classifier(points, mask.bool())
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.long().data).cpu().sum()
        correct_sum += correct.item()

    return correct_sum / (len(loader)*cfg['batch_size'])

def save_model_params_and_buffers_to_txt(model, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for name, param in model.named_parameters():
        np.savetxt(os.path.join(directory, f'{name}.txt'), param.detach().cpu().numpy().flatten())
    
    for name, buffer in model.named_buffers():
        np.savetxt(os.path.join(directory, f'{name}.txt'), buffer.detach().cpu().numpy().flatten())


def load_model_params_and_buffers_from_txt(model, directory):
    loaded_files = set()
    
    for name, param in model.named_parameters():
        param_file = os.path.join(directory, f'{name}.txt')
        if os.path.exists(param_file):
            param_data = np.loadtxt(param_file)
            param.data.copy_(torch.from_numpy(param_data).view_as(param))
            loaded_files.add(param_file)
        else:
            print(f"Warning: Parameter file '{param_file}' not found.")

    for name, buffer in model.named_buffers():
        buffer_file = os.path.join(directory, f'{name}.txt')
        if os.path.exists(buffer_file):
            buffer_data = np.loadtxt(buffer_file)
            buffer.data.copy_(torch.from_numpy(buffer_data).view_as(buffer))
            loaded_files.add(buffer_file)
        else:
            print(f"Warning: Buffer file '{buffer_file}' not found.")

    for file in os.listdir(directory):
        if file.endswith('.txt'):
            file_path = os.path.join(directory, file)
            if file_path not in loaded_files:
                print(f"Warning: Unloaded weight file '{file}' found in directory.")


def setup():
    os.makedirs('./checkpoints', exist_ok=True)
    os.makedirs('./logs', exist_ok=True)

    current_timestamp = time.time()
    local_time = time.localtime(current_timestamp)
    formatted_time = time.strftime('%m-%d_%H:%M', local_time)
    log_file = os.path.abspath(os.path.join('./logs', f'{formatted_time}.log'))

    level = logging.INFO
    fmt = f'%(asctime)-15s [%(levelname)s] | %(message)s'

    def _handler_apply(h):
        h.setLevel(level)
        h.setFormatter(logging.Formatter(fmt, datefmt='%Y-%m-%d %H:%M:%S'))
        return h

    handlers = [
        logging.StreamHandler(),
        logging.FileHandler(log_file),
        ]

    handlers = list(map(_handler_apply, handlers))

    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        format=fmt,
        level=level,
        handlers=handlers)

    logging.info('-----------------')
    logging.info(f'Arguments: {cfg}')
    logging.info('-----------------')

    logger = logging.getLogger(__name__)

def main():
    setup()

    data_path = '/home/tsyhahaha/CUDA-NN/data/splits'
    test_dataset = PointCloudDataset(root=data_path, split='test')
    test_dataloader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=True,collate_fn=inference_pad_collate_fn)
    logging.info("finish DATA LOADING")

    '''MODEL LOADING'''

    classifier = PointNet(cfg['num_class'])
    classifier.apply(inplace_relu)
    load_model_params_and_buffers_from_txt(classifier, cfg['model_path'])

    classifier = classifier.cuda()

    logging.info("finish MODEL LOADING")

    '''INFERENCING'''

    logging.info("start to INFERENCE")
    instance_acc = test(classifier, test_dataloader, num_class=cfg['num_class'])

    logging.info(f"Accurancy: {instance_acc}")

if __name__ == '__main__':
    main()