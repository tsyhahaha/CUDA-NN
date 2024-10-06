import os
import time
import logging

import torch
from torch.utils.data import DataLoader

import torch.utils.data

import torch.nn.parallel
import numpy as np
from tqdm import tqdm

from model import PointNet, Loss, inplace_relu
from dataset import PointCloudDataset, pad_collate_fn, random_point_dropout, random_scale_point_cloud, shift_point_cloud

cfg = {
    'num_class': 10,
    'total_epoch': 200,
    'batch_size': 32,
    'accumulation_step': 4,
}


def test(model, loader, num_class=10):
    mean_correct = []
    classifier = model.eval()

    # for j, (points, target) in tqdm(enumerate(loader), total=len(loader)): 
    for j, (points, target) in enumerate(loader):

        points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

    instance_acc = np.mean(mean_correct)

    return instance_acc

def save_model_params_and_buffers_to_txt(model, directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for name, param in model.named_parameters():
        np.savetxt(os.path.join(directory, f'{name}.txt'), param.detach().cpu().numpy().flatten())
    
    for name, buffer in model.named_buffers():
        np.savetxt(os.path.join(directory, f'{name}.txt'), buffer.detach().cpu().numpy().flatten())

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

    data_path = '../'

    train_dataset = PointCloudDataset(root=data_path, split='train')
    test_dataset = PointCloudDataset(root=data_path, split='test')

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg['batch_size'],
                                  shuffle=True,
                                  num_workers=10,
                                  drop_last=True,
                                  collate_fn=pad_collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False,collate_fn=pad_collate_fn)
    logging.info("finish DATA LOADING")


    '''MODEL LOADING'''

    classifier = PointNet(cfg['num_class'])
    criterion = Loss()
    classifier.apply(inplace_relu)

    classifier = classifier.cuda()
    criterion = criterion.cuda()

    optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    logging.info("finish MODEL LOADING")

    '''TRANING'''

    logging.info("start TRANING")
    total_epoch = cfg['total_epoch']
    accumulation_step = cfg['accumulation_step']
    for epoch in range(total_epoch):
        mean_correct = []
        classifier = classifier.train()

        # for batch_id, (points, target) in tqdm(enumerate(train_dataloader, 0), total=len(train_dataloader), smoothing=0.9):
        total_batch = len(train_dataloader)
        for batch_id, (points, target) in enumerate(train_dataloader, 0):
            optimizer.zero_grad()

            points = points.data.numpy()
            points = random_point_dropout(points)
            points[:, :, 0:3] = random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            points, target = points.cuda(), target.cuda()

            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat) / accumulation_step
            loss.backward()


            if (batch_id + 1) % accumulation_step == 0:
                lr = scheduler.get_last_lr()[0]
                logging.info('Epoch [%d/%d]\tBatch [%d/%d]  \tloss=%.3f\tlr=%.5f' % (epoch + 1, total_epoch, batch_id, total_batch, loss, lr))

                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.long().data).cpu().sum()
                mean_correct.append(correct.item() / float(points.size()[0]))

                optimizer.step()
                global_step += 1

        scheduler.step()

        train_instance_acc = np.mean(mean_correct)

        logging.info('Train Instance Accuracy: %f' % train_instance_acc)

        with torch.no_grad():
            instance_acc = test(classifier.eval(), test_dataloader, num_class=cfg['num_class'])

            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1
                torch.save(classifier.state_dict(), './checkpoints/best_model.pth')
                save_model_params_and_buffers_to_txt(classifier, f'./checkpoints/best_model')
            save_model_params_and_buffers_to_txt(classifier, f'./checkpoints/epoch_{epoch}')

            logging.info('Epoch [%d] - Test Instance Accuracy: %f' % (epoch + 1, instance_acc))
            logging.info('Best Instance Accuracy: %f' % (best_instance_acc))

    logging.info("finish TRANING")
    save_model_params_and_buffers_to_txt(classifier, f'./checkpoints/epoch_last')

if __name__ == '__main__':
    main()