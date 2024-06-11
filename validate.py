import os
import argparse
import torch
import torch.nn as nn
import sys

# Append root directory to system path for imports
repo_path, _ = os.path.split(os.path.realpath(__file__))
repo_path, _ = os.path.split(repo_path)
sys.path.append(repo_path)

from utils.seed import seed_all
from utils.config import CFG
from utils.dataset import get_dataset
from utils.model import get_model
from utils.logger import get_logger
from utils.io_tools import dict_to
from utils.metrics import Metrics
import utils.checkpoint as checkpoint
from tqdm import tqdm
import time
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='DSC validating')
    parser.add_argument(
        '--weights',
        dest='weights_file',
        default='',
        metavar='FILE',
        help='path to folder where model.pth file is',
        type=str,
    )
    parser.add_argument(
        '--dset_root',
        dest='dataset_root',
        default=None,
        metavar='DATASET',
        help='path to dataset root folder',
        type=str,
    )
    args = parser.parse_args()
    return args


def validate(model, dset, _cfg, logger, metrics):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    dtype = torch.float32  # Tensor type to be used

    # Moving optimizer and model to used device
    model = model.to(device=device)

    logger.info('=> Passing the network on the validation set...')
    time_list = []
    model.eval()

    with torch.no_grad():
        for t, (data, indices) in enumerate(tqdm(dset, ncols=100)):

            data = dict_to(data, device)
            start_time = time.time()
            scores, loss = model(data)
            time_list.append(time.time() - start_time)

            # Updating batch losses to then get mean for epoch loss
            metrics.losses_track.update_validaiton_losses(loss)

            if (t + 1) % _cfg._dict['VAL']['SUMMARY_PERIOD'] == 0:
                loss_print = '=> Iteration [{}/{}], Train Losses: '.format(t + 1, len(dset))
                for key in loss.keys():
                    loss_print += '{} = {:.6f},  '.format(key, loss[key])
                logger.info(loss_print[:-3])

            metrics.add_batch(prediction=scores, target=model.get_target(data))

        epoch_loss = metrics.losses_track.validation_losses['total'] / metrics.losses_track.validation_iteration_counts

        logger.info('=> [Total Validation Loss = {}]'.format(epoch_loss))
        for scale in metrics.evaluator.keys():
            loss_scale = metrics.losses_track.validation_losses['semantic_{}'.format(scale)].item() / metrics.losses_track.validation_iteration_counts
            logger.info('=> [Scale {}: Loss = {:.6f} - mIoU = {:.6f} - IoU = {:.6f} '
                        '- P = {:.6f} - R = {:.6f} - F1 = {:.6f}]'.format(scale, loss_scale,
                                                                          metrics.get_semantics_mIoU(scale).item(),
                                                                          metrics.get_occupancy_IoU(scale).item(),
                                                                          metrics.get_occupancy_Precision(scale).item(),
                                                                          metrics.get_occupancy_Recall(scale).item(),
                                                                          metrics.get_occupancy_F1(scale).item()))

        logger.info('=> Training set class-wise IoU:')
        for i in range(1, metrics.nbr_classes):
            class_name = dset.dataset.get_xentropy_class_string(i)
            class_score = metrics.evaluator['1_1'].getIoU()[1][i]
            logger.info('    => IoU {}: {:.6f}'.format(class_name, class_score))

        return time_list


def main():

    # https://github.com/pytorch/pytorch/issues/27588
    torch.backends.cudnn.enabled = False

    seed_all(0)

    args = parse_args()

    weights_f = args.weights_file
    dataset_f = args.dataset_root

    assert os.path.isfile(weights_f), '=> No file found at {}'

    checkpoint_path = torch.load(weights_f)
    config_dict = checkpoint_path.pop('config_dict')
    config_dict['DATASET']['DATA_ROOT'] = dataset_f

    # Read train configuration file
    _cfg = CFG()
    _cfg.from_dict(config_dict)
    # Setting the logger to print statements and also save them into logs file
    logger = get_logger(_cfg._dict['OUTPUT']['OUTPUT_PATH'], 'logs_val.log')

    logger.info('============ Validation weights: "%s" ============\n' % weights_f)
    dataset = get_dataset(_cfg._dict)

    logger.info('=> Loading network architecture...')
    model = get_model(_cfg._dict, phase='trainval')
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model = model.module
    logger.info(f'=> Model Parameters: {sum(p.numel() for p in model.parameters())/1000000.0} M')

    logger.info('=> Loading network weights...')
    model = checkpoint.load_model(model, weights_f, logger)

    nbr_iterations = len(dataset['val'])
    metrics = Metrics(_cfg._dict['DATASET']['NCLASS'], nbr_iterations, model.get_scales())
    metrics.reset_evaluator()
    metrics.losses_track.set_validation_losses(model.get_validation_loss_keys())
    metrics.losses_track.set_train_losses(model.get_train_loss_keys())

    time_list = validate(model, dataset['val'], _cfg, logger, metrics)

    logger.info('=> ============ Network Validation Done ============')
    logger.info('Inference time per frame is %.4f seconds\n' % (np.sum(time_list) / len(dataset['val'].dataset)))

    exit()


if __name__ == '__main__':
    main()
