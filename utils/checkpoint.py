from torch.nn.parallel import DataParallel, DistributedDataParallel
import torch
import os
from glob import glob

from .io_tools import _remove_recursively, _create_directory


def load(model, optimizer, scheduler, resume, path, logger):
  '''
  Load checkpoint file
  '''

  # If not resume, initialize model and return everything as it is
  if not resume:
    logger.info('=> No checkpoint. Initializing model from scratch')
    model.weights_init()
    epoch = 1
    return model, optimizer, scheduler, epoch

  # If resume, check that path exists and load everything to return
  else:
    file_path = glob(os.path.join(path, '*.pth'))[0]
    assert os.path.isfile(file_path), '=> No checkpoint found at {}'.format(path)
    checkpoint = torch.load(file_path)
    epoch = checkpoint.pop('startEpoch')

    s = model.module.state_dict() if isinstance(model, (DataParallel, DistributedDataParallel)) else model.state_dict()
    for key, val in checkpoint['model'].items():
      if key[:6] == 'module':
        key = key[7:]
      
      if key in s and s[key].shape == val.shape:
        s[key][...] = val
      elif key not in s:
        print('igonre weight from not found key {}'.format(key))
      else:
        print('ignore weight of mistached shape in key {}'.format(key))

    if isinstance(model, (DataParallel, DistributedDataParallel)):
      model.module.load_state_dict(s)
    else:
      model.load_state_dict(s)
    optimizer.load_state_dict(checkpoint.pop('optimizer'))
    scheduler.load_state_dict(checkpoint.pop('scheduler'))
    logger.info('=> Continuing training routine. Checkpoint loaded at {}'.format(file_path))
    return model, optimizer, scheduler, epoch


def load_model(model, filepath, logger):
  '''
  Load checkpoint file
  '''

  # check that path exists and load everything to return
  assert os.path.isfile(filepath), '=> No file found at {}'
  checkpoint = torch.load(filepath)

  s = model.module.state_dict() if isinstance(model, (DataParallel, DistributedDataParallel)) else model.state_dict()
  for key, val in checkpoint['model'].items():
    if key[:6] == 'module':
      key = key[7:]
      
    if key in s and s[key].shape == val.shape:
      s[key][...] = val
    elif key not in s:
      print('igonre weight from not found key {}'.format(key))
    else:
      print('ignore weight of mistached shape in key {}'.format(key))

  if isinstance(model, (DataParallel, DistributedDataParallel)):
    model.module.load_state_dict(s)
  else:
    model.load_state_dict(s)
  logger.info('=> Model loaded at {}'.format(filepath))
  return model


def save(path, model, optimizer, scheduler, epoch, config):
  '''
  Save checkpoint file
  '''

  # Remove recursively if epoch_last folder exists and create new one
  # _remove_recursively(path)
  _create_directory(path)

  weights_fpath = os.path.join(path, 'weights_epoch_{}.pth'.format(str(epoch).zfill(3)))

  torch.save({
    'startEpoch': epoch+1,  # To start on next epoch when loading the dict...
    'model': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'scheduler': scheduler.state_dict(),
    'config_dict': config
  }, weights_fpath)

  return weights_fpath