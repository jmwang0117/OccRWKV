import torch.optim as optim


def build_optimizer(_cfg, model):
    opt = _cfg._dict['OPTIMIZER']['TYPE']
    lr  = float(_cfg._dict['OPTIMIZER']['BASE_LR'])
    momentum = _cfg._dict['OPTIMIZER'].get('MOMENTUM', 0.9)  # Default momentum if not specified
    weight_decay = _cfg._dict['OPTIMIZER'].get('WEIGHT_DECAY', 0)  # Default weight decay if not specified

    if opt == 'Adam': 
        optimizer = optim.Adam(model.get_parameters(), lr=lr, betas=(0.9, 0.999))
    elif opt == 'AdamW':
        optimizer = optim.AdamW(model.get_parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
    elif opt == 'SGD': 
        optimizer = optim.SGD(model.get_parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    return optimizer


def build_scheduler(_cfg, optimizer):

  # Constant learning rate
  if _cfg._dict['SCHEDULER']['TYPE'] == 'constant':
    lambda1 = lambda epoch: 1
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)

  # Learning rate scaled by 0.98^(epoch)
  if _cfg._dict['SCHEDULER']['TYPE'] == 'power_iteration':
    lambda1 = lambda epoch: (0.98) ** (epoch)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)


  return scheduler