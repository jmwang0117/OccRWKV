from networks.dsc import DSC


def get_model(_cfg, phase='train'):
    return DSC(_cfg, phase=phase)
