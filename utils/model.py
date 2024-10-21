from networks.occrwkv import OccRWKV


def get_model(_cfg, phase='train'):
    return OccRWKV(_cfg, phase=phase)
