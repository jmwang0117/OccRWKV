from glob import glob
import torch

import os
import yaml
import numpy as np

def mask_op(data, x_min, x_max):
    mask = (data > x_min) & (data < x_max)
    return mask

def get_mask(pc, lims):
    mask_x = mask_op(pc[:, 0], lims[0][0] + 0.0001, lims[0][1] - 0.0001)
    mask_y = mask_op(pc[:, 1], lims[1][0] + 0.0001, lims[1][1] - 0.0001)
    mask_z = mask_op(pc[:, 2], lims[2][0] + 0.0001, lims[2][1] - 0.0001)
    mask = (mask_x) & (mask_y) & mask_z
    return mask


def unpack(compressed):
    ''' given a bit encoded voxel grid, make a normal voxel grid out of it.  '''
    uncompressed = np.zeros(compressed.shape[0] * 8, dtype=np.uint8)
    uncompressed[::8] = compressed[:] >> 7 & 1
    uncompressed[1::8] = compressed[:] >> 6 & 1
    uncompressed[2::8] = compressed[:] >> 5 & 1
    uncompressed[3::8] = compressed[:] >> 4 & 1
    uncompressed[4::8] = compressed[:] >> 3 & 1
    uncompressed[5::8] = compressed[:] >> 2 & 1
    uncompressed[6::8] = compressed[:] >> 1 & 1
    uncompressed[7::8] = compressed[:] & 1

    return uncompressed

def augmentation_random_flip(data, flip_type, is_scan=False):
    if flip_type==1:
        if is_scan:
            data[:, 0] = 51.2 - data[:, 0]
        else:
            data = np.flip(data, axis=0).copy()
    elif flip_type==2:
        if is_scan:
            data[:, 1] = -data[:, 1]
        else:
            data = np.flip(data, axis=1).copy()
    elif flip_type==3:
        if is_scan:
            data[:, 0] = 51.2 - data[:, 0]
            data[:, 1] = -data[:, 1]
        else:
            data = np.flip(np.flip(data, axis=0), axis=1).copy()
    return data

class SemanticKitti(torch.utils.data.Dataset):
    CLASSES = ('unlabeled',
               'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle',
               'person', 'bicyclist', 'motorcyclist', 'road',
               'parking', 'sidewalk', 'other-ground', 'building', 'fence',
               'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign')

    def __init__(self, data_root, data_config_file, setname,
                 lims,
                 sizes,
                 augmentation=False,
                 shuffle_index=False):
        self.data_root = data_root
        self.data_config = yaml.safe_load(open(data_config_file, 'r'))
        self.sequences = self.data_config["split"][setname]
        self.setname = setname
        self.labels = self.data_config['labels']
        self.learning_map = self.data_config["learning_map"]

        self.learning_map_inv = self.data_config["learning_map_inv"]
        self.color_map = self.data_config['color_map']

        self.lims = lims
        self.sizes = sizes
        self.augmentation = augmentation
        self.shuffle_index = shuffle_index

        self.filepaths = {}
        print(f"=> Parsing SemanticKITTI {self.setname}")
        self.get_filepaths()
        self.num_files_ = len(self.filepaths['occupancy'])
        print("Using {} scans from sequences {}".format(self.num_files_, self.sequences))
        print(f"Is aug: {self.augmentation}")

    def get_filepaths(self,):
        # fill in with names, checking that all sequences are complete
        if self.setname != 'test':
            for key in ['label_1_1', 'invalid_1_1', 'label_1_2', 'invalid_1_2', 'label_1_4', 'invalid_1_4', 'label_1_8', 'invalid_1_8', 'occluded', 'occupancy']:
                self.filepaths[key] = []
        else:
            self.filepaths['occupancy'] = []
        for seq in self.sequences:
            # to string
            seq = '{0:02d}'.format(int(seq))
            print("parsing seq {}".format(seq))
            if self.setname != 'test':
                # Scale 1_1
                self.filepaths['label_1_1'] += sorted(glob(os.path.join(self.data_root, seq, 'voxels', '*.label')))
                self.filepaths['invalid_1_1'] += sorted(glob(os.path.join(self.data_root, seq, 'voxels', '*.invalid')))
                # Scale 1_2
                self.filepaths['label_1_2'] += sorted(glob(os.path.join(self.data_root, seq, 'voxels', '*.label_1_2')))
                self.filepaths['invalid_1_2'] += sorted(glob(os.path.join(self.data_root, seq, 'voxels', '*.invalid_1_2')))
                # Scale 1_4
                self.filepaths['label_1_4'] += sorted(glob(os.path.join(self.data_root, seq, 'voxels', '*.label_1_4')))
                self.filepaths['invalid_1_4'] += sorted(glob(os.path.join(self.data_root, seq, 'voxels', '*.invalid_1_4')))
                # Scale 1_4
                self.filepaths['label_1_8'] += sorted(glob(os.path.join(self.data_root, seq, 'voxels', '*.label_1_8')))
                self.filepaths['invalid_1_8'] += sorted(glob(os.path.join(self.data_root, seq, 'voxels', '*.invalid_1_8')))

                # occluded
                self.filepaths['occluded'] += sorted(glob(os.path.join(self.data_root, seq, 'voxels', '*.occluded')))

            self.filepaths['occupancy'] += sorted(glob(os.path.join(self.data_root, seq, 'voxels', '*.bin')))

    def get_data(self, idx, flip_type):
        data_collection = {}
        sc_remap_lut = self.get_remap_lut(completion=True)
        ss_remap_lut = self.get_remap_lut()
        for typ in self.filepaths.keys():
            scale = int(typ.split('_')[-1]) if 'label' in typ or 'invalid' in typ else 1
            if 'label' in typ:
                scan_data = np.fromfile(self.filepaths[typ][idx], dtype=np.uint16)
                if scale == 1:
                    scan_data = sc_remap_lut[scan_data]
            else:
                scan_data = unpack(np.fromfile(self.filepaths[typ][idx], dtype=np.uint8))
            scan_data = scan_data.reshape((self.sizes[0]//scale, self.sizes[1]//scale, self.sizes[2]//scale))
            scan_data = scan_data.astype(np.float32)
            if self.augmentation:
                scan_data = augmentation_random_flip(scan_data, flip_type)
            data_collection[typ] = torch.from_numpy(scan_data)

        points_path = self.filepaths['occupancy'][idx].replace('voxels', 'velodyne')
        points = np.fromfile(points_path, dtype=np.float32)
        points = points.reshape((-1, 4))

        if self.setname != 'test':
            points_label_path = self.filepaths['occupancy'][idx].replace('voxels', 'labels').replace('.bin', '.label')
            points_label = np.fromfile(points_label_path, dtype=np.uint32)
            points_label = points_label.reshape((-1))
            points_label = points_label & 0xFFFF  # semantic label in lower half
            points_label = ss_remap_lut[points_label]

        if self.shuffle_index:
            pt_idx = np.random.permutation(np.arange(0, points.shape[0]))
            points = points[pt_idx]
            if self.setname != 'test':
                points_label = points_label[pt_idx]

        if self.lims:
            filter_mask = get_mask(points, self.lims)
            points = points[filter_mask]
            if self.setname != 'test':
                points_label = points_label[filter_mask]

        if self.augmentation:
            points = augmentation_random_flip(points, flip_type, is_scan=True)

        data_collection['points'] = torch.from_numpy(points)
        if self.setname != 'test':
            data_collection['points_label'] = torch.from_numpy(points_label)

        return data_collection


    def __len__(self):
        return self.num_files_

    def get_n_classes(self):
        return len(self.learning_map_inv)

    def get_remap_lut(self, completion=False):
        # put label from original values to xentropy
        # or vice-versa, depending on dictionary values
        # make learning map a lookup table
        maxkey = max(self.learning_map.keys())

        # +100 hack making lut bigger just in case there are unknown labels
        remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
        remap_lut[list(self.learning_map.keys())] = list(self.learning_map.values())

        # in completion we have to distinguish empty and invalid voxels.
        # Important: For voxels 0 corresponds to "empty" and not "unlabeled".
        if completion:
            remap_lut[remap_lut == 0] = 255  # map 0 to 'invalid'
            remap_lut[0] = 0  # only 'empty' stays 'empty'.
        
        return remap_lut

    def get_inv_remap_lut(self):
        '''
        remap_lut to remap classes of semantic kitti for training...
        :return:
        '''
        # make lookup table for mapping
        maxkey = max(self.learning_map_inv.keys())

        # +100 hack making lut bigger just in case there are unknown labels
        remap_lut = np.zeros((maxkey + 1), dtype=np.int32)
        remap_lut[list(self.learning_map_inv.keys())] = list(self.learning_map_inv.values())

        return remap_lut

    def to_color(self, label):
        # put label in original values
        label = SemanticKitti.map(label, self.learning_map_inv)
        # put label in color
        return SemanticKitti.map(label, self.color_map)

    def get_xentropy_class_string(self, idx):
        return self.labels[self.learning_map_inv[idx]]

    def __getitem__(self, idx):
        flip_type = np.random.randint(0, 4) if self.augmentation else 0
        return self.get_data(idx, flip_type), idx
