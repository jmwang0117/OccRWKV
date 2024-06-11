import torch
import torch_scatter
import torch.nn as nn
import torch.nn.functional as F


def quantitize(data, lim_min, lim_max, size, with_res=False):
    idx = (data - lim_min) / (lim_max - lim_min) * size.float()
    idxlong = idx.type(torch.cuda.LongTensor)
    if with_res:
        idx_res = idx - idxlong.float()
        return idxlong, idx_res
    else:
        return idxlong


class VFELayerMinus(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 name='',
                 last_vfe = False):
        super().__init__()
        self.name = 'VFELayerMinusSlim' + name
        if not last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        self.linear = nn.Linear(in_channels, self.units, bias=True)
        self.weight_linear = nn.Linear(6, self.units, bias=True)

    def forward(self, inputs, bxyz_indx, mean=None, activate=False, gs=None):
        x = self.linear(inputs)
        if activate:
            x = F.relu(x)
        if gs is not None:
            x = x * gs
        if mean is not None:
            x_weight = self.weight_linear(mean)
            if activate:
                x_weight = F.relu(x_weight)
            x = x * x_weight
        _, value = torch.unique(bxyz_indx, return_inverse=True, dim=0)
        max_feature, _ = torch_scatter.scatter_max(x, value, dim=0)
        gather_max_feature = max_feature[value, :]
        x_concated = torch.cat((x, gather_max_feature), dim=1)
        return x_concated


class PcPreprocessor(nn.Module):
    def __init__(self, lims, sizes, grid_meters, init_size=32, offset=0.5, pooling_scales=[0.5, 1, 2, 4, 6, 8]):
        # todo move to cfg
        super().__init__()
        self.sizes = torch.tensor(sizes).float()
        self.lims = lims
        self.pooling_scales = pooling_scales
        self.grid_meters = grid_meters
        self.offset = offset
        self.target_scale = 1

        self.multi_scale_top_layers = nn.ModuleDict()
        self.feature_list = {
            0.5: [10, init_size],
            1: [10, init_size],
        }
        self.target_scale = 1
        for scale in self.feature_list.keys():
            top_layer = VFELayerMinus(self.feature_list[scale][0],
                                          self.feature_list[scale][1],
                                          "top_layer_" + str(int(10*scale) if scale == 0.5 else scale))
            self.multi_scale_top_layers[str(int(10*scale) if scale == 0.5 else scale)] = top_layer

        self.aggtopmeanproj = nn.Linear(6, init_size, bias=True)
        self.aggtopproj = nn.Linear(2*init_size, init_size, bias=True)
        self.aggfusion = nn.Linear(init_size, init_size, bias=True)

    def add_pcmean_and_gridmean(self, pc, bxyz_indx, return_mean=False):
        _, value = torch.unique(bxyz_indx, return_inverse=True, dim=0)
        pc_mean = torch_scatter.scatter_mean(pc[:, :3], value, dim=0)[value]
        pc_mean_minus = pc[:, :3] - pc_mean

        m_pergird = torch.tensor(self.grid_meters, dtype=torch.float, device=pc.device)
        xmin_ymin_zmin = torch.tensor([self.lims[0][0], self.lims[1][0], self.lims[2][0]], dtype=torch.float, device=pc.device)
        pc_gridmean = (bxyz_indx[:, 1:].type(torch.cuda.FloatTensor) + self.offset) * m_pergird + xmin_ymin_zmin
        pc_gridmean_minus = pc[:, :3] - pc_gridmean

        pc_feature = torch.cat((pc, pc_mean_minus, pc_gridmean_minus), dim=1)  # same input n, 10
        mean = torch.cat((pc_mean_minus, pc_gridmean_minus), dim=1)  # different input_mean n, 6
        if return_mean:
            return pc_feature, mean
        else:
            return pc_feature

    def extract_geometry_features(self, pc, info):
        ms_mean_features = {}
        ms_pc_features = []
        for scale in self.feature_list.keys():
            bxyz_indx = info[scale]['bxyz_indx'].long()
            pc_feature, topview_mean = self.add_pcmean_and_gridmean(pc, bxyz_indx, return_mean=True)
            pc_feature = self.multi_scale_top_layers[str(int(10*scale) if scale == 0.5 else scale)](
                pc_feature, bxyz_indx, mean=topview_mean)
            ms_mean_features[scale] = topview_mean
            ms_pc_features.append(pc_feature)

        agg_tpfeature = F.relu(self.aggtopmeanproj(ms_mean_features[self.target_scale])) \
                        * F.relu(self.aggtopproj(torch.cat(ms_pc_features, dim=1)))
        agg_tpfeature = self.aggfusion(agg_tpfeature)

        bxyz_indx_tgt = info[self.target_scale]['bxyz_indx'].long()
        index, value = torch.unique(bxyz_indx_tgt, return_inverse=True, dim=0)
        maxf = torch_scatter.scatter_max(agg_tpfeature, value, dim=0)[0]

        return maxf, index, value

    def forward(self, pc, indicator):
        indicator_t = []
        tensor = torch.ones((1,), dtype=torch.long).to(pc)
        for i in range(len(indicator) - 1):
            indicator_t.append(tensor.new_full((indicator[i+1] - indicator[i],), i))
        indicator_t = torch.cat(indicator_t, dim=0)
        info = {'batch': len(indicator)-1}
        self.sizes = self.sizes.to(pc)

        for scale in self.pooling_scales:
            xidx, xres = quantitize(pc[:, 0], self.lims[0][0],
                                              self.lims[0][1], self.sizes[0] // scale, with_res=True)
            yidx, yres = quantitize(pc[:, 1], self.lims[1][0],
                                              self.lims[1][1], self.sizes[1] // scale, with_res=True)
            zidx, zres = quantitize(pc[:, 2], self.lims[2][0],
                                              self.lims[2][1], self.sizes[2] // scale, with_res=True)
            bxyz_indx = torch.stack([indicator_t, xidx, yidx, zidx], dim=-1)
            xyz_res = torch.stack([xres, yres, zres], dim=-1)
            info[scale] = {'bxyz_indx': bxyz_indx, 'xyz_res': xyz_res}

        voxel_feature, coord_ind, full_coord = self.extract_geometry_features(pc, info)

        return voxel_feature, coord_ind, full_coord, info


