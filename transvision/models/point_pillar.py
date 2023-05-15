import torch.nn as nn

from transvision.models.sub_modules.base_bev_backbone import BaseBEVBackbone
from transvision.models.sub_modules.pillar_vfe import PillarVFE
from transvision.models.sub_modules.point_pillar_scatter import PointPillarScatter


class PointPillar(nn.Module):
    def __init__(self, args):
        super(PointPillar, self).__init__()

        # PIllar VFE
        self.pillar_vfe = PillarVFE(args["pillar_vfe"], num_point_features=4, voxel_size=args["voxel_size"], point_cloud_range=args["lidar_range"])
        self.scatter = PointPillarScatter(args["point_pillar_scatter"])
        self.backbone = BaseBEVBackbone(args["base_bev_backbone"], 64)

        self.cls_head = nn.Conv2d(128 * 3, args["anchor_num"], kernel_size=1)
        self.reg_head = nn.Conv2d(128 * 3, 7 * args["anchor_num"], kernel_size=1)

        if "dir_args" in args.keys():
            self.use_dir = True
            self.dir_head = nn.Conv2d(128 * 3, args["dir_args"]["num_bins"] * args["anchor_num"], kernel_size=1)  # BIN_NUM = 2
        else:
            self.use_dir = False

    def forward(self, data_dict):

        voxel_features = data_dict["processed_lidar"]["voxel_features"]
        voxel_coords = data_dict["processed_lidar"]["voxel_coords"]
        voxel_num_points = data_dict["processed_lidar"]["voxel_num_points"]

        batch_dict = {"voxel_features": voxel_features, "voxel_coords": voxel_coords, "voxel_num_points": voxel_num_points}

        batch_dict = self.pillar_vfe(batch_dict)
        batch_dict = self.scatter(batch_dict)
        batch_dict = self.backbone(batch_dict)

        spatial_features_2d = batch_dict["spatial_features_2d"]

        psm = self.cls_head(spatial_features_2d)
        rm = self.reg_head(spatial_features_2d)

        output_dict = {"psm": psm, "rm": rm}

        if self.use_dir:
            dm = self.dir_head(spatial_features_2d)
            output_dict.update({"dm": dm})

        return output_dict
