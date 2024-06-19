from typing import Tuple

import matplotlib.pyplot as plt
import mmcv
import torch
from mmcv.runner import force_fp32
from mmdet3d.ops import bev_pool
from torch import nn

__all__ = ['BaseTransform', 'BaseDepthTransform', 'CoopBaseTransform', 'CoopInfraBaseTransform', 'CoopVehicleBaseTransform', 'CoopBaseDepthTransform']

lid_plot_idx = 0
plot_idx = 0


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = torch.Tensor(  # Change to LongTensor for non-coop models
        [(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    return dx, bx, nx


def visualize_feature_lidar(points, path):
    global plot_idx
    path = path + str(plot_idx) + '.png'
    plot_idx = plot_idx + 1
    fig = plt.figure(figsize=(500, 500))

    ax = plt.gca()
    ax.set_xlim((-250, 250))
    ax.set_ylim((-250, 250))
    ax.set_aspect(1)
    ax.set_axis_off()

    plt.scatter(
        points[:, 0],
        points[:, 1],
        s=15,
        c='white',
    )

    fig.savefig(
        path,
        dpi=10,
        facecolor='black',
        format='png',
        bbox_inches='tight',
        pad_inches=0,
    )
    plt.close()


def visualize_feature_image(image, path):
    global plot_idx
    path = path + str(plot_idx) + '.png'
    plot_idx = plot_idx + 1

    canvas = image.copy()

    mmcv.imwrite(canvas, path)


def visualize_feature_map_lidar_fused(feature):
    global lid_plot_idx
    lid_plot_idx += 1
    feature = feature.detach().cpu().squeeze(0)
    feature = feature[0]  # For when training
    gray_scale = torch.sum(feature, 0)
    gray_scale = gray_scale / feature.shape[0]
    plt.imshow(gray_scale)
    plt.savefig(str('/home/bevfusion/viz_tumtraf_featmap/features/lidar/fused/feature_map_' + str(lid_plot_idx) + '.png'), bbox_inches='tight')


class BaseTransform(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        image_size: Tuple[int, int],
        feature_size: Tuple[int, int],
        xbound: Tuple[float, float, float],
        ybound: Tuple[float, float, float],
        zbound: Tuple[float, float, float],
        dbound: Tuple[float, float, float],
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.feature_size = feature_size
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound

        dx, bx, nx = gen_dx_bx(self.xbound, self.ybound, self.zbound)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.C = out_channels
        self.frustum = self.create_frustum()
        self.D = self.frustum.shape[0]
        self.fp16_enabled = False

    @force_fp32()
    def create_frustum(self):
        iH, iW = self.image_size
        fH, fW = self.feature_size

        ds = (torch.arange(*self.dbound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW))
        D, _, _ = ds.shape

        xs = (torch.linspace(0, iW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW))
        ys = (torch.linspace(0, iH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW))

        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    @force_fp32()
    def get_geometry(
        self,
        camera2lidar_rots,
        camera2lidar_trans,
        intrins,
        post_rots,
        post_trans,
        **kwargs,
    ):
        B, N, _ = camera2lidar_trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = (torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)))
        # cam_to_lidar
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        combine = camera2lidar_rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)

        if 'extra_rots' in kwargs:
            extra_rots = kwargs['extra_rots']
            points = (extra_rots.view(B, 1, 1, 1, 1, 3, 3).repeat(1, N, 1, 1, 1, 1, 1).matmul(points.unsqueeze(-1)).squeeze(-1))
        if 'extra_trans' in kwargs:
            extra_trans = kwargs['extra_trans']
            points += extra_trans.view(B, 1, 1, 1, 1, 3).repeat(1, N, 1, 1, 1, 1)

        return points

    def get_cam_feats(self, x):
        raise NotImplementedError

    @force_fp32()
    def bev_pool(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = ((geom_feats[:, 0] >= 0)
                & (geom_feats[:, 0] < self.nx[0])
                & (geom_feats[:, 1] >= 0)
                & (geom_feats[:, 1] < self.nx[1])
                & (geom_feats[:, 2] >= 0)
                & (geom_feats[:, 2] < self.nx[2]))
        x = x[kept]
        geom_feats = geom_feats[kept]

        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)

        return final

    @force_fp32()
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        **kwargs,
    ):
        # rots = camera2ego[..., :3, :3]
        # trans = camera2ego[..., :3, 3]
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        # lidar2ego_rots = lidar2ego[..., :3, :3]
        # lidar2ego_trans = lidar2ego[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        x = self.get_cam_feats(img)
        x = self.bev_pool(geom, x)
        return x


class CoopBaseTransform(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, image_size: Tuple[int, int], feature_size: Tuple[int, int], xbound: Tuple[float, float, float],
                 ybound: Tuple[float, float, float], zbound: Tuple[float, float, float], dbound: Tuple[float, float, float], vehicle: bool) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.image_size = image_size
        self.feature_size = feature_size
        self.xbound = xbound
        self.ybound = ybound
        self.zbound = zbound
        self.dbound = dbound
        self.vehicle = vehicle

        dx, bx, nx = gen_dx_bx(self.xbound, self.ybound, self.zbound)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.C = out_channels
        self.frustum = self.create_frustum()
        self.D = self.frustum.shape[0]
        self.fp16_enabled = False

    @force_fp32()
    def create_frustum(self):
        iH, iW = self.image_size
        fH, fW = self.feature_size

        ds = (torch.arange(*self.dbound, dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW))
        D, _, _ = ds.shape

        xs = (torch.linspace(0, iW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW))
        ys = (torch.linspace(0, iH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW))

        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    @force_fp32()
    def get_geometry(
        self,
        camera2lidar_rots,
        camera2lidar_trans,
        intrins,
        post_rots,
        post_trans,
        **kwargs,
    ):
        B, N, _ = camera2lidar_trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = (torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1)))
        # cam_to_lidar
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:3],
            ),
            5,
        )
        combine = camera2lidar_rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += camera2lidar_trans.view(B, N, 1, 1, 1, 3)

        if 'extra_rots' in kwargs:
            extra_rots = kwargs['extra_rots']
            points = (extra_rots.view(B, 1, 1, 1, 1, 3, 3).repeat(1, N, 1, 1, 1, 1, 1).matmul(points.unsqueeze(-1)).squeeze(-1))
        if 'extra_trans' in kwargs:
            extra_trans = kwargs['extra_trans']
            points += extra_trans.view(B, 1, 1, 1, 1, 3).repeat(1, N, 1, 1, 1, 1)

        return points

    def get_cam_feats(self, x):
        raise NotImplementedError

    @force_fp32()
    def bev_pool(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.0)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix, device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = ((geom_feats[:, 0] >= 0)
                & (geom_feats[:, 0] < self.nx[0])
                & (geom_feats[:, 1] >= 0)
                & (geom_feats[:, 1] < self.nx[1])
                & (geom_feats[:, 2] >= 0)
                & (geom_feats[:, 2] < self.nx[2]))
        x = x[kept]
        geom_feats = geom_feats[kept]

        x = bev_pool(x, geom_feats, B, self.nx[2], self.nx[0], self.nx[1])

        # collapse Z
        final = torch.cat(x.unbind(dim=2), 1)

        return final

    @force_fp32()
    def forward(
        self,
        img,
        points,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        **kwargs,
    ):
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        x = self.get_cam_feats(img)
        x = self.bev_pool(geom, x)
        return x


class CoopInfraBaseTransform(CoopBaseTransform):

    def forward(
        self,
        training,
        img,
        points,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        vehicle2infrastructure,
        metas,
        **kwargs,
    ):
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]

        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        x = self.get_cam_feats(img)
        x = self.bev_pool(geom, x)
        return x


class CoopVehicleBaseTransform(CoopBaseTransform):

    def forward(
        self,
        training,
        img,
        points,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        vehicle2infrastructure,
        metas,
        **kwargs,
    ):
        intrins = camera_intrinsics[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]
        lidar_aug_rots = lidar_aug_matrix[..., :3, :3]
        lidar_aug_trans = lidar_aug_matrix[..., :3, 3]
        vehicle2infrastructure_rots = vehicle2infrastructure[..., :3, :3]
        vehicle2infrastructure_trans = vehicle2infrastructure[..., :3, 3]

        batch_size = len(points)

        extra_rots_list = []
        extra_trans_list = []
        for b in range(batch_size):
            extra_rots_list.append(lidar_aug_rots[b].matmul(vehicle2infrastructure_rots[b]).to(torch.float32))  # aug x v2i because v2i happens first
            extra_trans_list.append(torch.add(lidar_aug_trans[b], vehicle2infrastructure_trans[b]))

        extra_rots = torch.stack(extra_rots_list)
        extra_trans = torch.stack(extra_trans_list)

        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        x = self.get_cam_feats(img)
        x = self.bev_pool(geom, x)
        return x


class BaseDepthTransform(BaseTransform):

    @force_fp32()
    def forward(
        self,
        img,
        points,
        sensor2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        cam_intrinsic,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        **kwargs,
    ):
        # rots = sensor2ego[..., :3, :3]
        # trans = sensor2ego[..., :3, 3]
        intrins = cam_intrinsic[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        # lidar2ego_rots = lidar2ego[..., :3, :3]
        # lidar2ego_trans = lidar2ego[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        camera2lidar_trans = camera2lidar[..., :3, 3]

        # print(img.shape, self.image_size, self.feature_size)

        batch_size = len(points)
        # print("batch_size", batch_size)
        depth = torch.zeros(batch_size, img.shape[1], 1, *self.image_size).to(points[0].device)

        for b in range(batch_size):
            cur_coords = points[b][:, :3]
            cur_img_aug_matrix = img_aug_matrix[b]
            cur_lidar_aug_matrix = lidar_aug_matrix[b]
            cur_lidar2image = lidar2image[b]

            # inverse aug
            cur_coords -= cur_lidar_aug_matrix[:3, 3]
            cur_coords = torch.inverse(cur_lidar_aug_matrix[:3, :3]).matmul(cur_coords.transpose(1, 0))
            # lidar2image
            cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)
            # get 2d coords
            dist = cur_coords[:, 2, :]
            cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
            cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

            # imgaug
            cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
            cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
            cur_coords = cur_coords[:, :2, :].transpose(1, 2)

            # normalize coords for grid sample
            cur_coords = cur_coords[..., [1, 0]]

            on_img = ((cur_coords[..., 0] < self.image_size[0]) & (cur_coords[..., 0] >= 0) & (cur_coords[..., 1] < self.image_size[1]) & (cur_coords[..., 1] >= 0))
            for c in range(on_img.shape[0]):
                masked_coords = cur_coords[c, on_img[c]].long()
                masked_dist = dist[c, on_img[c]]
                depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist

        extra_rots = lidar_aug_matrix[..., :3, :3]
        extra_trans = lidar_aug_matrix[..., :3, 3]
        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        x = self.get_cam_feats(img, depth)
        x = self.bev_pool(geom, x)
        return x


class CoopBaseDepthTransform(CoopBaseTransform):

    @force_fp32()
    def forward(
        self,
        training,
        img,
        points,
        lidar2camera,
        lidar2image,
        cam_intrinsic,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        vehicle2infrastructure,
        metas,
        **kwargs,
    ):
        intrins = cam_intrinsic[..., :3, :3]
        post_rots = img_aug_matrix[..., :3, :3]
        post_trans = img_aug_matrix[..., :3, 3]
        camera2lidar_rots = camera2lidar[..., :3, :3]
        # print("camera2lidar_trans shape", camera2lidar.shape)
        camera2lidar_trans = camera2lidar[..., :3, 3]
        lidar_aug_rots = lidar_aug_matrix[..., :3, :3]
        lidar_aug_trans = lidar_aug_matrix[..., :3, 3]
        vehicle2infrastructure_rots = vehicle2infrastructure[..., :3, :3]
        vehicle2infrastructure_trans = vehicle2infrastructure[..., :3, 3]

        # print(img.shape, self.image_size, self.feature_size)

        batch_size = len(points)
        # print("device", points[0].device)
        # print("batch size", batch_size)
        # print("img shape", img.shape)
        # print("image size", self.image_size)
        depth = torch.zeros(batch_size, img.shape[1], 1, *self.image_size).to(points[0].device)

        for b in range(batch_size):
            if self.vehicle:
                # we are in vehicle, currently infra coord post transform
                cur_coords = points[b][:, :3]
                cur_img_aug_matrix = img_aug_matrix[b]
                cur_lidar2image = lidar2image[b]

                # if not training:
                # visualize_feature_lidar(cur_coords.detach().cpu().numpy(), "/home/bevfusion/viz_tumtraf_featmap/features/lidar/infra/orival")

                # inverse LiDAR aug (infra and vehicle augs are the same) -> Only have rotation and translation, applied similarly to original, so just follow original inversion
                cur_coords -= lidar_aug_trans[b]
                cur_coords = torch.inverse(lidar_aug_rots[b]).matmul(cur_coords.transpose(1, 0))

                # if not training:
                # visualize_feature_lidar(cur_coords.transpose(1, 0).detach().cpu().numpy(), "/home/bevfusion/viz_tumtraf_featmap/features/lidar/infra/auginversed")

                cur_coords = cur_coords.transpose(1, 0)

                # inverse registration matrix -> when registering we multiplied with v2i_rots -> to inverse it we multiply with inverse of v2i_rots

                cur_coords -= vehicle2infrastructure_trans[b]
                cur_coords = torch.inverse(vehicle2infrastructure_rots[b]).matmul(cur_coords.transpose(1, 0))

                # if not training:
                # visualize_feature_lidar(cur_coords.detach().cpu().numpy(), "/home/bevfusion/viz_tumtraf_featmap/features/lidar/infra/i2vval")

                # test_coords = cur_coords.transpose(1, 0)
                # test_coords += torch.add(vehicle2infrastructure_trans[b], lidar_aug_trans[b])
                # test_coords = (vehicle2infrastructure_rots[b].matmul(lidar_aug_rots[b])).matmul(test_coords.transpose(1,0))

                # visualize_feature_lidar(test_coords.transpose(1, 0).detach().cpu().numpy(), "/home/bevfusion/viz_tumtraf_featmap/features/lidar/infra/testrevert")

                # lidar2image
                cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
                cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)

                # get 2d coords
                dist = cur_coords[:, 2, :]
                cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
                cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

                # imgaug
                cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
                cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
                cur_coords = cur_coords[:, :2, :].transpose(1, 2)

                # normalize coords for grid sample
                cur_coords = cur_coords[..., [1, 0]]

                on_img = ((cur_coords[..., 0] < self.image_size[0]) & (cur_coords[..., 0] >= 0) & (cur_coords[..., 1] < self.image_size[1]) & (cur_coords[..., 1] >= 0))
                for c in range(on_img.shape[0]):
                    masked_coords = cur_coords[c, on_img[c]].long()
                    masked_dist = dist[c, on_img[c]]
                    depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist
            else:
                # we are in infrastructure, currently infra coord post transform
                cur_coords = points[b][:, :3]
                cur_img_aug_matrix = img_aug_matrix[b]
                cur_lidar2image = lidar2image[b]

                # inverse LiDAR aug (infra and vehicle augs are the same) -> Only have rotation and translation, applied similarly to original, so just follow original inversion
                cur_coords -= lidar_aug_trans[b]
                cur_coords = torch.inverse(lidar_aug_rots[b]).matmul(cur_coords.transpose(1, 0))

                # lidar2image
                cur_coords = cur_lidar2image[:, :3, :3].matmul(cur_coords)
                cur_coords += cur_lidar2image[:, :3, 3].reshape(-1, 3, 1)

                # get 2d coords
                dist = cur_coords[:, 2, :]
                cur_coords[:, 2, :] = torch.clamp(cur_coords[:, 2, :], 1e-5, 1e5)
                cur_coords[:, :2, :] /= cur_coords[:, 2:3, :]

                # imgaug
                cur_coords = cur_img_aug_matrix[:, :3, :3].matmul(cur_coords)
                cur_coords += cur_img_aug_matrix[:, :3, 3].reshape(-1, 3, 1)
                cur_coords = cur_coords[:, :2, :].transpose(1, 2)

                # normalize coords for grid sample
                cur_coords = cur_coords[..., [1, 0]]

                on_img = ((cur_coords[..., 0] < self.image_size[0]) & (cur_coords[..., 0] >= 0) & (cur_coords[..., 1] < self.image_size[1]) & (cur_coords[..., 1] >= 0))
                for c in range(on_img.shape[0]):
                    masked_coords = cur_coords[c, on_img[c]].long()
                    masked_dist = dist[c, on_img[c]]
                    depth[b, c, 0, masked_coords[:, 0], masked_coords[:, 1]] = masked_dist

        if self.vehicle:
            extra_rots_list = []
            extra_trans_list = []
            for b in range(batch_size):
                extra_rots_list.append(lidar_aug_rots[b].matmul(vehicle2infrastructure_rots[b]))  # aug x v2i because v2i happens first
                extra_trans_list.append(torch.add(lidar_aug_trans[b], vehicle2infrastructure_trans[b]))

            extra_rots = torch.stack(extra_rots_list)
            extra_trans = torch.stack(extra_trans_list)
        else:
            extra_rots = lidar_aug_matrix[..., :3, :3]
            extra_trans = lidar_aug_matrix[..., :3, 3]

        geom = self.get_geometry(
            camera2lidar_rots,
            camera2lidar_trans,
            intrins,
            post_rots,
            post_trans,
            extra_rots=extra_rots,
            extra_trans=extra_trans,
        )

        x = self.get_cam_feats(img, depth)
        x = self.bev_pool(geom, x)
        return x
