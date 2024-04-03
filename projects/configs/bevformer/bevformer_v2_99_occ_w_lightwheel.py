_base_ = ['./bevformer_base_occ_w_lightwheel.py']

model = dict(
    img_backbone=dict(_delete_=True, type='VoVNet', spec_name='V-99-eSE', norm_eval=True, frozen_stages=1, input_ch=3, out_features=['stage3', 'stage4', 'stage5']),
    img_neck=dict(_delete_=True, type='FPN', in_channels=[512, 1024, 2048], out_channels=256, start_level=1, add_extra_convs='on_output', num_outs=4, relu_before_extra_convs=True),
)

load_from = 'dd3d_det_final.pth'
