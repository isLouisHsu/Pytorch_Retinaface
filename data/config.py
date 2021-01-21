# config.py

cfg_mnet = {
    'name': 'mobilenet0.25',
    'in_channels': 3,
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': "./weights/pretrained/mobilenetV1X0.25_pretrain.tar",
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

cfg_re50 = {
    'name': 'Resnet50',
    'in_channels': 3,
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 8,
    'ngpu': 1,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 840,
    'pretrain': "/home/louishsu/.cache/torch/hub/checkpoints/resnet50-19c8e357.pth",
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 256,
    'out_channel': 256
}

# --------------------------------------------------------------------------------------

cfg_re18 = {
    'name': 'Resnet18',
    'in_channels': 3,
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 8,
    'ngpu': 1,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 480,
    'pretrain': "/home/louishsu/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth",
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 64,
    'out_channel': 256
}

cfg_re34 = {
    'name': 'Resnet34',
    'in_channels': 3,
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 8,
    'ngpu': 1,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 480,
    'pretrain': "/home/louishsu/.cache/torch/hub/checkpoints/resnet34-333f7ec4.pth",
    'return_layers': {'layer2': 1, 'layer3': 2, 'layer4': 3},
    'in_channel': 64,
    'out_channel': 256
}

cfg_eff_b0 = {
    'name': 'Efficientnet-b0',
    'in_channels': 3,
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 8,
    'ngpu': 1,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 480,
    'pretrain': "/home/louishsu/.cache/torch/hub/checkpoints/tf_efficientnet_b0_ns-c0e6a31c.pth",
    'return_layers': {'2': 3, '4': 5, '6': 7},
    'in_channel': None,
    'out_channel': 256
}

cfg_eff_b4 = {
    'name': 'Efficientnet-b4',
    'in_channels': 3,
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 4,
    'ngpu': 1,
    'epoch': 100,
    'decay1': 70,
    'decay2': 90,
    'image_size': 480,
    'pretrain': "/home/louishsu/.cache/torch/hub/checkpoints/tf_efficientnet_b4_ns-d6313a46.pth",
    'return_layers': {'2': 3, '4': 5, '6': 7},
    'in_channel': None,
    'out_channel': 256
}