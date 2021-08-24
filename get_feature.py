"""
抽取 sh的C3D feature
kenitice mean = [0.43216, 0.394666, 0.37645] and std = [0.22803, 0.22145, 0.216989]

"""
from pre_train_model.C3D_STD import C3DBackbone as C3D_net
from pre_train_model.I3D_STD import I3D as I3D_net
import torch
import torch.nn as nn

import os
import glob
import cv2
import numpy as np
import transforms
from icecream import ic

MEAN_and_STD = {
    "c3d_mean": [104.0, 117.0, 128.0],
    "c3d_std": [1.0, 1.0, 1.0],
    "i3d_mean": [123.675, 116.28, 103.53],
    "i3d_std": [58.395, 57.12, 57.375],
}

C3D_pth = r"./checkpoint/C3D_Sport1M.pth"
I3D_pth = r"./checkpoint/I3D_model_rgb.pth"


def load_C3D_model_pickle(model, c3d_pickle, fc_layer="fc6"):
    """
    c3d model pickle version

    :return:
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pickle_dict = torch.load(c3d_pickle)

    model_dict = model.state_dict()

    load_model_dict = {k: v for k, v in pickle_dict.items() if k in model_dict.keys()}

    model.load_state_dict(load_model_dict)

    # if fc_layer == 'fc6':
    #     model = nn.Sequential(*list(model.children())[:-5])

    # load model to gpu
    model.to(device)

    return model


def load_C3D_MIST_Pth(model, c3d_pth):
    """"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pth_dict = torch.load(c3d_pth)
    # replace
    pth_replace_dict = replace_mist_checkpoint(pth_dict)

    model_dict = model.state_dict()

    load_model_dict = {k: v for k, v in pth_replace_dict.items() if k in model_dict.keys()}

    model.load_state_dict(load_model_dict)

    # if fc_layer == 'fc6':
    #     model = nn.Sequential(*list(model.children())[:-5])

    # load model to gpu
    model.to(device)

    return model


def replace_mist_checkpoint(model_dict):
    """
    replace item backbone -> ""
    conv1a -> conv1
    conv2a -> conv2

    :return:
    """
    replace_dict = {
        k.replace("backbone.", "").replace("1a", "1").replace("2a", "2"): v for k, v in model_dict.items()
    }
    return replace_dict


def load_model_parm(model, model_pth, ):
    """
    load pre-train model parm
    :return:
    """
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    pth_dict = torch.load(model_pth)

    model_dict = model.state_dict()

    load_model_dict = {k: v for k, v in pth_dict.items() if k in model_dict.keys()}

    model.load_state_dict(load_model_dict)

    # model.to(device)
    #
    # return model


def load_c3d_pretrained_model(net, checkpoint_path, name=None):
    checkpoint = torch.load(checkpoint_path)
    state_dict = net.state_dict()
    base_dict = {}
    checkpoint_keys = checkpoint.keys()
    for k, v in state_dict.items():
        for _k in checkpoint_keys:

            if k in _k:
                base_dict[k] = checkpoint[_k]

    state_dict.update(base_dict)
    net.load_state_dict(state_dict)
    print('model load pretrained weights')


def load_video_get_feature(video_path, pre_train_model, transform):
    """"""

    video_cap = cv2.VideoCapture(video_path)

    frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    # feature temporal length =16
    temporal_length = 16
    # how many feature ? drop the last 15 (at most)
    feature_num = frame_count // temporal_length

    if feature_num < 0:
        raise NotImplementedError(
            "too short for this video {}".format(video_path)
        )

    # resize?
    clip_buffer = np.zeros((16, frame_height, frame_width, 3), np.dtype("float32"))
    ret = True
    feature_count = 0
    temporal_count = 0
    features = []
    while ret:
        ret, frame = video_cap.read()
        if temporal_count == 16:
            # make feature
            out_feature = feature_exactor(clip_buffer, pre_train_model, transform)
            features.append(out_feature)

            clip_buffer = np.zeros((16, frame_height, frame_width, 3), np.dtype("float32"))

            temporal_count = 0
            feature_count += 1

        if not ret or feature_count == feature_num:
            print("finish this video get feature count: {}".format(feature_count))
            print("feature num should be:{}".format(feature_num))
            if feature_count != feature_num:
                ic(video_path)
            break



        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        clip_buffer[temporal_count] = frame
        temporal_count += 1

    return np.array(features)


def load_list_imgs(img_list, ):
    filedata = []

    for single_img in img_list:
        img = cv2.cvtColor(cv2.imread(single_img), cv2.COLOR_BGR2RGB)

        filedata.append(img)

    return np.array(filedata, dtype=np.float32)


def load_img_get_feature(img_folder, pre_train_model, transform):
    """"""
    img_paths = glob.glob(
        os.path.join(
            img_folder, "*.jpg"
        )
    )
    temporal_len = 16
    feature_num = len(img_paths) // temporal_len

    features = []
    for i in range(feature_num):
        sub_img = img_paths[i * temporal_len:(i + 1) * temporal_len]
        # load imgs
        clip = load_list_imgs(sub_img)
        out_feature = feature_exactor(clip, pre_train_model, transform)
        features.append(out_feature)
    ic(img_folder.split("\\")[-1], feature_num, len(features))
    return np.array(features)


def feature_exactor(feature, model, transform):
    """

    :param model:
    :param feature: numpy array type in t,h,w,c
    :return:
    """
    # build transform
    # feature=torch.from_numpy(feature).permute(3,0,1,2).unsqueeze(dim=0)
    # feature = feature.permute(1, 0, 2, 3).unsqueeze(dim=0)
    # feature = feature.cuda().float()

    feature = transform(feature)
    if feature.ndim == 4:
        feature = feature.unsqueeze(dim=0)

    model.eval()
    with torch.no_grad():  # just do one feature
        # do feature exactor
        out_feature = model(feature.cuda())
        out_feature = out_feature.detach().cpu().numpy()
    return out_feature


def get_dict(txt_file):
    txt_dict = {}
    with open(txt_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            video_name = line.split(",")[0]
            video_class = line.split(",")[1]
            txt_dict[video_name] = video_class
    return txt_dict


def sh_feature(sh_path, sh_feature_save_path, model, transform):
    normal_dict = {
        "0": "Normal",
        "1": "Abnormal"
    }
    sh_new_train_txt = r"./SH_new_split/SH_Train_New.txt"
    sh_new_test_txt = r"./SH_new_split/SH_Test_New.txt"

    train_dict = get_dict(sh_new_train_txt)
    test_dict = get_dict(sh_new_test_txt)

    origin_sh_train_data = glob.glob(
        os.path.join(
            sh_path, "training/videos", "*.avi"
        )
    )
    origin_sh_test_data = glob.glob(
        os.path.join(
            sh_path, "testing/frames", "*"
        )
    )

    for one_video in origin_sh_train_data:
        video_name = one_video.split("\\")[-1].split(".")[0]
        if video_name in train_dict.keys():
            save_folder = "train"
            video_class = normal_dict[str(train_dict[video_name])]
        elif video_name in test_dict.keys():
            save_folder = "test"
            video_class = normal_dict[str(test_dict[video_name])]
        else:
            raise NotImplementedError(
                "miss the video name:{}".format(video_name)
            )
        os.makedirs(
            os.path.join(
                sh_feature_save_path, save_folder, video_class
            ), exist_ok=True
        )

        feature_path = os.path.join(
            sh_feature_save_path, save_folder, video_class, video_name + ".npy"
        )
        if os.path.isfile(feature_path):
            continue
        feature = load_video_get_feature(
            one_video, model, transform
        )

        np.save(
            feature_path, feature
        )
    for one_img_folder in origin_sh_test_data:
        video_name = one_img_folder.split("\\")[-1]
        if video_name in train_dict.keys():
            save_folder = "train"
            video_class = normal_dict[str(train_dict[video_name])]
        elif video_name in test_dict.keys():
            save_folder = "test"
            video_class = normal_dict[str(test_dict[video_name])]
        else:
            raise NotImplementedError(
                "miss the video name:{}".format(video_name)
            )
        os.makedirs(
            os.path.join(
                sh_feature_save_path, save_folder, video_class
            ), exist_ok=True
        )

        feature_path = os.path.join(
            sh_feature_save_path, save_folder, video_class, video_name + ".npy"
        )
        if os.path.isfile(feature_path):
            continue
        feature = load_img_get_feature(one_img_folder, model, transform)

        np.save(
            feature_path, feature
        )


def ucf_feature(ucf_path, ucf_save_path, model, transform):
    Abnormal_type = ['Abuse', 'Arrest', 'Arson', 'Assault', 'Burglary',
                     'Explosion', 'Fighting', 'RoadAccidents', 'Robbery',
                     'Shooting', 'Shoplifting', 'Stealing', 'Vandalism', ]
    ucf_train_normal_list = glob.glob(
        os.path.join(
            ucf_path, "train", "Normal", "*.mp4"
        )
    )
    ucf_train_abnormal_list = []

    for abnormal_class in Abnormal_type:
        ucf_train_abnormal_list.extend(
            glob.glob(
                os.path.join(
                    ucf_path, "train", abnormal_class, "*.mp4"
                )
            )
        )

    ucf_test_normal_list = glob.glob(
        os.path.join(
            ucf_path, "test", "Normal", "*.mp4"
        )
    )

    ucf_test_abnormal_list = []

    for abnormal_class in Abnormal_type:
        ucf_test_abnormal_list.extend(
            glob.glob(
                os.path.join(
                    ucf_path, "test", abnormal_class, "*.mp4"
                )
            )
        )

    ucf_all_file = ucf_train_normal_list+ucf_train_abnormal_list + ucf_test_abnormal_list  + ucf_test_normal_list
    for single_video in ucf_all_file:
        video_name = single_video.split("\\")[-1].split(".")[0]
        abnormal_flag = single_video.split("\\")[-2]
        if abnormal_flag in Abnormal_type:
            abnormal_flag = "Abnormal"
        else:
            abnormal_flag = "Normal"

        train_flag = single_video.split("\\")[-3]
        save_folder = os.path.join(
            ucf_save_path, train_flag, abnormal_flag
        )
        os.makedirs(save_folder, exist_ok=True)
        feature_path = os.path.join(
            save_folder, video_name + ".npy"
        )
        if os.path.isfile(feature_path):
            continue
        feature = load_video_get_feature(
            single_video, model, transform
        )
        np.save(
            feature_path, feature
        )


if __name__ == "__main__":
    print()
    # demo_video_name = "01_001"
    #
    # compare_feature = r"E:\datasets\shanghaitech_I3D\dataset\features_video\i3d\rgb\01_001/feature.npy"
    #
    # demo_video = r"D:\dataset\shanghaitech\training\videos/01_001.avi"

    # step 1 load one model
    model_name = "C3D"

    if model_name in ["C3D"]:
        net = C3D_net().cuda()
        # load  checkpoint
        trans = transforms.Compose([transforms.Resize([128, 171]),
                                    transforms.CenterCrop(112),
                                    transforms.ClipToTensor(div_255=False),
                                    transforms.Normalize(mean=MEAN_and_STD["c3d_mean"], std=MEAN_and_STD["c3d_std"])])
        load_c3d_pretrained_model(
            net, C3D_pth
        )

    elif model_name in ["I3D"]:
        net = I3D_net().cuda()
        trans = transforms.Compose([transforms.Resize([224, 224]),
                                    transforms.ClipToTensor(div_255=False),
                                    transforms.Normalize(mean=MEAN_and_STD["i3d_mean"], std=MEAN_and_STD["i3d_std"])])
        load_model_parm(net, I3D_pth)
    else:
        raise NotImplementedError(
            "check the model name:{}".format(model_name)
        )

    # make exactor

    sh_path = r"D:\dataset\shanghaitech"
    sh_save_path = r"E:\datasets\SH_C3D"
    sh_feature(
        sh_path, sh_save_path, net, trans
    )

    ucf_path = r"D:\dataset\UCF_Crime"
    ucf_save_path = r"E:\datasets\UCF_Crime_C3D"
    ucf_feature(ucf_path, ucf_save_path, net, trans)
