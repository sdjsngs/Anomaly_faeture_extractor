# 使用C3D以及I3D 抽取 shanghaitech 以及 ucf crime 的 feature

shanghaitech 数据 存放位置 D:\xxx\shanghaitech 其中train部分为video test 部分为jpg
ucf crime 数据存放位置  D:\xxx\UCF_Crime tain 和test

细节部分参考论文以及 mmaction  

## C3D  模型文件：   数据的预处理部分
    先 resize -> 128, 171 
    再 center crop -> 112 
    均值和方差  img_norm_cfg = dict(mean=[104, 117, 128], std=[1, 1, 1], to_bgr=False)


# I3D 部分  模型问题

    直接 reszie -> 224  
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
