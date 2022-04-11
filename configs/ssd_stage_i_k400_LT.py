cfg = dict(
    data_root_train='data/kinetics400/videos_train/',
    data_root_val='data/kinetics400/videos_val/',
    pretrained_clip='pretrained/RN50.pt',

    use_mcloader=True,
    data_set='Kinetics',
    dataset='Kinetics',
    drop_last=True,
    index_bias=0,

    train_list_file='data/kinetics400/kinetics_video_train_list.txt',
    val_list_file='data/kinetics400/kinetics_video_val_list.txt',

    num_segments=8,
    new_length=1,
    is_video=True,

    io_backend='petrel',
    find_unused_parameters=True,
    num_classes=400,
    only_video=True,
    smoothing=False,
    broadcast_bn_buffer=True
)