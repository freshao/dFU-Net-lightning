import argparse


def get_opts():
    parser = argparse.ArgumentParser()

    parser.add_argument('--debug', type=bool, default=False,
                        help='debug')
    parser.add_argument('--test', type=bool, default=False,
                        help='switch for test')
    parser.add_argument('--num_gpus', type=int, default=6,
                        help='number of gpus')
    parser.add_argument('--batch', type=int, default=8,
                        help='number of batch to be used in training')
    parser.add_argument('--slice', type=int, default=1,
                        help='number of slice to be used in training')
    parser.add_argument('--strategy', type=str, default='ddp',
                        help='strategy to be used in training')
    parser.add_argument('--num_classes', type=int,
                        default=1,
                        help='number of classes to be used in training')
    parser.add_argument('--ckpt_path', type=str,
                        # default="/opt/data/private/data/ckpts/ce-resimg+LITS/epoch=23.ckpt",
                        default="",
                        help='pretrained checkpoint path to load')
    parser.add_argument('--load_params', type=bool,
                        default=False,
                        # choices=['params', 'model'],
                        help='which type to load .pth')
    parser.add_argument('--imgs_dir', type=str,
                        # default='/home/sj/workspace/m/MA_NET/LITS/train/train_small/',
                        # default='/opt/data/private/data/chao_data/Train_Sets/',
                        default='/opt/data/default/medical/data/LITS/LITS/img_dir/train_slice_1_1/',
                        # default='/opt/data/private/data/LITS/LITS/img_dir/train_slice_5_1/',
                        # default='/opt/data/default/medical/data/LITS/LITS/img_dir/train_slice_5_1/',
                        # default='/opt/data/default/medical/data/chao_data/img_dir/train_slice_5_1/',
                        # default='/data/LITS/img_dir/train_slice_5/',
                        # default='/data/chao_data/Train_Sets/',
                        # default='/data/chao_data/img_dir/train',
                        help='image directory of history dataset')
    parser.add_argument('--masks_dir', type=str,
                        # default='/home/sj/workspace/m/MA_NET/LITS/train/target_small/',
                        default='/opt/data/default/medical/data/LITS/LITS/ann_dir/train_slice_1_1/',
                        # default='/opt/data/private/data/LITS/LITS/ann_dir/train_slice_5_1/',
                        # default='/opt/data/default/medical/data/LITS/LITS/ann_dir/train_slice_5_1/',
                        # default='/opt/data/default/medical/data/chao_data/ann_dir/train_slice_5_1/',
                        # default='/data/LITS/ann_dir/train_slice_5/',
                        # default='/opt/data/private/data/chao_data/Train_Sets/',
                        # default='/data/chao_data/Train_Sets/',
                        # default='/data/chao_data/ann_dir/train',
                        help='image directory of history masks dataset')
    parser.add_argument('--test_imgs_dir', type=str,
                        # default='/home/sj/workspace/m/MA_NET/LITS/train/train_small/',
                        # default='/opt/data/default/medical/data/chao_data/img_dir/train_slice_1_1/',
                        # default='/opt/data/default/medical/data/LITS/LITS/img_dir/val_slice_5_1/',
                        default='/opt/data/default/medical/data/LITS/LITS/img_dir/val_slice_1_1/',
                        # default='/opt/data/default/medical/data/chao_data/img_dir/train_slice_5_1/',
                        # default='/data/LITS/img_dir/val_slice_5/',
                        # default='/opt/data/private/data/chao_data/Valid_Sets/',
                        # default='/data/chao_data/Valid_Sets/',
                        # default='/data/chao_data/img_dir/val',
                        help='image directory of history test dataset')
    parser.add_argument('--test_masks_dir', type=str,
                        # default='/home/sj/workspace/m/MA_NET/LITS/train/target_small/',
                        # default='/opt/data/default/medical/data/chao_data/ann_dir/train_slice_1_1/',
                        # default='/opt/data/default/medical/data/LITS/LITS/ann_dir/val_slice_5_1/',
                        default='/opt/data/default/medical/data/LITS/LITS/ann_dir/val_slice_1_1/',
                        # default='/opt/data/default/medical/data/chao_data/ann_dir/train_slice_5_1/',
                        # default='/data/LITS/ann_dir/val_slice_5/',
                        # default='/opt/data/private/data/chao_data/Valid_Sets/',
                        # default='/data/chao_data/Valid_Sets/',
                        # default='/data/chao_data/ann_dir/val',
                        help='image directory of history test masks dataset')

    parser.add_argument('--pred_imgs_save_dir', type=str,
                        # default='/data/LITS/img_dir/val_slice_5/',
                        # default='/opt/data/private/data/LITS/result/',
                        # default='/opt/data/private/data/LITS/result/',
                        default='/opt/data/default/medical/data/LITS/result/',
                        help='image directory of history test dataset')

    parser.add_argument('--prefixes_to_ignore', nargs='+', type=str, default=['loss'],
                        help='the prefixes to ignore in the checkpoint state dict')
    parser.add_argument('--val_percent', type=float, default=0.1,
                        help='number of validation percent to be used in training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='number of lr to be used in training')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='learning rate momentum')
    parser.add_argument('--num_epochs', type=int, default=110,
                        help='number of training epochs')
    parser.add_argument('--exp_name', type=str, default='dense-resimg-5+lits',
                        help='experiment name')
    parser.add_argument('--use_amp', default=False, action="store_true",
                        help='use mixed precision training (NOT SUPPORTED!)')
    # parser.add_argument('--dataset_name', type=str, default='dtu',
    #                     choices=['dtu', 'blendedmvs'],
    #                     help='which dataset to history/val')

    # parser.add_argument('--n_views', type=int, default=3,
    #                     help='number of views (including ref) to be used in training')
    # parser.add_argument('--levels', type=int, default=3, choices=[3],
    #                     help='number of FPN levels (fixed to be 3!)')
    # parser.add_argument('--depth_interval', type=float, default=2.65,
    #                     help='depth interval for the finest level, unit in mm')
    # parser.add_argument('--n_depths', nargs='+', type=int, default=[8, 32, 48],
    #                     help='number of depths in each level')
    # parser.add_argument('--interval_ratios', nargs='+', type=float, default=[1.0, 2.0, 4.0],
    #                     help='depth interval ratio to multiply with --depth_interval in each level')
    # parser.add_argument('--num_groups', type=int, default=1, choices=[1, 2, 4, 8],
    #                     help='number of groups in groupwise correlation, must be a divisor of 8')
    # parser.add_argument('--loss_type', type=str, default='sl1',
    #                     choices=['sl1'],
    #                     help='loss to use')
    #
    # parser.add_argument('--batch_size', type=int, default=1,
    #                     help='batch size')
    #
    #
    #
    # parser.add_argument('--optimizer', type=str, default='sgd',
    #                     help='optimizer type',
    #                     choices=['sgd', 'adam', 'radam', 'ranger'])
    # parser.add_argument('--lr', type=float, default=1e-3,
    #                     help='learning rate')
    # parser.add_argument('--momentum', type=float, default=0.9,
    #                     help='learning rate momentum')
    # parser.add_argument('--weight_decay', type=float, default=1e-5,
    #                     help='weight decay')
    # parser.add_argument('--lr_scheduler', type=str, default='steplr',
    #                     help='scheduler type',
    #                     choices=['steplr', 'cosine', 'poly'])
    # #### params for warmup, only applied when optimizer == 'sgd' or 'adam'
    # parser.add_argument('--warmup_multiplier', type=float, default=1.0,
    #                     help='lr is multiplied by this factor after --warmup_epochs')
    # parser.add_argument('--warmup_epochs', type=int, default=0,
    #                     help='Gradually warm-up(increasing) learning rate in optimizer')
    # ###########################
    # #### params for steplr ####
    # parser.add_argument('--decay_step', nargs='+', type=int, default=[20],
    #                     help='scheduler decay step')
    # parser.add_argument('--decay_gamma', type=float, default=0.1,
    #                     help='learning rate decay amount')
    # ###########################
    # #### params for poly ####
    # parser.add_argument('--poly_exp', type=float, default=0.9,
    #                     help='exponent for polynomial learning rate decay')
    # ###########################

    return parser.parse_args()