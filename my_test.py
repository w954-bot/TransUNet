import argparse
import logging
import os
import random
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets.dataset_synapse import Synapse_dataset, ResizeGenerator
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
from torchvision import transforms


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str, default='datasets/EEC2022/train_npz',
                        help='root dir for npz data')
    parser.add_argument('--list_dir', type=str, default='lists/EEC2022',
                        help='dir containing test/val/train txt files')
    parser.add_argument('--dataset', type=str, default='Custom',
                        help='experiment name shown in logs/snapshot path')
    parser.add_argument('--split', type=str, default='test', choices=['test', 'val', 'train'],
                        help='which list split to evaluate')
    parser.add_argument('--list_name', type=str, default='',
                        help='optional explicit list filename, e.g. test.txt; if empty use <split>.txt')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='number of classes including background')
    parser.add_argument('--max_iterations', type=int, default=30000,
                        help='used to build snapshot name, keep same as train')
    parser.add_argument('--max_epochs', type=int, default=150,
                        help='used to build snapshot name, keep same as train')
    parser.add_argument('--batch_size', type=int, default=24,
                        help='used to build snapshot name, keep same as train')
    parser.add_argument('--img_size', type=int, default=448,
                        help='input size used in train')
    parser.add_argument('--n_skip', type=int, default=0,
                        help='skip number used in train')
    parser.add_argument('--vit_name', type=str, default='ViT-B_16',
                        help='model name used in train')
    parser.add_argument('--vit_patches_size', type=int, default=16,
                        help='vit patch size used in train')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='deterministic inference')
    parser.add_argument('--base_lr', type=float, default=0.01,
                        help='used to build snapshot name, keep same as train')
    parser.add_argument('--seed', type=int, default=1234,
                        help='random seed')
    parser.add_argument('--snapshot_path', type=str, default='',
                        help='optional explicit checkpoint directory; if empty, auto-build from train naming')
    parser.add_argument('--checkpoint_name', type=str, default='best_model.pth',
                        help='checkpoint file name inside snapshot_path')
    return parser.parse_args()


def compute_case_mean_dice(pred, label, num_classes):
    """Per-case mean Dice over foreground classes with medical empty-foreground handling."""
    dices = []
    for c in range(1, num_classes):
        pred_c = (pred == c)
        label_c = (label == c)
        pred_sum = pred_c.sum()
        label_sum = label_c.sum()

        if label_sum == 0:
            dices.append(1.0 if pred_sum == 0 else 0.0)
            continue

        intersect = np.logical_and(pred_c, label_c).sum()
        dice = (2.0 * intersect + 1e-5) / (pred_sum + label_sum + 1e-5)
        dices.append(float(dice))

    if len(dices) == 0:
        return 1.0
    return float(np.mean(dices))


def build_snapshot_path(args):
    if args.snapshot_path:
        return args.snapshot_path

    exp = 'TU_' + args.dataset + str(args.img_size)
    snapshot_path = "../model/{}/{}".format(exp, 'TU')
    snapshot_path = snapshot_path + '_pretrain'
    snapshot_path += '_' + args.vit_name
    snapshot_path = snapshot_path + '_skip' + str(args.n_skip)
    snapshot_path = snapshot_path + '_vitpatch' + str(args.vit_patches_size) if args.vit_patches_size != 16 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.max_iterations)[0:2] + 'k' if args.max_iterations != 30000 else snapshot_path
    snapshot_path = snapshot_path + '_epo' + str(args.max_epochs) if args.max_epochs != 30 else snapshot_path
    snapshot_path = snapshot_path + '_bs' + str(args.batch_size)
    snapshot_path = snapshot_path + '_lr' + str(args.base_lr) if args.base_lr != 0.01 else snapshot_path
    snapshot_path = snapshot_path + '_' + str(args.img_size)
    snapshot_path = snapshot_path + '_s' + str(args.seed) if args.seed != 1234 else snapshot_path
    return snapshot_path


def main():
    args = parse_args()

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    config_vit = CONFIGS_ViT_seg[args.vit_name]
    config_vit.n_classes = args.num_classes
    config_vit.n_skip = args.n_skip
    config_vit.patches.size = (args.vit_patches_size, args.vit_patches_size)
    if args.vit_name.find('R50') != -1:
        config_vit.patches.grid = (
            int(args.img_size / args.vit_patches_size),
            int(args.img_size / args.vit_patches_size),
        )

    model = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes).cuda()

    snapshot_path = build_snapshot_path(args)
    checkpoint_path = os.path.join(snapshot_path, args.checkpoint_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError("Checkpoint not found: {}".format(checkpoint_path))

    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    list_name = args.list_name if args.list_name else args.split + '.txt'
    split_list_path = os.path.join(args.list_dir, list_name)
    if not os.path.exists(split_list_path):
        raise FileNotFoundError("List not found: {}".format(split_list_path))

    db_eval = Synapse_dataset(
        base_dir=args.root_path,
        list_dir=args.list_dir,
        split=args.split,
        transform=transforms.Compose([ResizeGenerator(output_size=[args.img_size, args.img_size])]),
    )
    evalloader = DataLoader(db_eval, batch_size=1, shuffle=False, num_workers=1)

    log_folder = './test_log/test_log_' + args.dataset + str(args.img_size)
    os.makedirs(log_folder, exist_ok=True)
    snapshot_name = os.path.basename(snapshot_path)
    logging.basicConfig(
        filename=os.path.join(log_folder, snapshot_name + '_my_test.txt'),
        level=logging.INFO,
        format='[%(asctime)s.%(msecs)03d] %(message)s',
        datefmt='%H:%M:%S'
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    logging.info("Loading checkpoint: {}".format(checkpoint_path))
    logging.info("{} cases: {}".format(args.split, len(db_eval)))

    case_dices = []
    with torch.no_grad():
        for i_batch, sampled_batch in tqdm(enumerate(evalloader), total=len(evalloader)):
            image_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()
            outputs = model(image_batch)
            preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)

            pred_np = preds.squeeze(0).cpu().numpy()
            label_np = label_batch.squeeze(0).cpu().numpy()
            case_name = sampled_batch['case_name'][0]

            case_dice = compute_case_mean_dice(pred_np, label_np, args.num_classes)
            case_dices.append(case_dice)
            logging.info('idx %d case %s mean_dice %f', i_batch, case_name, case_dice)

    mean_dice = float(np.mean(case_dices)) if case_dices else 0.0
    logging.info('%s mean_dice: %f', args.split, mean_dice)
    print('{} mean_dice: {:.6f}'.format(args.split, mean_dice))


if __name__ == '__main__':
    main()
