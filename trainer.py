import argparse
import logging
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import DiceLoss
from torchvision import transforms


def _compute_mean_dice(logits, labels, num_classes):
    """Compute validation Dice over foreground classes (1..num_classes-1).

    Medical-style handling for empty-foreground cases:
    - gt empty & pred empty   -> Dice = 1.0 (do not penalize)
    - gt empty & pred nonempty -> Dice = 0.0 (false positive penalty)
    - gt nonempty             -> standard Dice
    """
    preds = torch.argmax(torch.softmax(logits, dim=1), dim=1)
    dices = []
    for c in range(1, num_classes):
        pred_c = (preds == c).float()
        label_c = (labels == c).float()
        pred_sum = torch.sum(pred_c)
        label_sum = torch.sum(label_c)

        if label_sum == 0:
            if pred_sum == 0:
                dices.append(torch.tensor(1.0, device=logits.device))
            else:
                dices.append(torch.tensor(0.0, device=logits.device))
            continue

        intersect = torch.sum(pred_c * label_c)
        dice = (2.0 * intersect + 1e-5) / (pred_sum + label_sum + 1e-5)
        dices.append(dice)

    if not dices:
        return 1.0
    return float(torch.mean(torch.stack(dices)).item())

def trainer_synapse(args, model, snapshot_path):
    from datasets.dataset_synapse import Synapse_dataset, RandomGenerator, ResizeGenerator
    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size * args.n_gpu
    # max_iterations = args.max_iterations
    db_train = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="train",
                               transform=transforms.Compose(
                                   [RandomGenerator(output_size=[args.img_size, args.img_size])]))
    val_list_path = os.path.join(args.list_dir, "val.txt")
    if not os.path.exists(val_list_path):
        raise FileNotFoundError("Validation list not found: {}. Please provide val.txt for per-epoch validation.".format(val_list_path))
    db_val = Synapse_dataset(base_dir=args.root_path, list_dir=args.list_dir, split="val",
                             transform=transforms.Compose(
                                 [ResizeGenerator(output_size=[args.img_size, args.img_size])]))

    print("The length of train set is: {}".format(len(db_train)))
    print("The length of val set is: {}".format(len(db_val)))

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    if args.n_gpu > 1:
        model = nn.DataParallel(model)
    model.train()
    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    writer = SummaryWriter(snapshot_path + '/log')
    iter_num = 0
    max_epoch = args.max_epochs
    max_iterations = args.max_epochs * len(trainloader)  # max_epoch = max_iterations // len(trainloader) + 1
    logging.info("{} iterations per epoch. {} max iterations ".format(len(trainloader), max_iterations))
    best_performance = -1.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):
            image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
            outputs = model(image_batch)
            loss_ce = ce_loss(outputs, label_batch[:].long())
            loss_dice = dice_loss(outputs, label_batch, softmax=True)
            loss = 0.5 * loss_ce + 0.5 * loss_dice
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_

            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)

            logging.info('iteration %d : loss : %f, loss_ce: %f' % (iter_num, loss.item(), loss_ce.item()))

            if iter_num % 20 == 0:
                image = image_batch[1, 0:1, :, :]
                image = (image - image.min()) / (image.max() - image.min())
                writer.add_image('train/Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(outputs, dim=1), dim=1, keepdim=True)
                writer.add_image('train/Prediction', outputs[1, ...] * 50, iter_num)
                labs = label_batch[1, ...].unsqueeze(0) * 50
                writer.add_image('train/GroundTruth', labs, iter_num)

        # Validation each epoch and save best model by val Dice.
        model.eval()
        val_dice_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for sampled_batch in valloader:
                image_batch, label_batch = sampled_batch['image'], sampled_batch['label']
                image_batch, label_batch = image_batch.cuda(), label_batch.cuda()
                outputs = model(image_batch)
                val_dice_sum += _compute_mean_dice(outputs, label_batch, num_classes)
                val_count += 1

        val_dice = val_dice_sum / max(val_count, 1)
        writer.add_scalar('val/mean_dice', val_dice, epoch_num + 1)
        logging.info('epoch %d : val_mean_dice : %f' % (epoch_num + 1, val_dice))

        if val_dice > best_performance:
            best_performance = val_dice
            save_mode_path = os.path.join(snapshot_path, 'best_model.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save best model to {} (val_mean_dice={:.6f})".format(save_mode_path, best_performance))

        if epoch_num >= max_epoch - 1:
            save_mode_path = os.path.join(snapshot_path, 'last_model.pth')
            torch.save(model.state_dict(), save_mode_path)
            logging.info("save model to {}".format(save_mode_path))
            iterator.close()
            break

    writer.close()
    return "Training Finished!"
