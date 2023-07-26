import argparse
import logging
import os
import pprint

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import numpy as np

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus2fp import DeepLabV3Plus
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter, update_teacher, get_adaptive_binary_mask
from util.dist_helper import setup_distributed
import time

parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True

def main():
    setup_seed(42)
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        tb_dir = args.save_path + '/{}'.format(time.strftime("%b%d_%d-%H-%M", time.localtime()))
        # generate_tb_dir = args.save_path + '/tb'
        writer = SummaryWriter(log_dir=tb_dir)
        
        os.makedirs(args.save_path, exist_ok=True)
    
    cudnn.enabled = True
    cudnn.benchmark = True

    model = DeepLabV3Plus(cfg)
    model_tea = DeepLabV3Plus(cfg)
    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': cfg['lr']},
                    {'params': [param for name, param in model.named_parameters() if 'backbone' not in name],
                    'lr': cfg['lr'] * cfg['lr_multi']}], lr=cfg['lr'], momentum=0.9, weight_decay=1e-4)
    
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model_tea = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_tea)
    model.cuda()
    model_tea.cuda()

    model     = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)
    model_tea = torch.nn.parallel.DistributedDataParallel(model_tea, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)

    for param in model_tea.parameters():
        param.detach_()

    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)
    mse_loss = nn.MSELoss().cuda(local_rank)

    trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path, num_aug=cfg['num_aug'], 
                             random_num_sampling=cfg['flag_use_random_num_sampling'])
    trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler_u)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best = 0.0
    previous_best_T = 0.0
    epoch = -1
    
    # if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
    #     checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
    #     checkpoint_t = torch.load(os.path.join(args.save_path, 't_latest.pth'))
    #     model.load_state_dict(checkpoint['model'])
    #     model_tea.load_state_dict(checkpoint_t['model'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     epoch = checkpoint['epoch']
    #     previous_best = checkpoint['previous_best']
    #     previous_best_T = checkpoint_t['previous_best'] 

        
    #     if rank == 0:
    #         logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}, Previous T best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best, previous_best_T))

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_w_fp = AverageMeter()
        total_loss_s_mse = AverageMeter()
        total_loss_fp_mse = AverageMeter()
        total_mask_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u, trainloader_u)

        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2),
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, ignore_mask_mix, _, _)) in enumerate(loader):
            # print(img_u_w.shape,img_u_w_mix.shape,'===========================================================')
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2, ignore_mask = img_u_s1.cuda(), img_u_s2.cuda(), ignore_mask.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()
            ignore_mask_mix = ignore_mask_mix.cuda()

            # with torch.no_grad():
            #     model_tea.eval()

            pred_u_w_mix_tea = model_tea(img_u_w_mix).detach()
            # conf_u_w_mix_tea = pred_u_w_mix_tea.softmax(dim=1).max(dim=1)[0]
            mask_u_w_mix_tea = pred_u_w_mix_tea.argmax(dim=1)

            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
                img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]

            model.train()

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            preds, preds_fp, preds_Fp = model(torch.cat((img_x, img_u_w)), True)
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            pred_u_w_fp = preds_fp[num_lb:]
            pred_u_w_Fp = preds_Fp[num_lb:]

            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2))).chunk(2)

            pred_u_w = pred_u_w.detach()
            adaptive_binary_mask = get_adaptive_binary_mask(pred_u_w)
            # conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)

            preds_tea = model_tea(img_u_w)
            # conf_u_w_tea = preds_tea.softmax(dim=1).max(dim=1)[0]
            mask_u_w_tea = preds_tea.argmax(dim=1)

# ==========================================================================================================
            preds_tea_mix1, preds_tea_mix2 = preds_tea.clone(), preds_tea.clone()
            preds_tea_mix1[cutmix_box1.unsqueeze(1).expand(preds_tea_mix1.shape) == 1] = \
                pred_u_w_mix_tea[cutmix_box1.unsqueeze(1).expand(pred_u_w_mix_tea.shape) == 1]
            preds_tea_mix2[cutmix_box2.unsqueeze(1).expand(preds_tea_mix2.shape) == 1] = \
                pred_u_w_mix_tea[cutmix_box2.unsqueeze(1).expand(pred_u_w_mix_tea.shape) == 1]
            
            adaptive_binary_mask_tea_mix1 = get_adaptive_binary_mask(preds_tea_mix1)
            adaptive_binary_mask_tea_mix2 = get_adaptive_binary_mask(preds_tea_mix2)
# ===========================================================================================================
            # mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = \
            #     mask_u_w_tea.clone(), conf_u_w_tea.clone(), ignore_mask.clone()
            # mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = \
            #     mask_u_w_tea.clone(), conf_u_w_tea.clone(), ignore_mask.clone()
            mask_u_w_cutmixed1, ignore_mask_cutmixed1 = \
            mask_u_w_tea.clone(), ignore_mask.clone()
            mask_u_w_cutmixed2, ignore_mask_cutmixed2 = \
                mask_u_w_tea.clone(), ignore_mask.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix_tea[cutmix_box1 == 1]
            # conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix_tea[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask_mix[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix_tea[cutmix_box2 == 1]
            # conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix_tea[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask_mix[cutmix_box2 == 1]

            loss_x = criterion_l(pred_x, mask_x)

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * ((adaptive_binary_mask_tea_mix1) & (ignore_mask_cutmixed1 != 255))
            loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()

            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * ((adaptive_binary_mask_tea_mix2) & (ignore_mask_cutmixed2 != 255))
            loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()

            loss_u_w_fp = criterion_u(pred_u_w_fp, mask_u_w)
            loss_u_w_fp = loss_u_w_fp * ((adaptive_binary_mask) & (ignore_mask != 255))
            loss_u_w_fp = loss_u_w_fp.sum() / (ignore_mask != 255).sum().item()

            loss_u_w_Fp = criterion_u(pred_u_w_Fp, mask_u_w)
            loss_u_w_Fp = loss_u_w_Fp * ((adaptive_binary_mask) & (ignore_mask != 255))
            loss_u_w_Fp = loss_u_w_Fp.sum() / (ignore_mask != 255).sum().item()

            pred_u_s1_norm = (pred_u_s1-torch.mean(pred_u_s1, dim=1, keepdim=True)) / torch.std(pred_u_s1, dim=1, keepdim=True) # Normalization
            pred_u_s2_norm = (pred_u_s2-torch.mean(pred_u_s2, dim=1, keepdim=True)) / torch.std(pred_u_s2, dim=1, keepdim=True)
            pred_u_f_norm  = (pred_u_w_fp-torch.mean(pred_u_w_fp, dim=1, keepdim=True)) / torch.std(pred_u_w_fp, dim=1, keepdim=True) 
            pred_u_F_norm  = (pred_u_w_Fp-torch.mean(pred_u_w_Fp, dim=1, keepdim=True)) / torch.std(pred_u_w_Fp, dim=1, keepdim=True) 

            loss_u_s_mse = mse_loss(pred_u_s1_norm, pred_u_s2_norm)
            loss_u_fp_mse = mse_loss(pred_u_f_norm, pred_u_F_norm)
            scale = 1/2.0
            if epoch < 1 and i < 1:
                keep_rate = 0
                loss =  scale*(loss_x + loss_u_w_fp * 0.25 + loss_u_w_Fp * 0.25 + (loss_u_s_mse + loss_u_fp_mse) * 0.01)
            else:
                keep_rate = 0.996
                loss =  scale*(loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.25 + loss_u_w_Fp * 0.25 + \
                        (loss_u_s_mse + loss_u_fp_mse) * 0.01)

            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
            total_loss_w_fp.update((loss_u_w_fp.item() + loss_u_w_Fp.item()) / 2.0)
            total_loss_s_mse.update(loss_u_s_mse.item())
            total_loss_fp_mse.update(loss_u_fp_mse.item())
            total_loss_x.update(loss_x.item())

            
            mask_ratio = ((adaptive_binary_mask) & (ignore_mask != 255)).sum().item() / \
                (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']

            # if epoch < 1 and i < 1:
            #     keep_rate = 0
                
            # # elif epoch < 4:
            # #     keep_rate = 0.996
            # # elif epoch < 20:
            # #     keep_rate = 0.9996
            # #     # Iters = iters
            # else:
            #     keep_rate = 0.996# - (0.9996 - 0.996) / ((cfg['epochs']-31) * len(trainloader_u)) * (iters - Iters)  
            update_teacher(model, model_tea, keep_rate)
            
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_s', (loss_u_s1.item() + loss_u_s2.item()) / 2.0, iters)
                writer.add_scalar('train/loss_w_fp', (loss_u_w_fp.item() + loss_u_w_Fp.item()) / 2.0, iters)
                writer.add_scalar('train/loss_s_mse', loss_u_s_mse.item(), iters)
                writer.add_scalar('train/loss_fp_mse', loss_u_fp_mse.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)
            
            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                # print(max_v)
                logger.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss w_fp: {:.3f}, Mask ratio: '
                            '{:.3f}'.format(i, total_loss.avg, total_loss_x.avg, total_loss_s.avg,
                                            total_loss_w_fp.avg, total_mask_ratio.avg))

        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'

        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg)
        mIOU_tea, iou_class2 = evaluate(model_tea, valloader, eval_mode, cfg)

        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                            'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}, MeanIoU_T: {:.2f}\n'.format(eval_mode, mIoU, mIOU_tea))
            
            writer.add_scalar('eval/mIoU', mIoU, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

        is_best = mIoU > previous_best
        is_best_T = mIOU_tea > previous_best_T
        previous_best = max(mIoU, previous_best)
        previous_best_T = max(mIOU_tea, previous_best_T)
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))
        if rank == 0:
            checkpoint = {
                'model': model_tea.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best_T,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 't_latest.pth'))
            if is_best_T:
                torch.save(checkpoint, os.path.join(args.save_path, 't_best.pth'))


if __name__ == '__main__':
    main()
