import glob
import os
import copy

import torch
import tqdm
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist

from pcdet.models.model_utils.dsnorm import set_ds_target
from pcdet.models.model_utils import mmd
from pcdet.config import cfg

def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False,
                    cur_epoch=0, total_epochs=0, retain_graph=False,
                    ):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    cur_train_dict = {
        'cur_epoch': cur_epoch,
        'total_epochs': total_epochs,
        'total_it_each_epoch': total_it_each_epoch,
    }

    if type(model) == torch.nn.parallel.distributed.DistributedDataParallel:
        ch_model = model.module
        dist_train = True
    else:
        ch_model = model
        dist_train = False
    num_class = ch_model.num_class
    pseudo_match_cls = [torch.zeros((2, 0)).cuda() for _ in range(num_class)]
    for cur_it in range(total_it_each_epoch):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()
        optimizer.zero_grad()

        cur_train_dict['cur_it'] = cur_it
        batch['cur_train_meta'] = cur_train_dict

        loss, tb_dict, disp_dict = model_func(model, batch)
        if 'ad_cls_pred' in disp_dict:
            for cls in range(num_class):
                pseudo_match_cls[cls] = torch.cat([pseudo_match_cls[cls],
                                                   disp_dict['ad_cls_pred'][cls]], dim=1)
            disp_dict.pop('ad_cls_pred')

        loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()

        accumulated_iter += 1
        disp_dict.update({'loss': loss.item(), 'lr': cur_lr})

        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', loss, accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    # print(key, val)
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)

    adaptive_thres_flag = ch_model.use_adaptive_thres if hasattr(ch_model, 'use_adaptive_thres') else False
    if adaptive_thres_flag:
        a_thres = ch_model.adaptive_thres
        u_thres = ch_model.thresh
        hybrid_thres_flag = ch_model.use_hybrid_thres
        rel_adaptive_thres = ch_model.rel_adaptive_thres
        abs_thres = ch_model.absolute_thres
        
        perc_pl_list = []
        for cls in range(num_class):
            num_pred = pseudo_match_cls[cls].shape[1]

            sort_ind = pseudo_match_cls[cls][0].argsort(descending=True)
            pseudo_match_cls[cls][0] = pseudo_match_cls[cls][0][sort_ind]
            pseudo_match_cls[cls][1] = pseudo_match_cls[cls][1][sort_ind]

            match_cum = torch.cumsum(pseudo_match_cls[cls][1], dim=0)
            match_cum = match_cum / (torch.arange(num_pred).cuda() + 1)
            if (match_cum > a_thres).sum():
                c_thres = pseudo_match_cls[cls][0][(match_cum > a_thres).nonzero()[-1][0]]
                u_thres[cls] = c_thres.cpu().item()
            if hybrid_thres_flag:
                if rel_adaptive_thres:
                    n_pl = pseudo_match_cls[cls][0].shape[0]
                    u_num = int(n_pl * rel_adaptive_thres[0])
                    l_num = int(n_pl * rel_adaptive_thres[1])
                    up_thres, low_thres = pseudo_match_cls[cls][0][u_num], pseudo_match_cls[cls][0][l_num]
                    if u_thres[cls] < low_thres:
                        u_thres[cls] = low_thres
                    if u_thres[cls] > up_thres:
                        u_thres[cls] = up_thres
                elif u_thres[cls] < abs_thres[cls]:
                    u_thres[cls] = abs_thres[cls]

            num_pl = (pseudo_match_cls[cls][0] > u_thres[cls]).sum()
            perc_pl = num_pl / (pseudo_match_cls[cls][0].shape[0])
            perc_pl_list.append(perc_pl.item())


        if dist_train:
            # average dist u_thres
            world_size = dist.get_world_size()
            u_thres = torch.tensor(u_thres).cuda()
            group = dist.new_group([i for i in range(world_size)])
            dist.barrier()
            dist.all_reduce(u_thres, op=dist.ReduceOp.SUM, group=group)
            u_thres = u_thres / world_size
            u_thres = u_thres.cpu().tolist()
        print('PL Percent:', perc_pl_list)
        print('Adaptive Threshold :', u_thres)

        ch_model.thresh = u_thres

    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_one_epoch_dann(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                         rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False,
                         cur_epoch=0, total_epochs=0, retain_graph=False,
                         ):
    if total_it_each_epoch == len(train_loader):
        dataloader_iter = iter(train_loader)

    if rank == 0:
        pbar = tqdm.tqdm(total=total_it_each_epoch, leave=leave_pbar, desc='train', dynamic_ncols=True)

    cur_train_dict = {
        'cur_epoch': cur_epoch,
        'total_epochs': total_epochs,
        'total_it_each_epoch': total_it_each_epoch,
    }

    if type(model) == torch.nn.parallel.distributed.DistributedDataParallel:
        use_local_alignment = model.module.use_local_alignment
        use_domain_cls = model.module.backbone_2d.use_domain_cls
        if use_local_alignment:
            dla_cfg = model.module.dla_cfg
    else:
        use_local_alignment = model.use_local_alignment
        use_domain_cls = model.backbone_2d.use_domain_cls
        if use_local_alignment:
            dla_cfg = model.dla_cfg
    for cur_it in range(total_it_each_epoch):
        try:
            src_batch, trg_batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            src_batch, trg_batch = next(dataloader_iter)
            print('new iters')

        lr_scheduler.step(accumulated_iter)

        try:
            cur_lr = float(optimizer.lr)
        except:
            cur_lr = optimizer.param_groups[0]['lr']

        if tb_log is not None:
            tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)

        model.train()

        cur_train_dict['cur_it'] = cur_it
        cur_train_dict['data_split'] = True
        src_batch['cur_train_meta'] = cur_train_dict
        trg_batch['cur_train_meta'] = cur_train_dict

        optimizer.zero_grad()
        src_batch['domain_target'] = False
        src_batch['use_local_alignment'] = False
        src_loss, tb_dict, disp_dict = model_func(model, src_batch)
        src_loss.backward(retain_graph=retain_graph)
        if not retain_graph:
            clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
            optimizer.step()

        with torch.autograd.set_detect_anomaly(True):
            trg_batch['domain_target'] = True
            trg_batch['use_local_alignment'] = False
            trg_d_loss = torch.tensor(0.).cuda()
            if use_domain_cls:
                if not retain_graph:
                    optimizer.zero_grad()
                trg_d_loss, trg_tb_dict, trg_disp_dict = model_func(model, trg_batch)
                trg_d_loss.backward()
                clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
                optimizer.step()
                if 'domain_cls_loss' in trg_tb_dict:
                    tb_dict['trg_domain_cls_loss'] = trg_tb_dict['domain_cls_loss']
                    tb_dict['src_domain_cls_loss'] = tb_dict.pop('domain_cls_loss')

        loss_node_adv = torch.tensor(0.).cuda()
        if use_local_alignment:
            optimizer.zero_grad()
            src_batch['use_local_alignment'] = True
            trg_batch['use_local_alignment'] = True
            with torch.autograd.set_detect_anomaly(True):
                if dla_cfg.DIR == 'st':
                    with torch.no_grad():
                        _, tb_dict, disp_dict, s_dla_feat = model_func(model, src_batch)
                    _, trg_tb_dict, trg_disp_dict, t_dla_feat = model_func(model, trg_batch)
                else:
                    _, tb_dict, disp_dict, s_dla_feat = model_func(model, src_batch)
                    with torch.no_grad():
                        _, trg_tb_dict, trg_disp_dict, t_dla_feat = model_func(model, trg_batch)
                K = min(s_dla_feat.shape[0], t_dla_feat.shape[0])
                s_dla_feat, t_dla_feat = s_dla_feat[:K], t_dla_feat[:K]
                sigma_list = [0.01, 0.1, 1, 10, 100]
                loss_node_adv = 1 * mmd.mix_rbf_mmd2(s_dla_feat, t_dla_feat, sigma_list)
                loss_node_adv.backward()
                clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
                optimizer.step()

            tb_dict['domain_align_loss'] = loss_node_adv.item()

        accumulated_iter += 1
        disp_dict.update({'loss': src_loss.item() + trg_d_loss.item() + loss_node_adv.item(), 'lr': cur_lr})

        # log to console and tensorboard
        if rank == 0:
            pbar.update()
            pbar.set_postfix(dict(total_it=accumulated_iter))
            tbar.set_postfix(disp_dict)
            tbar.refresh()

            if tb_log is not None:
                tb_log.add_scalar('train/loss', src_loss.item() + trg_d_loss.item() + loss_node_adv.item(), accumulated_iter)
                tb_log.add_scalar('meta_data/learning_rate', cur_lr, accumulated_iter)
                for key, val in tb_dict.items():
                    # print(key, val)
                    tb_log.add_scalar('train/' + key, val, accumulated_iter)
    if rank == 0:
        pbar.close()
    return accumulated_iter

def train_model(model, optimizer, train_loader, model_func, lr_scheduler, optim_cfg,
                start_epoch, total_epochs, start_iter, rank, tb_log, ckpt_save_dir, train_sampler=None,
                lr_warmup_scheduler=None, ckpt_save_interval=1, max_ckpt_save_num=50,
                merge_all_iters_to_one_epoch=False):
    if cfg.DATA_CONFIG.get('PROG_AUG', None) and cfg.DATA_CONFIG.PROG_AUG.ENABLED and \
        start_epoch > 0:
        for cur_epoch in range(start_epoch):
            if cur_epoch in cfg.DATA_CONFIG.PROG_AUG.UPDATE_AUG:
                train_loader.dataset.trg_dataset.data_augmentor.re_prepare(
                    augmentor_configs=None, intensity=cfg.DATA_CONFIG.PROG_AUG.SCALE)

    accumulated_iter = start_iter
    with tqdm.trange(start_epoch, total_epochs, desc='epochs', dynamic_ncols=True, leave=(rank == 0)) as tbar:
        total_it_each_epoch = len(train_loader)
        if merge_all_iters_to_one_epoch:
            assert hasattr(train_loader.dataset, 'merge_all_iters_to_one_epoch')
            train_loader.dataset.merge_all_iters_to_one_epoch(merge=True, epochs=total_epochs)
            total_it_each_epoch = len(train_loader) // max(total_epochs, 1)

        dataloader_iter = iter(train_loader)
        for cur_epoch in tbar:
            if train_sampler is not None:
                train_sampler.set_epoch(cur_epoch)

            # train one epoch
            if lr_warmup_scheduler is not None and cur_epoch < optim_cfg.WARMUP_EPOCH:
                cur_scheduler = lr_warmup_scheduler
            else:
                cur_scheduler = lr_scheduler

            # curriculum data augmentation for SSDA
            if cfg.DATA_CONFIG.get('PROG_AUG', None) and cfg.DATA_CONFIG.PROG_AUG.ENABLED and \
                (cur_epoch in cfg.DATA_CONFIG.PROG_AUG.UPDATE_AUG):
                train_loader.dataset.trg_dataset.data_augmentor.re_prepare(
                    augmentor_configs=None, intensity=cfg.DATA_CONFIG.PROG_AUG.SCALE)

            if 'DANN' in type(train_loader.dataset).__name__ and train_loader.dataset.divide_data:
                train_epoch_func = train_one_epoch_dann
            else:
                train_epoch_func = train_one_epoch

            accumulated_iter = train_epoch_func(
                model, optimizer, train_loader, model_func,
                lr_scheduler=cur_scheduler,
                accumulated_iter=accumulated_iter, optim_cfg=optim_cfg,
                rank=rank, tbar=tbar, tb_log=tb_log,
                leave_pbar=(cur_epoch + 1 == total_epochs),
                total_it_each_epoch=total_it_each_epoch,
                dataloader_iter=dataloader_iter,
                cur_epoch=cur_epoch,
                total_epochs=total_epochs,
                retain_graph=cfg.DATA_CONFIG.get('RETAIN_GRAPH', False)
            )

            # save trained model
            trained_epoch = cur_epoch + 1
            if trained_epoch % ckpt_save_interval == 0 and rank == 0:

                ckpt_list = glob.glob(str(ckpt_save_dir / 'checkpoint_epoch_*.pth'))
                ckpt_list.sort(key=os.path.getmtime)

                if ckpt_list.__len__() >= max_ckpt_save_num:
                    for cur_file_idx in range(0, len(ckpt_list) - max_ckpt_save_num + 1):
                        os.remove(ckpt_list[cur_file_idx])

                ckpt_name = ckpt_save_dir / ('checkpoint_epoch_%d' % trained_epoch)
                save_checkpoint(
                    checkpoint_state(model, optimizer, trained_epoch, accumulated_iter), filename=ckpt_name,
                )


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()  # ordered dict
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def checkpoint_state(model=None, optimizer=None, epoch=None, it=None):
    optim_state = optimizer.state_dict() if optimizer is not None else None
    if model is not None:
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model_state = model_state_to_cpu(model.module.state_dict())
        else:
            model_state = model.state_dict()
    else:
        model_state = None

    try:
        import pcdet
        version = 'pcdet+' + pcdet.__version__
    except:
        version = 'none'

    return {'epoch': epoch, 'it': it, 'model_state': model_state, 'optimizer_state': optim_state, 'version': version}


def save_checkpoint(state, filename='checkpoint'):
    if False and 'optimizer_state' in state:
        optimizer_state = state['optimizer_state']
        state.pop('optimizer_state', None)
        optimizer_filename = '{}_optim.pth'.format(filename)
        torch.save({'optimizer_state': optimizer_state}, optimizer_filename)

    filename = '{}.pth'.format(filename)
    torch.save(state, filename)
