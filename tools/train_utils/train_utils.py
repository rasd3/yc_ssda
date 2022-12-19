import glob
import os

import torch
import tqdm
from torch.nn.utils import clip_grad_norm_

from pcdet.models.model_utils.dsnorm import set_ds_target
from pcdet.models.model_utils import mmd

def train_one_epoch(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False,
                    cur_epoch=0, total_epochs=0,
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
    if rank == 0:
        pbar.close()
    return accumulated_iter


def train_one_epoch_dann(model, optimizer, train_loader, model_func, lr_scheduler, accumulated_iter, optim_cfg,
                    rank, tbar, total_it_each_epoch, dataloader_iter, tb_log=None, leave_pbar=False,
                    cur_epoch=0, total_epochs=0,
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
    else:
        use_local_alignment = model.use_local_alignment
        use_domain_cls = model.backbone_2d.use_domain_cls
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
        src_loss.backward()
        clip_grad_norm_(model.parameters(), optim_cfg.GRAD_NORM_CLIP)
        optimizer.step()

        trg_batch['domain_target'] = True
        trg_batch['use_local_alignment'] = False
        trg_d_loss = torch.tensor(0.).cuda()
        if use_domain_cls:
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
            _, tb_dict, disp_dict, s_dla_feat = model_func(model, src_batch)
            _, trg_tb_dict, trg_disp_dict, t_dla_feat = model_func(model, trg_batch)
            sigma_list = [0.01, 0.1, 1, 10, 100]
            B, K, C = s_dla_feat.shape
            loss_node_adv = 1 * mmd.mix_rbf_mmd2(s_dla_feat.reshape(-1, C),
                                                 t_dla_feat.reshape(-1, C),
                                                 sigma_list)
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
