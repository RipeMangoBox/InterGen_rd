import sys
sys.path.append(sys.path[0] + r"/../")
import torch
import lightning.pytorch as pl
import torch.optim as optim
from collections import OrderedDict
from datasets import DataModule
from configs import get_config
from os.path import join as pjoin
from torch.utils.tensorboard import SummaryWriter
from models import *
from tools.infer import generate_samples_loop
from datasets.dataloader import DD100lf_dl
from models import InterGen
from LightingModel import LitInterGen
os.environ['PL_TORCH_DISTRIBUTED_BACKEND'] = 'nccl'
from lightning.pytorch.strategies import DDPStrategy
torch.set_float32_matmul_precision('medium')

import os
from argparse import ArgumentParser, Namespace
import yaml
from lightning import Trainer
import shutil

class CopyConfigCallback(pl.Callback):
    def __init__(self, config_paths):
        super().__init__()
        self.config_paths = config_paths if isinstance(config_paths, list) else [config_paths]

    def on_train_start(self, trainer, pl_module):
        # 获取当前的日志目录（事件目录）
        log_dir = trainer.log_dir

        # 检查config文件是否存在
        for config_path in self.config_paths:
            if os.path.exists(config_path):
                shutil.copy(config_path, log_dir)
                print(f"Config file copied to {log_dir}")
            else:
                print(f"Config file {self.config_path} not found!")
                
def get_hparams():
    parser = ArgumentParser()
    parser.add_argument("--dataset_root", default='./data/motions_processed')
    parser.add_argument("--hparams_file", default='./configs/train.yaml')
    parser = Trainer.add_argparse_args(parser)
    default_params = parser.parse_args()

    conf_name = os.path.basename(default_params.hparams_file)
    if default_params.hparams_file.endswith(".yaml"):
        hparams_json = yaml.full_load(open(default_params.hparams_file))

    params = vars(default_params)
    params.update(hparams_json)
    params.update(vars(default_params))

    hparams = Namespace(**params)

    return hparams, conf_name

class LitTrainModel(pl.LightningModule):
    def __init__(self, model, cfg):
        super().__init__()
        # cfg init
        self.cfg = cfg
        self.mode = cfg.TRAIN.MODE

        self.automatic_optimization = False

        self.save_root = pjoin(self.cfg.GENERAL.CHECKPOINT, self.cfg.GENERAL.EXP_NAME)
        self.model_dir = pjoin(self.save_root, 'model')
        self.meta_dir = pjoin(self.save_root, 'meta')
        self.log_dir = pjoin(self.save_root, 'log')

        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.meta_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        self.model = model

        self.writer = SummaryWriter(self.log_dir)

    def _configure_optim(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=float(self.cfg.TRAIN.LR), weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=10, max_iters=self.cfg.TRAIN.EPOCH, verbose=True)
        return [optimizer], [scheduler]

    def configure_optimizers(self):
        return self._configure_optim()

    def forward(self, batch_data):
        name, text, motion1, motion2, motion_lens = batch_data
        motion1 = motion1.detach().float()  # .to(self.device)
        motion2 = motion2.detach().float()  # .to(self.device)
        motions = torch.cat([motion1, motion2], dim=-1)

        B, T = motion1.shape[:2]

        batch = OrderedDict({})
        batch["text"] = text
        batch["motions"] = motions.reshape(B, T, -1).type(torch.float32)
        batch["motion_lens"] = motion_lens.long()

        loss, loss_logs = self.model(batch)
        return loss, loss_logs

    def on_train_start(self):
        self.rank = 0
        self.world_size = 1
        self.start_time = time.time()
        self.it = self.cfg.TRAIN.LAST_ITER if self.cfg.TRAIN.LAST_ITER else 0
        self.epoch = self.cfg.TRAIN.LAST_EPOCH if self.cfg.TRAIN.LAST_EPOCH else 0
        self.logs = OrderedDict()


    def training_step(self, batch, batch_idx):
        loss, loss_logs = self.forward(batch)
        opt = self.optimizers()
        opt.zero_grad()
        self.manual_backward(loss)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
        opt.step()

        return {"loss": loss,
            "loss_logs": loss_logs}


    def on_train_batch_end(self, outputs, batch, batch_idx):
        if outputs.get('skip_batch') or not outputs.get('loss_logs'):
            return
        for k, v in outputs['loss_logs'].items():
            if k not in self.logs:
                self.logs[k] = v.item()
            else:
                self.logs[k] += v.item()

        self.it += 1
        if self.it % self.cfg.TRAIN.LOG_STEPS == 0 and self.device.index == 0:
            mean_loss = OrderedDict({})
            for tag, value in self.logs.items():
                mean_loss[tag] = value / self.cfg.TRAIN.LOG_STEPS
                self.writer.add_scalar(tag, mean_loss[tag], self.it)
            self.logs = OrderedDict()
            print_current_loss(self.start_time, self.it, mean_loss,
                               self.trainer.current_epoch,
                               inner_iter=batch_idx,
                               lr=self.trainer.optimizers[0].param_groups[0]['lr'])

    def on_train_epoch_end(self):
        # pass
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()

    def validation_step(self, batch, batch_idx):
        loss, loss_logs = self(batch)
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)

        output = {"val_loss": loss}
        return output

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log('VAL_LOSS', avg_loss, sync_dist=True)
        
        if self.current_epoch % 10 == 0:
            self.synthesis_and_vis_one_sample()

    def save(self, file_name):
        state = {}
        try:
            state['model'] = self.model.module.state_dict()
        except:
            state['model'] = self.model.state_dict()
        torch.save(state, file_name, _use_new_zipfile_serialization=False)
        return

    def synthesis_and_vis_one_sample(self):
        prompts_path = "prompts.txt"
        result_dir = self.logger.log_dir
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        generate_samples_loop(self.model, self.cfg, self.cfg, prompts_path, result_dir=result_dir, epoch=self.current_epoch)

def build_models(cfg):
    if cfg.NAME == "InterGen":
        model = InterGen(cfg)
    return model

def load_checkpoint(ckpt):
    litmodel = LitInterGen.load_from_checkpoint(checkpoint_path=ckpt, model=model, cfg=cfg, val_dl=val_dl, test_dl=test_dl)
    print("checkpoint state loaded!")
    return litmodel.cuda()

def train_mode(litmodel):
    if train_cfg.TRAIN.RESUME is not None:
        ckpt = train_cfg.TRAIN.RESUME
        print(f"resuming from checkpoint: {ckpt}")
        litmodel = load_checkpoint(ckpt)
    else:
        litmodel = LitInterGen(model=model, cfg=cfg, val_dl=val_dl, test_dl=test_dl)
        ckpt = None
        
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        save_top_k=4, 
        monitor='TRAIN_LOSS', 
        mode='min', 
        filename='model-{epoch:02d}-{step:05d}-{TRAIN_LOSS:.4f}-{VAL_LOSS:.4f}', 
        save_last=False, 
        verbose=True
    )
    
    fid_checkpoint = pl.callbacks.ModelCheckpoint(
        save_top_k=4,  # 保存全局最优的4个
        monitor='Eval_FID',  # 确保训练时该指标被正确记录
        mode='min',  # FID越低越好
        filename='fid_best_{epoch:02d}-{Eval_FID:.2f}',
        verbose=True,
    )
    
    copy_config_callback = CopyConfigCallback([model_config_path, train_config_path, dataset_config_path])
    max_epochs = getattr(cfg.TRAIN, 'EPOCH', 2000)
    
    trainer = pl.Trainer(
        # accumulate_grad_batches=2, # manually set in training_step()
        devices=[1],
        accelerator='gpu',
        max_epochs=max_epochs,
        callbacks=[checkpoint_callback, fid_checkpoint, copy_config_callback],
        # **cfg.Trainer,
        # num_sanity_val_steps=0,
    )
    trainer.fit(model=litmodel, train_dataloaders=train_dl, val_dataloaders=val_dl, ckpt_path=ckpt)

def synthesis_and_vis_mode(litmodel):
    litmodel.synthesis(config_path)
    
def vis_gt_mode(litmodel, no_video=False, vis_mode='intergen', seqlen=300):
    litmodel.vis_gt(no_video=no_video, vis_mode=vis_mode, seqlen=seqlen)
    
def calc_motion_normalizer_mode(litmodel):
    train_dl, val_dl, test_dl = DD100lf_dl(dataset_cfg, full_length=True)
    litmodel = LitInterGen(model, cfg, val_dl, train_dl)
    litmodel.calc_motion_normalizer()
    
if __name__ == '__main__':
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    config_path = 'configs'
    model_config_path = f"{config_path}/model.yaml"
    train_config_path = f"{config_path}/train.yaml"
    dataset_config_path = f"{config_path}/datasets.yaml"
    
    print(os.getcwd())
    model_cfg = get_config(model_config_path)
    train_cfg = get_config(train_config_path)
    data_cfg = get_config(dataset_config_path)
    datatset_train = data_cfg.get('dataset_train', 'dd100lf')
    dataset_cfg = data_cfg[datatset_train]
    
    cfg = model_cfg
    cfg.update(train_cfg)
    cfg.update(dataset_cfg)

    # datamodule = DataModule(data_cfg, train_cfg.TRAIN.BATCH_SIZE, train_cfg.TRAIN.NUM_WORKERS)
    train_dl, val_dl, test_dl = DD100lf_dl(dataset_cfg)
    model = build_models(cfg)

    litmodel = LitInterGen(model, cfg, val_dl, train_dl)
    
    mode = 'train'
    if mode == 'train':
        train_mode(litmodel)
    elif mode == 'synthesis':
        ckpt = model_cfg.CHECKPOINT
        litmodel = load_checkpoint(ckpt)
        synthesis_and_vis_mode(litmodel)
    elif mode == 'vis_gt':
        vis_gt_mode(litmodel, no_video=False, vis_mode='intergen', seqlen=240)
    elif mode == 'calc_motion_normalizer':
        calc_motion_normalizer_mode(litmodel)