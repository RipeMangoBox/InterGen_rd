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
from datasets import InterHumanDataset
os.environ['PL_TORCH_DISTRIBUTED_BACKEND'] = 'nccl'
from lightning.pytorch.strategies import DDPStrategy
torch.set_float32_matmul_precision('medium')

import os
from argparse import ArgumentParser, Namespace
import yaml
from lightning import Trainer

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

def data_loader(dataset_root, data_cfg, batch_size, num_workers=16, shuffle=True):

    print("dataset_root: " + dataset_root)
    dataset = InterHumanDataset(data_cfg)

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        shuffle=shuffle,
        drop_last=True,
    )
    
def dataloaders(dataset_root, data_cfg, batch_size, num_workers):

    train_dl = data_loader(dataset_root, data_cfg.interhuman_train, batch_size, num_workers, shuffle=True)    
    val_dl = data_loader(dataset_root, data_cfg.interhuman_val, batch_size, num_workers, shuffle=False)
    
    return train_dl, val_dl


if __name__ == '__main__':
    print(os.getcwd())
    model_cfg = get_config("configs/model.yaml")
    train_cfg = get_config("configs/train.yaml")
    data_cfg = get_config("configs/datasets.yaml")


    # datamodule = DataModule(data_cfg, train_cfg.TRAIN.BATCH_SIZE, train_cfg.TRAIN.NUM_WORKERS)
    hparams, conf_name = get_hparams()
    train_dl, val_dl = dataloaders(hparams.dataset_root, data_cfg, hparams.batch_size, hparams.num_dataloader_workers)
    model = build_models(model_cfg)

    if train_cfg.TRAIN.RESUME:
        ckpt = torch.load(train_cfg.TRAIN.RESUME, map_location="cpu")
        for k in list(ckpt["state_dict"].keys()):
            if "model" in k:
                ckpt["state_dict"][k.replace("model.", "")] = ckpt["state_dict"].pop(k)
        model.load_state_dict(ckpt["state_dict"], strict=True)
        print("checkpoint state loaded!")
    litmodel = LitTrainModel(model, train_cfg)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath=litmodel.model_dir,
                                                       every_n_epochs=train_cfg.TRAIN.SAVE_EPOCH)
    trainer = pl.Trainer(
        default_root_dir=litmodel.model_dir,
        devices=[0], accelerator='gpu',
        max_epochs=train_cfg.TRAIN.EPOCH,
        strategy=DDPStrategy(find_unused_parameters=True),
        precision=32,
        callbacks=[checkpoint_callback],

    )
    # trainer.fit(model=litmodel, datamodule=datamodule)
    trainer.fit(model=litmodel, train_dataloaders=train_dl, val_dataloaders=val_dl)
