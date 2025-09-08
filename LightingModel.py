import torch
import lightning.pytorch as pl
import torch.optim as optim
from collections import OrderedDict
from utils.metrics_duet_on_training import DuetMetrics
from models import *
from collections import OrderedDict, defaultdict
from utils.intergen_vis import generate_one_sample
from tqdm import tqdm
from utils.utils import MotionNormalizerTorch
from torch.cuda.amp.autocast_mode import autocast
from utils.utils import save_pos3d

os.environ['PL_TORCH_DISTRIBUTED_BACKEND'] = 'nccl'
from lightning.pytorch.strategies import DDPStrategy
torch.set_float32_matmul_precision('medium')


# model named InterGen, using pytorch-lightning training framework, so called LitInterGen
class LitInterGen(pl.LightningModule):
    def __init__(self, model, cfg, val_dl, test_dl):
        super().__init__()
        self.save_hyperparameters(cfg)

        # cfg init
        self.cfg = cfg
        self.val_dl = val_dl
        self.test_dl = test_dl
        # self.mode = cfg.TRAIN.MODE
        self.data_dtype = torch.bfloat16 if cfg.TRAIN.PRECISION == 'bf16' else torch.float16 if cfg.TRAIN.PRECISION == 16 else torch.float32

        self.automatic_optimization = False

        self.accumulate_grad_batches = cfg.TRAIN.ACCUMULATE_GRAD_BATCHES

        self.model = model
        
        self.use_clip = getattr(cfg, "USE_CLIP", True)

        self.duet_metric_calc = DuetMetrics()
        self.last_duet_metrics = defaultdict(list)
        self.start_eval_epoch = getattr(cfg, 'START_EVAL_EPOCH', 50)
        self.eval_n_epochs = getattr(cfg, 'EVAL_N_EPOCHS', 10)
        self.start_synthesis_epoch = getattr(cfg, 'START_SYNTHESIS_EPOCH', 30)
        self.synthesis_n_epochs = getattr(cfg, 'SYNTHESIS_N_EPOCHS', 10)
        self.motion_normalizer = MotionNormalizerTorch(mode='both')
        
    def _configure_optim(self):
        optimizer = optim.AdamW(self.model.parameters(), lr=float(self.cfg.TRAIN.LR), weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)
        scheduler = CosineWarmupScheduler(optimizer=optimizer, warmup=10, max_iters=self.cfg.TRAIN.EPOCH, verbose=True)
        return [optimizer], [scheduler]

    def configure_optimizers(self):
        return self._configure_optim()

    def forward(self, batch_data):
        with autocast():
            batch = self.batch_data_process(batch_data)
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
        opt = self.optimizers()
        loss, loss_logs = self.forward(batch)
        self.log('TRAIN_LOSS', loss, prog_bar=True, sync_dist=True)
        
        # forward, calculate mean loss, which is identical to pytorch implementation
        self.manual_backward(loss / self.accumulate_grad_batches)
        
        # grad accumulation 
        if (batch_idx + 1) % self.accumulate_grad_batches == 0:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            opt.step()
            opt.zero_grad()

        return {"TRAIN_LOSS": loss,
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
        if self.it % self.cfg.TRAIN.LOG_STEPS == 0:
            mean_loss = OrderedDict({})
            for tag, value in self.logs.items():
                mean_loss[tag] = value / self.cfg.TRAIN.LOG_STEPS
                # self.writer.add_scalar(tag, mean_loss[tag], self.it)
                self.log(name=tag, value=mean_loss[tag], prog_bar=False, logger=True, sync_dist=True)
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
        # 计算分段验证损失
        avg_segmented_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        self.log('VAL_LOSS', avg_segmented_loss, sync_dist=True)

        # 计算whole test集上的损失
        whole_losses = []
        for batch in tqdm(self.test_dl, desc=f'[*] calc_whole_loss'):
            loss, _ = self(batch)
            whole_losses.append(loss.detach())
        if len(whole_losses) > 0:
            avg_whole_loss = torch.stack(whole_losses).mean()
            self.log('WHVAL_LOSS', avg_whole_loss, sync_dist=True)
        else:
            avg_whole_loss = None

        # eval
        if self.current_epoch >= self.start_eval_epoch and self.current_epoch % self.eval_n_epochs == 0:
            print(f'Eval after validation at epoch {self.current_epoch}')
            self.eval_during_training()
        else:
            self.last_duet_metrics.setdefault('fid_k', 660)
            self.log(f'Eval_FID', self.last_duet_metrics['fid_k']+2, prog_bar=True, logger=True, sync_dist=False)
            
        # synthesis and vis
        if self.current_epoch >= self.start_synthesis_epoch and self.current_epoch % self.synthesis_n_epochs == 0:
            self.synthesis_and_vis(self.current_epoch)

    def eval_during_training(self):
        with autocast():
            jointsl_list, jointsf_list = self.calc_duet_joints()
            duet_metrics = self.duet_metric_calc.eval_duet_metrics(jointsl_list, jointsf_list)
    
        for key, value in duet_metrics.items():
            self.log(f'Eval/{key}', value, prog_bar=False, logger=True, sync_dist=False)
        
        self.log(f'Eval_FID', duet_metrics['fid_k'], prog_bar=True, logger=True, sync_dist=False)
        self.last_duet_metrics = duet_metrics

    def batch_data_process(self, batch_data):
        dtype = self.data_dtype
        name, text, motion1, motion2, motion_lens = batch_data
        lmotion = motion1.cuda().detach().to(dtype)  # .to(self.device)
        fmotion = motion2.cuda().detach().to(dtype)  # .to(self.device)

        batch = OrderedDict({})
        batch["text"] = text.cuda().to(dtype)
        batch["motion_lens"] = motion_lens.cuda().long()
        
        batch["fmotion"] = self.motion_normalizer.forward(fmotion)
        batch["lmotion"] = self.motion_normalizer.forward(lmotion)
        return batch

    def calc_duet_joints(self):
        device = self.device
        with torch.no_grad():
            model = self.model.to(device).eval()
            print("InterGen Duet Eval...", flush=True)
            followers = []
            leaders = []
            for i, batch_data in enumerate(tqdm(self.test_dl, desc=f'[*] calc_duet_joints')):
                batch = self.batch_data_process(batch_data)
                output = model.forward_test(batch)['output']
                
                B, T, D = output.shape
                lmotion_output, fmotion_output = torch.split(output, [D//2, D//2], dim=-1)
                lmotion_output = self.motion_normalizer.backward(lmotion_output)
                fmotion_output = self.motion_normalizer.backward(fmotion_output)
                lpos = lmotion_output[..., :22*3]
                fpos = fmotion_output[..., :22*3]
             
                followers.append(fpos.cpu().data.numpy())
                leaders.append(lpos.cpu().data.numpy())
            return leaders, followers

    def synthesis_and_vis(self, epoch):
        def synthesis_and_vis_one_sample(epoch, dataset, dataset_name, sample_num, max_len):
            # 对self.test_dl中第0下标的数据进行合成和可视化
            device = self.device
            with torch.no_grad(), autocast():
                model = self.model.to(device).eval()
                for batch_data in dataset[:sample_num]:
                    # batch_data: name, text, motion1, motion2, motion_lens
                    name, text, motion1, motion2, motion_lens = batch_data
                    fname = name[0] if isinstance(name, (list, tuple)) else str(name)
                    
                    batch = self.batch_data_process(batch_data)
                    output = model.forward_test(batch)['output']
                    B, T, D = output.shape
                    lmotion_output, fmotion_output = torch.split(output, [D//2, D//2], dim=-1)
                    lmotion_output = self.motion_normalizer.backward(lmotion_output)
                    fmotion_output = self.motion_normalizer.backward(fmotion_output)
                    lpos = lmotion_output[0, :, :22*3].cpu().data.numpy()
                    fpos = fmotion_output[0, :, :22*3].cpu().data.numpy()
                    
                    # 保存和可视化
                    expdir = self.trainer.log_dir
                    synthdir = os.path.join(expdir, "synthesis_log")
                    npy_dir = os.path.join(synthdir, "npy")
                    video_dir = os.path.join(synthdir, "videos")
                    os.makedirs(npy_dir, exist_ok=True)
                    os.makedirs(video_dir, exist_ok=True)
                    
                    # save npy
                    if dataset_name == 'test':
                        save_pos3d(fpos, lpos, npy_dir, fname)
                    # save video
                    motion_both = [lpos[:max_len], fpos[:max_len]]
                    generate_one_sample(motion_both, f"{fname}_{dataset_name}_epoch{epoch}", video_dir)
                    
        synthesis_and_vis_one_sample(epoch, self.val_dl, 'val', sample_num=2, max_len=300)
        synthesis_and_vis_one_sample(epoch, self.test_dl, 'test', sample_num=1, max_len=600)
            
    def synthesis(self, result_dir):
        def synthesis_and_vis(duration):
            # 对self.test_dl中第0下标的数据进行合成和可视化
            device = self.device
            # 保存和可视化
            expdir = result_dir
            synthdir = os.path.join(expdir, "test_synthesis_log")
            npy_dir = os.path.join(synthdir, f"npy/{duration}")
            video_dir = os.path.join(synthdir, f"videos/{duration}")
            os.makedirs(npy_dir, exist_ok=True)
            os.makedirs(video_dir, exist_ok=True)
            
            with torch.no_grad(), autocast():
                model = self.model.to(device).eval()
                for batch_data in self.test_dl:
                    # batch_data: name, text, motion1, motion2, motion_lens
                    name, text, motion1, motion2, motion_lens = batch_data
                    fname = name[0] if isinstance(name, (list, tuple)) else str(name)
                    
                    batch = self.batch_data_process(batch_data)
                    output = model.forward_test(batch)['output']
                    B, T, D = output.shape
                    lmotion_output, fmotion_output = torch.split(output, [D//2, D//2], dim=-1)
                    lmotion_output = self.motion_normalizer.backward(lmotion_output)
                    fmotion_output = self.motion_normalizer.backward(fmotion_output)
                    lpos = lmotion_output[0, :duration, :22*3].cpu().data.numpy()
                    fpos = fmotion_output[0, :duration, :22*3].cpu().data.numpy()

                    # save npy
                    np.save(os.path.join(npy_dir, f"{fname}.npy"), np.concatenate([lpos, fpos], axis=1))
                    # save video
                    motion_both = [lpos, fpos]
                    generate_one_sample(motion_both, f"{fname}", video_dir)
                    
        synthesis_and_vis(duration=240)
        synthesis_and_vis(duration=None)
                
    def vis_gt(self, no_video=False, vis_mode='intergen', seqlen=240):
        expdir = os.path.join("./", "results/generated")
        self.gtdir = os.path.join(expdir, "gt")
        os.makedirs(self.gtdir, exist_ok=True)
        os.makedirs(os.path.join(self.gtdir, 'videos'), exist_ok=True)
        
        with torch.no_grad():
            print("GT Vis...", flush=True)
            followers = []
            leaders = []
            dance_names = []
            for i, batch_data in enumerate(tqdm(self.test_dl, desc='Generating Dance Poses')):
                name, text, motion1, motion2, motion_lens = batch_data
                lmotion = motion1.float().cpu().data.numpy()
                fmotion = motion2.float().cpu().data.numpy()
                lpos = lmotion[0, :seqlen, :22*3]
                fpos = fmotion[0, :seqlen, :22*3]
                fname = name[0]
             
                followers.append(fpos)
                leaders.append(lpos)
                save_pos3d(fpos, lpos, self.gtdir, fname)
                
                save_dir = os.path.join(self.gtdir, vis_mode)
                
                if not no_video:
                    if vis_mode == 'intergen':
                        motion_both = [lpos, fpos]
                        # vis after each sample
                        generate_one_sample(motion_both, fname, save_dir)
                
    def calc_motion_normalizer(self):
        # 计算motion和music的normalizer
        with torch.no_grad():
            print("Calc Motion Normalizer...", flush=True)
            followers = []
            leaders = []
            musics = []
            for i, batch_data in enumerate(tqdm(self.test_dl, desc='Generating Dance Poses')):
                name, text, motion1, motion2, motion_lens = batch_data
                lmotion = motion1.float().cpu().data.numpy()
                fmotion = motion2.float().cpu().data.numpy()
                lpos = lmotion[0]
                fpos = fmotion[0]

                followers.append(fpos)
                leaders.append(lpos)
                musics.append(text[0].float().cpu().data.numpy())

            # 1. motion1、motion2整体计算
            all_motion = np.concatenate([np.concatenate(leaders, axis=0), np.concatenate(followers, axis=0)], axis=0)
            global_mean_rd = np.mean(all_motion, axis=0)
            global_std_rd = np.std(all_motion, axis=0)
            
            os.makedirs(f"./data/normalizer/rd", exist_ok=True)
            np.save(f"./data/normalizer/rd/global_mean_rd.npy", global_mean_rd)
            np.save(f"./data/normalizer/rd/global_std_rd.npy", global_std_rd)

            # 2. motion1和motion2单独计算
            leaders_cat = np.concatenate(leaders, axis=0)
            followers_cat = np.concatenate(followers, axis=0)
            global_mean_rdl = np.mean(leaders_cat, axis=0)
            global_std_rdl = np.std(leaders_cat, axis=0)
            global_mean_rdf = np.mean(followers_cat, axis=0)
            global_std_rdf = np.std(followers_cat, axis=0)
            np.save(f"./data/normalizer/rd/global_mean_rdl.npy", global_mean_rdl)
            np.save(f"./data/normalizer/rd/global_std_rdl.npy", global_std_rdl)
            np.save(f"./data/normalizer/rd/global_mean_rdf.npy", global_mean_rdf)
            np.save(f"./data/normalizer/rd/global_std_rdf.npy", global_std_rdf)

            # 3. text
            musics_cat = np.concatenate(musics, axis=0)
            global_mean_rd_music = np.mean(musics_cat, axis=0)
            global_std_rd_music = np.std(musics_cat, axis=0)
            
            np.save("./data/normalizer/rd/global_mean_rd_music.npy", global_mean_rd_music)
            np.save("./data/normalizer/rd/global_std_rd_music.npy", global_std_rd_music)

            print("Saved motion and music normalizer statistics to ./data/")
    