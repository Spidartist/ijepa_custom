import yaml
import logging
import pprint
import torch
import numpy as np
import sys
import os
import wandb
from .utils.logging import CSVLogger, gpu_timer, grad_logger, AverageMeter
from .helper import load_checkpoint, init_model, init_opt
from .masks.multiblock import MaskCollator as MBMaskCollator
from .transforms import make_polyp_transforms
from .datasets.polyp import make_polyp
import copy
from .models.ijepa import IJEPA
import torch.nn.functional as F

class Trainer():
    def __init__(self, cfg_path):
        self.seed_anything()
        self.setup_logger()
        self.start_epoch = 0
        self.setup_config(cfg_path)
        self.setup_model()
        self.setup_data()
        self.setup_optimizer()
        self.load_checkpoint()

    def setup_logger(self):
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        self.logger = logging.getLogger()

    def seed_anything(self):
        _GLOBAL_SEED = 0
        np.random.seed(_GLOBAL_SEED)
        torch.manual_seed(_GLOBAL_SEED)
        torch.backends.cudnn.benchmark = True

    def setup_config_logger(self, args):
        folder = args["logging"]["folder"]
        tag = args["logging"]["write_tag"]

        dump = os.path.join(folder, "params-ijepa.yaml")
        with open(dump, "w") as f:
            yaml.dump(args, f)

        log_file = os.path.join(folder, f"{tag}.csv")
        self.save_path = os.path.join(folder, f"{tag}" + "-ep{epoch}.pth.tar")
        self.latest_path = os.path.join(folder, f"{tag}-latest.pth.tar")
        self.load_path = None
        if self.load_model:
            self.load_path = os.path.join(folder, self.r_file) if self.r_file is not None else self.latest_path

        # -- make csv_logger
        self.csv_logger = CSVLogger(
            log_file,
            ("%d", "epoch"),
            ("%d", "itr"),
            ("%.5f", "loss"),
            ("%.5f", "mask-A"),
            ("%.5f", "mask-B"),
            ("%d", "time (ms)"),
        )
        wandb.login(key="cca12c93cb17351580e3f9fd5136347e65a3463d")
        wandb.init(
            project="ijepa-polyp-16",
            config={
                "wd": self.wd,
                "patch_size": self.patch_size,
                "model_name": self.model_name,
                "batch_size": self.batch_size,
                "warmup": self.warmup,
                "lr": self.lr,
            },
        )

    def setup_config_meta(self, args):
        # -- META
        self.model_name = args["meta"]["model_name"]
        self.load_model = args["meta"]["load_checkpoint"]
        self.r_file = args["meta"]["read_checkpoint"]
        self.pred_depth = args["meta"]["pred_depth"]
        self.pred_emb_dim = args["meta"]["pred_emb_dim"]
        if not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda:0")
            torch.cuda.set_device(self.device)
    
    def setup_config_data(self, args):
        # -- DATA
        self.use_gaussian_blur = args["data"]["use_gaussian_blur"]
        self.use_horizontal_flip = args["data"]["use_horizontal_flip"]
        self.use_color_distortion = args["data"]["use_color_distortion"]
        self.color_jitter = args["data"]["color_jitter_strength"]
        # --
        self.batch_size = args["data"]["batch_size"]
        self.pin_mem = args["data"]["pin_mem"]
        self.num_workers = args["data"]["num_workers"]
        self.root_path = args["data"]["root_path"]
        self.image_folder = args["data"]["image_folder"]
        self.crop_size = args["data"]["crop_size"]
        self.crop_scale = args["data"]["crop_scale"]
    
    def setup_config_mask(self, args):
        # -- MASK
        self.allow_overlap = args["mask"][
            "allow_overlap"
        ]  # whether to allow overlap b/w context and target blocks
        self.patch_size = args["mask"]["patch_size"]  # patch-size for model training
        self.num_enc_masks = args["mask"]["num_enc_masks"]  # number of context blocks
        self.min_keep = args["mask"]["min_keep"]  # min number of patches in context block
        self.enc_mask_scale = args["mask"]["enc_mask_scale"]  # scale of context blocks
        self.num_pred_masks = args["mask"]["num_pred_masks"]  # number of target blocks
        self.pred_mask_scale = args["mask"]["pred_mask_scale"]  # scale of target blocks
        self.aspect_ratio = args["mask"]["aspect_ratio"]  # aspect ratio of target blocks

    def setup_config_optim(self, args):
        # -- OPTIMIZATION
        self.ema = args["optimization"]["ema"]
        self.ipe_scale = args["optimization"]["ipe_scale"]  # scheduler scale factor (def: 1.0)
        self.wd = float(args["optimization"]["weight_decay"])
        self.final_wd = float(args["optimization"]["final_weight_decay"])
        self.num_epochs = args["optimization"]["epochs"]
        self.warmup = args["optimization"]["warmup"]
        self.start_lr = args["optimization"]["start_lr"]
        self.lr = args["optimization"]["lr"]
        self.final_lr = args["optimization"]["final_lr"]
        

    def setup_config(self, cfg_path):
        # --
        self.log_freq = 1
        self.checkpoint_freq = 10
        # --
        args = None
        with open(cfg_path, 'r') as y_file:
            args = yaml.load(y_file, Loader=yaml.FullLoader)
            self.logger.info('loaded params...')
            pp = pprint.PrettyPrinter(indent=4)
            pp.pprint(args)

        self.setup_config_meta(args)
        self.setup_config_data(args)
        self.setup_config_mask(args)
        self.setup_config_optim(args)
        self.setup_config_logger(args)

    def setup_model(self):
        # -- init model
        encoder, predictor = init_model(
            device=self.device,
            patch_size=self.patch_size,
            crop_size=self.crop_size,
            pred_depth=self.pred_depth,
            pred_emb_dim=self.pred_emb_dim,
            model_name=self.model_name,
        )
        target_encoder = copy.deepcopy(encoder)
        self.model = IJEPA(encoder, predictor, target_encoder)

    def setup_optimizer(self):
        ipe = len(self.unsupervised_loader)

        # -- init optimizer and scheduler
        self.optimizer, self.scaler, self.scheduler, self.wd_scheduler = init_opt(
            encoder=self.model.context_encoder,
            predictor=self.model.predictor,
            wd=self.wd,
            final_wd=self.final_wd,
            start_lr=self.start_lr,
            ref_lr=self.lr,
            final_lr=self.final_lr,
            iterations_per_epoch=ipe,
            warmup=self.warmup,
            num_epochs=self.num_epochs,
            ipe_scale=self.ipe_scale,
            use_bfloat16=True,
        )
        ipe = len(self.unsupervised_loader)
        # -- momentum schedule
        self.momentum_scheduler = (
            self.ema[0] + i * (self.ema[1] - self.ema[0]) / (ipe * self.num_epochs * self.ipe_scale)
            for i in range(int(ipe * self.num_epochs * self.ipe_scale) + 1)
        )

    def setup_data(self):
        # -- make data transforms
        self.mask_collator = MBMaskCollator(
            input_size=self.crop_size,
            patch_size=self.patch_size,
            pred_mask_scale=self.pred_mask_scale,
            enc_mask_scale=self.enc_mask_scale,
            aspect_ratio=self.aspect_ratio,
            nenc=self.num_enc_masks,
            npred=self.num_pred_masks,
            allow_overlap=self.allow_overlap,
            min_keep=self.min_keep,
        )

        transform = make_polyp_transforms(
            crop_size=self.crop_size,
            crop_scale=self.crop_scale,
            gaussian_blur=self.use_gaussian_blur,
            horizontal_flip=self.use_horizontal_flip,
            color_distortion=self.use_color_distortion,
            color_jitter=self.color_jitter,
        )

        # -- init data-loaders/samplers
        _, self.unsupervised_loader = make_polyp(
            transform=transform,
            batch_size=self.batch_size,
            collator=self.mask_collator,
            pin_mem=self.pin_mem,
            num_workers=self.num_workers,
            training=True,
            root_path=self.root_path,
            image_folder=self.image_folder,
            drop_last=True,
        )

    def load_checkpoint(self):
        ipe = len(self.unsupervised_loader)
        if self.load_model:
            (
                self.model.context_encoder,
                self.model.predictor,
                self.model.target_encoder,
                self.optimizer,
                self.scaler,
                self.start_epoch,
            ) = load_checkpoint(
                device=self.device,
                r_path=self.load_path,
                encoder=self.model.context_encoder,
                predictor=self.model.predictor,
                target_encoder=self.model.target_encoder,
                opt=self.optimizer,
                scaler=self.scaler,
            )
            for _ in range(self.start_epoch * ipe):
                self.scheduler.step()
                self.wd_scheduler.step()
                next(self.momentum_scheduler)
                self.mask_collator.step()

    def save_checkpoint(self, epoch, loss_meter):
        save_dict = {
            "encoder": self.model.context_encoder.state_dict(),
            "predictor": self.model.predictor.state_dict(),
            "target_encoder": self.model.target_encoder.state_dict(),
            "opt": self.optimizer.state_dict(),
            "scaler": None if self.scaler is None else self.scaler.state_dict(),
            "epoch": epoch,
            "loss": loss_meter.avg,
            "batch_size": self.batch_size,
            "lr": self.lr,
        }
        torch.save(save_dict, self.latest_path)
        if (epoch + 1) % self.checkpoint_freq == 0:
            torch.save(save_dict, self.save_path.format(epoch=f"{epoch + 1}"))
    
    def load_imgs(self, udata, masks_enc, masks_pred):
        # -- unsupervised imgs
        imgs = udata.to(self.device, non_blocking=True)
        masks_1 = [u.to(self.device, non_blocking=True) for u in masks_enc]
        masks_2 = [u.to(self.device, non_blocking=True) for u in masks_pred]
        return (imgs, masks_1, masks_2)
    
    def loss_fn(self, z, h):
        loss = F.smooth_l1_loss(z, h)
        return loss
    
    def log_stats(self, epoch, itr, loss, maskA_meter, maskB_meter, etime, _new_wd, _new_lr, grad_stats):
        self.csv_logger.log(
            epoch + 1, itr, loss, maskA_meter.val, maskB_meter.val, etime
        )
        if (itr % self.log_freq == 0) or np.isnan(loss) or np.isinf(loss):
            self.logger.info(
                "[%d, %5d] loss: %.3f "
                "masks: %.1f %.1f "
                "[wd: %.2e] [lr: %.2e] "
                "[mem: %.2e] "
                "(%.1f ms)"
                % (
                    epoch + 1,
                    itr,
                    self.loss_meter.avg,
                    maskA_meter.avg,
                    maskB_meter.avg,
                    _new_wd,
                    _new_lr,
                    torch.cuda.max_memory_allocated() / 1024.0**2,
                    self.time_meter.avg,
                )
            )
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "loss": self.loss_meter.avg,
                    "wd": _new_wd,
                    "lr": _new_lr,
                }
            )

            if grad_stats is not None:
                self.logger.info(
                    "[%d, %5d] grad_stats: [%.2e %.2e] (%.2e, %.2e)"
                    % (
                        epoch + 1,
                        itr,
                        grad_stats.first_layer,
                        grad_stats.last_layer,
                        grad_stats.min,
                        grad_stats.max,
                    )
                )
    
    def train_step(self, imgs, masks_pred, masks_enc):
        _new_lr = self.scheduler.step()
        _new_wd = self.wd_scheduler.step()
        
        # Step 1. Forward
        with torch.cuda.amp.autocast(
            dtype=torch.bfloat16, enabled=True
        ):
            z, h = self.model(imgs, masks_pred, masks_enc)
            loss = self.loss_fn(z, h)

        #  Step 2. Backward & step
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        grad_stats = grad_logger(self.model.context_encoder.named_parameters())
        self.optimizer.zero_grad()

        # Step 3. momentum update of target encoder
        with torch.no_grad():
            m = next(self.momentum_scheduler)
            for param_q, param_k in zip(
                self.model.context_encoder.parameters(), self.model.target_encoder.parameters()
            ):
                param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

        return (float(loss), _new_lr, _new_wd, grad_stats)

    def train_one_epoch(self, epoch):
        self.logger.info("Epoch %d" % (epoch + 1))
        self.loss_meter = AverageMeter()
        maskA_meter = AverageMeter()
        maskB_meter = AverageMeter()
        self.time_meter = AverageMeter()
        self.logger.info("Set up meters for epoch %d" % (epoch + 1))
        for itr, (udata, masks_enc, masks_pred) in enumerate(self.unsupervised_loader):
            imgs, masks_enc, masks_pred = self.load_imgs(udata, masks_enc, masks_pred)
            maskA_meter.update(len(masks_enc[0][0]))
            maskB_meter.update(len(masks_pred[0][0]))
            (loss, _new_lr, _new_wd, grad_stats), etime = gpu_timer(self.train_step, imgs=imgs, masks_pred=masks_pred, masks_enc=masks_enc)
            self.loss_meter.update(loss)
            self.time_meter.update(etime)
            self.log_stats(epoch, itr, loss, maskA_meter, maskB_meter, etime, _new_wd, _new_lr, grad_stats)
            assert not np.isnan(loss), "loss is nan"
    
    def train(self):
        for epoch in range(self.start_epoch, self.num_epochs):
            self.train_one_epoch(epoch)
            # -- Save Checkpoint after every epoch
            self.logger.info("avg. loss %.3f" % self.loss_meter.avg)
            self.save_checkpoint(epoch + 1, self.loss_meter)

