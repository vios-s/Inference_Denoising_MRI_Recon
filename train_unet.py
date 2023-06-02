import os
from pathlib import Path
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from pl_modules import FastMriDataModule, UnetModule
from data.mri_data import fetch_dir
from data.masking import create_mask_for_mask_type
from data.transforms import UnetDataTransform

torch.set_float32_matmul_precision('medium')

def build_args():
    parser = ArgumentParser()
    
    # basic args
    path_config = Path("Dataset/fastmri_dirs.yaml")
    num_gpus = 1
    batch_size = 24
    
    data_path = fetch_dir("knee_path", path_config)
    default_root_dir = fetch_dir("log_path", path_config) / "unet"
    
    parser.add_argument("--mode", default="train", type=str, choices=["train", "test"])
    parser.add_argument("--mask_type", default="random", type=str, choices=["random", "equispaced"])
    parser.add_argument("--center_fractions", default=[0.08], type=list)
    parser.add_argument("--accelerations", default=[4], type=list)
    parser.add_argument("--ckpt_path", default=None, type=str)
    parser = FastMriDataModule.add_data_specific_args(parser)
    parser = UnetModule.add_model_specific_args(parser)
    parser.set_defaults(
        data_path=data_path,
        gpus=num_gpus,
        seed=42,
        batch_size=batch_size,
        default_root_dir=default_root_dir,
        max_epochs=100,
        test_path=None
    )
    
    args = parser.parse_args()
    
    # checkpoints
    checkpoint_dir = args.default_root_dir / "checkpoints"
    if not checkpoint_dir.exists():
        checkpoint_dir.mkdir(parents=True)
        
    args.callbacks = [
        pl.callbacks.ModelCheckpoint(
            dirpath=checkpoint_dir,
            save_top_k=5,
            verbose=True,
            monitor="validation_loss",
            mode="min",
        )
    ]
    
    if args.ckpt_path is None:
        ckpt_list = sorted(checkpoint_dir.glob("*.ckpt"), key=os.path.getmtime)
        if ckpt_list:
            args.ckpt_path = str(ckpt_list[-1])
            
    print(args.ckpt_path)
    
            
    return args
    

def main():
    args = build_args()
    pl.seed_everything(args.seed)
    
    # * data
    # masking
    mask = create_mask_for_mask_type(args.mask_type, args.center_fractions, args.accelerations)
    # data transform
    train_transform = UnetDataTransform(args.challenge, mask_func=mask, use_seed=False)
    val_transform = UnetDataTransform(args.challenge, mask_func=mask, use_seed=True)
    test_transform = UnetDataTransform(args.challenge, mask_func=mask, use_seed=True, noise_lvl=args.noise_lvl)
    
    # pl data module
    data_module = FastMriDataModule(
        data_path=args.data_path,
        challenge=args.challenge,
        train_transform=train_transform,
        val_transform=val_transform,
        test_transform=test_transform,
        test_split=args.test_split,
        test_path=args.test_path,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    
    # * model
    model = UnetModule(
        in_chans=args.in_chans,
        out_chans=args.out_chans,
        chans=args.chans,
        num_pool_layers=args.num_pool_layers,
        drop_prob=args.drop_prob,
        lr=args.lr,
        lr_step_size=args.lr_step_size,
        lr_gamma=args.lr_gamma,
        weight_decay=args.weight_decay,
    )
    
    # * trainer
    trainer = pl.Trainer(
        logger=True,
        callbacks=args.callbacks,
        max_epochs=args.max_epochs,
        default_root_dir=args.default_root_dir,
        
    )
    
    # * run
    if args.mode == 'train':
        trainer.fit(model, data_module, ckpt_path=args.ckpt_path)
    elif args.mode == 'test':
        trainer.test(model, data_module, ckpt_path=args.ckpt_path)
    else:
        raise ValueError(f'Invalid mode: {args.mode}')
    
    
if __name__ == '__main__':
    main()