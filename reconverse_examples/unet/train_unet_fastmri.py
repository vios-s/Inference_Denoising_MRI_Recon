import os
from pathlib import Path
from argparse import ArgumentParser

import torch
import lightning.pytorch as pl

import sys
sys.path.append('../../')

from reconverse.data.fastmri.fastmri_data import fetch_dir
from reconverse.data.fastmri.subsample import create_mask_for_mask_type
from reconverse.data.fastmri.fastmri_transforms import UnetDataTransform
from reconverse.pl_modules import FastMriDataModule, UnetModule

torch.set_float32_matmul_precision('medium')

def cli_main():
    parser = ArgumentParser()
    
    # Basic 
    exp_name = "Unet_Fastmri"
    path_config = Path('../../reconverse/Datasets/datasets_dirs.yaml')
    data_path = fetch_dir("fastmri_knee_path", path_config)
    default_log_path = Path("../../logs") / exp_name

    parser.add_argument("--mode", default="train", type=str, choices=["train", "test"])
    parser.add_argument("--mask_type", choices=("random", "equispaced_fraction"), default="random", type=str, help="Type of k-space mask")
    parser.add_argument("--center_fractions", nargs="+", default=[0.08], type=float, help="Number of center lines to use in mask")
    parser.add_argument("--accelerations", nargs="+", default=[4], type=int, help="Acceleration rates to use for masks")
    parser.add_argument('--num_gpus', type=int, default=1, help='number of GPUs')
    parser.add_argument('--strategy', type=str, default='ddp', help='backend for distributed training')
    parser.add_argument("--exp_name", default=exp_name, type=str)
    parser.add_argument("--model", default="crnn_sr", type=str, choices=["cinenet", "crnn", "crnn_sr"])
    parser.add_argument("--ckpt_path", default=None, type=str)
    parser.add_argument("--max_epochs", default=50, type=int)

    parser = FastMriDataModule.add_data_specific_args(parser)
    parser = UnetModule.add_model_specific_args(parser)
        
    parser.set_defaults(
        # Lightning Data Module args
        data_path=data_path,
        test_path=None,  
    
        # Lightning Module args
        in_chans=1,  # number of input channels to U-Net
        out_chans=1,  # number of output chanenls to U-Net
        chans=32,  # number of top-level U-Net channels
        num_pool_layers=4,  # number of U-Net pooling layers
        drop_prob=0.0,  # dropout probability
        lr=0.001,  # RMSProp learning rate
        lr_step_size=40,  # epoch at which to decrease learning rate
        lr_gamma=0.1,  # extent to which to decrease learning rate
        weight_decay=0.0,  # weight decay regularization strength
        
        # Trainer args
        batch_size=8,
        replace_sampler_ddp=False,  # this is necessary for volume dispatch during val
        seed=42,  # random seed
        deterministic=True,  # makes things slower, but deterministic
        default_root_dir=default_log_path,  # directory for logs and checkpoints
        max_epochs=50,  # max number of epochs
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
    
    mask = create_mask_for_mask_type(
        args.mask_type, args.center_fractions, args.accelerations
    )
    # use random masks for train transform, fixed masks for val transform
    train_transform = UnetDataTransform(args.challenge, mask_func=mask, use_seed=False)
    val_transform = UnetDataTransform(args.challenge, mask_func=mask)
    test_transform = UnetDataTransform(args.challenge)
    
    # ------------
    # data
    # ------------
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
        distributed_sampler=False#(args.strategy in ("ddp", "ddp_cpu")),
    )

    # ------------
    # model
    # ------------
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
    
    print("Done Loading Data and Model...")
    
    trainer = pl.Trainer(
        num_sanity_val_steps=2,
        accelerator='gpu',
        #strategy='ddp',
        #devices=[0, 1, 2],
        logger=True,
        callbacks=args.callbacks,
        max_epochs=args.max_epochs,
        default_root_dir=args.default_root_dir,        
    )
    
    if args.mode == 'train':
        trainer.fit(model, data_module)
    
    elif args.mode == 'test':
        trainer.test(model, data_module, ckpt_path=args.ckpt_path)
    else:
        raise ValueError(f"Invalid mode: {args.mode}")
    

if __name__ == "__main__":
    cli_main()