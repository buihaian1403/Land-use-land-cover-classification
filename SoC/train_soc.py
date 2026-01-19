import os
import logging
import random
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import Dataset

from utils import net_builder, get_logger, count_parameters
from train_utils import TBLog, get_SGD, get_cosine_schedule_with_warmup, get_nodecay_schedule, RandomSampler
from models.soc.soc import SoC
from lib.datasets.iNatDataset import iNatDataset
from torchvision import  transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
import argparse
import h5py
import rasterio
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1' 
import gc
torch.backends.cudnn.benchmark = True
def main(args):
    '''
    For (Distributed)DataParallelism,
    main(args) spawn each process (main_worker) to each GPU.
    '''
    ngpus_per_node = max(torch.cuda.device_count(),1) # number of gpus of each node
    save_path = os.path.join(args.save_dir, args.save_name)
    if os.path.exists(save_path) and not args.overwrite:
        raise Exception('already existing model: {}'.format(save_path))
    if args.resume:
        if args.load_path is None:
            raise Exception('Resume of training requires --load_path in the args')
        if os.path.abspath(save_path) == os.path.abspath(args.load_path) and not args.overwrite:
            raise Exception('Saving & Loading pathes are same. \
                            If you want over-write, give --overwrite in the argument.')
        
    if args.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ.get("WORLD_SIZE",1))
    
    # distributed: true if manually selected or if world_size > 1
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed    

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')
    # divide the batch_size according to the number of nodes
    args.batch_size = int(args.batch_size / args.world_size)    
    
    if args.multiprocessing_distributed:
        # now, args.world_size means num of total processes in all nodes
        args.world_size = ngpus_per_node * args.world_size 
        # args=(,) means the arguments of main_worker
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args)) 
    else:
        main_worker(args.gpu, ngpus_per_node, args)
    

def main_worker(gpu, ngpus_per_node, args):
    '''
    main_worker is conducted on each GPU.
    '''
    
    global best_acc1
    args.gpu = gpu
    
    # set random seed for reproducibility 
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    
    # SET UP FOR DISTRIBUTED TRAINING
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu # compute global rank
        
        # set distributed group:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    
    # SET save_path and logger
    save_path = os.path.join(args.save_dir, args.save_name)
    logger_level = "WARNING"
    tb_log = None
    if args.rank % ngpus_per_node == 0:
        tb_log = TBLog(save_path, 'tensorboard')
        logger_level = "INFO"
    
    logger = get_logger(args.save_name, save_path, logger_level)
    logger.warning(f"USE GPU: {args.gpu} for training")
    
    args.bn_momentum = 1.0 - args.ema_m
    _net_builder = net_builder(args.net, 
                               {'depth': args.depth, 
                                'widen_factor': args.widen_factor,
                                'leaky_slope': args.leaky_slope,
                                'bn_momentum': args.bn_momentum,
                                'dropRate': args.dropout})
    
    def initialize_model():
        model = SoC(_net_builder,
                     args.num_classes,
                     args.ema_m,
                     args.ulb_loss_ratio,
                     num_eval_iter=args.num_eval_iter,
                     num_train_iter=args.num_train_iter,
                     num_tracked_batch=args.num_tracked_batch,
                     alpha=args.alpha,
                     save_dir=args.save_dir,
                     save_name=args.save_name,
                     gpu=args.gpu,
                     tb_log=tb_log,
                     logger=logger)
        logger.info(f'Number of Trainable Params: {count_parameters(model.train_model)}')
        return model

    # Initialize the model and optimizer
    model = initialize_model()
    # SET Optimizer & LR Scheduler
    ## construct SGD and cosine lr scheduler
    optimizer = get_SGD(model.train_model, 'SGD', args.lr, args.momentum, args.weight_decay)
    if args.lr_decay=='cos':
        if args.pretrained:
            scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    args.num_train_iter,
                                                    num_warmup_steps=args.num_train_iter*0.1)
        else:
            scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    2**20,
                                                    num_warmup_steps=args.num_train_iter*0)
    else:
        scheduler = get_nodecay_schedule(optimizer)
    ## set SGD and cosine lr on SoC 
    model.set_optimizer(optimizer, scheduler)
    
    
    # SET Devices for (Distributed) DataParallel
    # Check GPU availability upfront
    if not torch.cuda.is_available():
        raise Exception("GPU training is required but no GPU is available.")

    # GPU setup
    if args.gpu is not None and args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        logger.info(f"Using GPU: {args.gpu}")
    else:
        logger.info("Using CPU for training.")

    # Distributed setup
    if args.distributed:
        dist.init_process_group(
            backend=args.dist_backend, 
            init_method=args.dist_url, 
            world_size=args.world_size, 
            rank=args.rank
        )
        logger.info(f"Initialized distributed training with rank {args.rank}/{args.world_size}")

    # Model distribution
    if args.distributed:
        if args.gpu is not None:
            model.train_model = torch.nn.parallel.DistributedDataParallel(
                model.train_model.cuda(args.gpu), device_ids=[args.gpu]
            )
            model.eval_model = torch.nn.parallel.DistributedDataParallel(
                model.eval_model.cuda(args.gpu), device_ids=[args.gpu]
            )
        else:
            model.train_model = torch.nn.parallel.DistributedDataParallel(
                model.train_model.cuda()
            )
            model.eval_model = torch.nn.parallel.DistributedDataParallel(
                model.eval_model.cuda()
            )
    else:
        if args.gpu is not None:
            model.train_model = model.train_model.to(f"cuda:{args.gpu}")
            model.eval_model = model.eval_model.to(f"cuda:{args.gpu}")
        else:
            model.train_model = model.train_model.cpu()
            model.eval_model = model.eval_model.cpu()

    # Check and log model architecture
    logger.info(f"Model architecture: {model}")

    # Construct Dataset & DataLoader
    
    if args.dataset=='semi_fungi' or args.dataset=='semi_aves':
        data_transforms = {
            'train': transforms.Compose([
    #             transforms.Resize(args.input_size), 
                transforms.RandomResizedCrop(args.input_size),
                # transforms.ColorJitter(Brightness=0.4, Contrast=0.4, Color=0.4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'test': transforms.Compose([
                transforms.Resize(args.input_size), 
                transforms.CenterCrop(args.input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        }
        data_transforms['l_train'] = data_transforms['train']
        data_transforms['u_train'] = data_transforms['train']
        data_transforms['val'] = data_transforms['test']

        root_path = args.data_dir

        if args.trainval:
            ## following "A Realistic Evaluation of Semi-Supervised Learning for Fine-Grained Classification"
            ## l_train + val are used for labeled training data
            l_train = 'l_train_val'
        else:
            l_train = 'l_train'

        if args.unlabel == 'in':
            u_train = 'u_train_in'
        elif args.unlabel == 'inout':
            u_train = 'u_train'

        ## set val to test when using l_train + val for training
        if args.trainval:
            split_fname = ['test', 'test']
        else:
            split_fname = ['val', 'test']

        image_datasets = {split: iNatDataset(root_path, split_fname[i], args.dataset,
            transform=data_transforms[split], return_name=True) \
            for i,split in enumerate(['val', 'test'])}
        image_datasets['u_train'] = iNatDataset(root_path, u_train, args.dataset,
            transform=data_transforms['u_train'])
        image_datasets['l_train'] = iNatDataset(root_path, l_train, args.dataset,
            transform=data_transforms['train'], return_name=True)

        print("labeled data : {}, unlabeled data : {}".format(len(image_datasets['l_train']), len(image_datasets['u_train'])))
        print("validation data : {}, test data : {}".format(len(image_datasets['val']), len(image_datasets['test'])))

        num_classes = image_datasets['l_train'].get_num_classes() 
        
        print("#classes : {}".format(num_classes))

        loader_dict['train_lb'] = DataLoader(image_datasets['l_train'],
                        batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True,
                        sampler=RandomSampler(len(image_datasets['l_train']), args.num_train_iter * args.batch_size))

        mu = args.uratio
        loader_dict['train_ulb'] = DataLoader(image_datasets['u_train'],
                        batch_size=args.batch_size * mu, num_workers=args.num_workers, drop_last=True,
                        sampler=RandomSampler(len(image_datasets['u_train']), args.num_train_iter * args.batch_size * mu))
        # loader_dict['val'] = DataLoader(image_datasets['val'],
                        # batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False)
        loader_dict['eval_ulb'] = DataLoader(image_datasets['u_train'],
                        batch_size=args.batch_size * mu, shuffle=False, num_workers=args.num_workers, drop_last=False)
        loader_dict['eval'] = DataLoader(image_datasets['test'],
                        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=False,
                        sampler=RandomSampler(len(image_datasets['test']), len(image_datasets['test'])))
        
    elif args.dataset == 'my_dataset':
        print(f"load data")
        # Load data from files
        with h5py.File(args.input_h5, 'r') as f:
            labels = f['label'][:]         # mode label
            all_i = f['i'][:]
            all_j = f['j'][:]                  
            mask = labels != 0
            labeled_indices = np.where(mask)[0]
            if len(labeled_indices) < args.subset_size:
                raise ValueError(f"Only {len(labeled_indices)} labeled samples found, need {args.subset_size}")
            subset_idx = np.random.choice(labeled_indices, args.subset_size, replace=False)
            subset_idx = np.sort(subset_idx)
            data_labeled = f['data'][subset_idx].astype(np.float32)  # (subset, 48, 48, 10)
            labels_labeled = labels[subset_idx]
            print(f"Loaded {data_labeled.shape[0]:,} labeled tiles")

        gc.collect()

        class MyDataset(Dataset):
            def __init__(self, images, labels, transform=None, return_name=False, num_classes=None):
                self.images = images
                self.labels = labels
                self.transform = transform
                self.return_name = return_name
                if labels is not None:
                    self.num_classes = num_classes if num_classes is not None else len(set(labels))
                else:
                # If labels are None, set num_classes to None or any default value
                    self.num_classes = num_classes if num_classes is not None else 0
            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                image = self.images[idx].astype(np.int16)
                if self.labels is not None:
                    label = self.labels[idx]
                else:
                    label = 0    

                # Apply transformation if any
                if self.transform:
                    image = self.transform(image)

                if self.return_name:
                    return image, label
                
                elif self.return_name is False and self.labels is not None:
                    return image, label, idx
                
                else:
                    return image, idx

            def get_num_classes(self):
                return self.num_classes      
        
        # Define any transformation if needed (e.g., normalization)
        transform = transforms.Lambda(lambda x: torch.from_numpy(x).permute(2, 0, 1).float())  # No permutation
                # Create DataLoader for labeled data
        chunk_size = args.chunk_size
        with h5py.File(args.input_h5, 'r') as f:
            n_tiles = f['data'].shape[0]
            i_positions = f['i'][:]
            j_positions = f['j'][:]
            labels = f['label'][:]                         # mode label
            mask = labels != 0
            unlabeled_indices = np.where(labels == 0)[0]       
        for chunk_idx in range(0, len(unlabeled_indices), chunk_size):
            chunk_end = min(chunk_idx + chunk_size, len(unlabeled_indices))
            chunk_indices = unlabeled_indices[chunk_idx:chunk_end]
            with h5py.File(args.input_h5, 'r') as f:
                chunk_data = f['data'][chunk_indices].astype(np.float32)
            loader_dict={}
            oversampler = SMOTE(random_state=42)
            data_labeled, labels_labeled = oversampler.fit_resample(data_labeled.reshape(data_labeled.shape[0], -1), labels_labeled)
            data_labeled = data_labeled.reshape(-1, 48, 48, 10)
            x_train, x_temp, y_train, y_temp = train_test_split(data_labeled, labels_labeled, train_size=0.56, random_state=42, stratify=labels_labeled)
            x_test, x_eval, y_test, y_eval = train_test_split(x_temp, y_temp, test_size=0.32)
            
            # Handle unlabeled data chunks
            train_dataset = MyDataset(x_train, y_train, transform=transform, return_name=True)
            test_dataset = MyDataset(x_eval, y_eval, transform=transform, return_name = True)
            print(f"train dataset shape:{x_train.shape}, evaluation dataset shape: {x_test.shape}")
            loader_dict = {
                'train_lb': DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=torch.cuda.is_available()),
                'eval': DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=False, pin_memory=torch.cuda.is_available())
            }
            train_ulb_dataset = MyDataset(chunk_data, labels=None, transform=transform, return_name=False)
            eval_ulb_dataset = MyDataset(x_test, y_test, transform=transform, return_name=False)
            print(f"test dataset shape: {x_eval.shape}")
            num_classes = train_dataset.get_num_classes()
            mu = args.uratio
            # Create DataLoaders for unlabeled data
            loader_dict['train_ulb'] = DataLoader(
                train_ulb_dataset, batch_size=args.batch_size*mu, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=torch.cuda.is_available()
                )
            loader_dict['eval_ulb'] = DataLoader(
                eval_ulb_dataset, batch_size=args.batch_size*mu, shuffle=True, num_workers=args.num_workers, drop_last=False, pin_memory=torch.cuda.is_available()
                )
            print(f"'loaded all to loader_dict")
            for loader_name, loader in loader_dict.items():
                print(f"Checking {loader_name} shape:")
                for data in loader:
                    images = data[0]  # Always get the first item, which is `images`
                    print(f"  Data shape: {images.shape}")
                    break  # Only check the first batch

            print(f"Loaded chunk {chunk_idx // chunk_size + 1} with {len(chunk_data)} samples")
            # Set up and train the model
            model.set_data_loader(loader_dict)
            if args.resume:
                model.load_model(args.load_path, args.load_path_soc)
            trainer = model.train
            for epoch in range(args.epoch):
                trainer()
            model.save_model(f'latest_model_chunk_{chunk_idx}.pth', save_path)
                
            torch.cuda.empty_cache()
            gc.collect()
            print(f"Finished processing chunk {chunk_idx // chunk_size + 1}")
            
    # Clear GPU memory after each chunk
   
    logging.warning("Training is finished.")

    # === FINAL PREDICTION WITH PER-CHUNK MODELS + CENTER PIXEL MAPPING + DIRECT RASTER ===
    logger.info("Starting final prediction using per-chunk models and center-pixel mapping...")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    # Rebuild inference model (same architecture as training)
    _net_builder = net_builder(args.net, {
        "depth": args.depth,
        "widen_factor": args.widen_factor,
        "leaky_slope": args.leaky_slope,
        "bn_momentum": 1.0 - args.ema_m,
        "dropRate": args.dropout,
    })

    infer_model = SoC(
        _net_builder,
        args.num_classes,
        args.ema_m,
        args.ulb_loss_ratio,
        num_eval_iter=args.num_eval_iter,
        num_train_iter=args.num_train_iter,
        num_tracked_batch=args.num_tracked_batch,
        alpha=args.alpha,
        save_dir=os.path.join(args.save_dir, args.save_name),
        save_name=args.save_name,
        gpu=args.gpu,
    )
    infer_model.eval_model = infer_model.eval_model.to(device)
    infer_model.eval_model.eval()

    transform = transforms.Lambda(lambda x: torch.from_numpy(x).permute(2, 0, 1).float())

    # Load metadata from HDF5
    with h5py.File(args.input_h5, 'r') as f:
        all_i = f['i'][:]           # top-left row
        all_j = f['j'][:]           # top-left col
        all_labels = f['label'][:]
        total_tiles = f['data'].shape[0]

    labeled_mask = all_labels != 0
    unlabeled_indices = np.where(~labeled_mask)[0]

    # Load blueprint for full raster size and geo info
    with rasterio.open(args.blueprint_raster) as src:
        height, width = src.height, src.width
        profile = src.profile.copy()
        pred_raster = np.full((height, width), args.nodata, dtype=np.int8)

    # Fill known labeled pixels (center)
    known_count = 0
    for idx in np.where(labeled_mask)[0]:
        r = int(all_i[idx])
        c = int(all_j[idx])
        if 0 <= r < height and 0 <= c < width:
            pred_raster[r, c] = all_labels[idx]
            known_count += 1
    logger.info(f"Filled {labeled_mask.sum()} labeled center pixels.")

    # Predict all unlabeled tiles using the correct per-chunk model
    chunk_size = args.chunk_size
    batch_size = args.batch_size
    
    for chunk_start in range(0, len(unlabeled_indices), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(unlabeled_indices))
        chunk_indices = unlabeled_indices[chunk_start:chunk_end]
        num_tiles_in_chunk = len(chunk_idx)

        ckpt_filename = f"latest_model_chunk_{chunk_start}.pth"
        ckpt_path = os.path.join(args.save_dir, args.save_name,ckpt_filename)
        if not os.path.isfile(ckpt_path):
            logger.warning(f"Checkpoint missing: {ckpt_path} → skipping chunk {num_tiles_in_chunk:,} tiles (indices {chunk_start:,} – {chunk_end-1:,})")
            continue

        logger.info(f"Processing chunk starting at tile {chunk_start:,} ({num_tiles_in_chunk:,} tiles) → loading {ckpt_filename}")
        ckpt = torch.load(ckpt_path, map_location=device)

        # Load eval_model weights
        state_dict = ckpt.get("eval_model", ckpt.get("model", ckpt))
        infer_model.load_state_dict(state_dict, strict=False)

        # Load chunk data from HDF5
        with h5py.File(args.input_h5, 'r') as f:
            chunk_data = f['data'][chunk_idx].astype(np.float32)

        # Fill missing values (-32768)
        for b in range(chunk_data.shape[-1]):
            band = chunk_data[..., b]
            valid = band != -32768
            if valid.any():
                band[~valid] = band[valid].mean()
            else:
                band[~valid] = 0.0
            chunk_data[..., b] = band

        chunk_tensor_list = []
        for img in chunk_data:
            chunk_tensor_list.append(transform(img))
            if len(chunk_tensor_list) % 1000 == 0:
                logger.debug(f"Transformed {len(chunk_tensor_list):,} / {num_tiles_in_chunk:,} images")
        chunk_tensor = torch.stack(chunk_tensor_list)
        chunk_tensor = chunk_tensor.to(device)
        preds = []
        with torch.no_grad():            
            for i in range(0, len(chunk_tensor), batch_size):
                batch = chunk_tensor[i:i + batch_size].to(device)
                outputs = infer_model.eval_model(batch)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                _, predicted = torch.max(outputs, 1)
                preds.extend(predicted.cpu().numpy() + 1)  # 0-11 → 1-12

        # Map to center pixels
        written_count = 0
        out_of_bounds_count = 0
        for local_idx, global_idx in enumerate(chunk_indices):
            r = int(all_i[global_idx])
            c = int(all_j[global_idx])
            if 0 <= r < height and 0 <= c < width:
                pred_raster[r, c] = preds[local_idx]
                written_count += 1
            else:
                out_of_bounds_count += 1    
        logger.info(f"Chunk starting at {chunk_start:,} completed → wrote {written_count:,} / {num_tiles_in_chunk:,} centers")

        gc.collect()
        torch.cuda.empty_cache()

    # Save final raster
    profile.update({
        "dtype": rasterio.int8,
        "count": 1,
        "nodata": args.nodata
    })

    output_path = os.path.join(args.save_dir, "Soc_pred.tif")
    os.makedirs(args.save_dir, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(pred_raster, 1)

    logger.info(f"Final full-resolution LULC map saved to: {output_path}")

     
if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser(description='')
    
    '''
    Saving & loading of the model.
    '''
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('--save_name', type=str, default='soc')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_path', type=str, default=None, help='the path to checkpoints')
    parser.add_argument('--load_path_soc', type=str, default=None, help='the path to soc.pkl containing centroids and label_matrix (not necessary)')
    parser.add_argument('--overwrite', action='store_true')
    
    '''
    Training Configuration of SoC
    '''
    
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--num_train_iter', type=int, default=2000, 
                        help='total number of training iterations')
    parser.add_argument('--num_eval_iter', type=int, default=2000,
                        help='evaluation frequency')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='total number of batch size of labeled data')
    parser.add_argument('--uratio', type=int, default=4,
                        help='the ratio of unlabeled data to labeled data in each mini-batch')
    parser.add_argument('--eval_batch_size', type=int, default=32,
                        help='batch size of evaluation data loader (it does not affect the accuracy)')
    parser.add_argument('--num_tracked_batch', type=int, default=32, help='total number of batch tracked by CTT')
    parser.add_argument('--alpha', type=float, default=1.1, help='use {2.5,4} for {semi_aves,semi_fungi}')
    parser.add_argument('--ema_m', type=float, default=0.999, help='ema momentum for eval_model')
    parser.add_argument('--ulb_loss_ratio', type=float, default=1.0)
    
    '''
    Optimizer configurations
    '''
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--lr_decay', type=str, default='cos', help='use {cos,none} for {cosine decay,no decay}')
    parser.add_argument('--pretrained', action='store_true', default=False)
    parser.add_argument('--chunk_size', type=int, default=100000)

    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='resnet50', 
            help='use {resnet18/50/101,wrn,wrnvar,cnn13} for {ResNet-18/50/101,Wide ResNet,Wide ResNet-Var(WRN-37-2),CNN-13}')
    # for Wide ResNet
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--widen_factor', type=int, default=2)
    parser.add_argument('--leaky_slope', type=float, default=0.01)
    parser.add_argument('--dropout', type=float, default=0.00)
    
    '''
    Data Configurations
    '''
    
    parser.add_argument('--data_dir', type=str, default='./home/truongvinh123/SoC4SS-FGVC-master')
    parser.add_argument('--dataset', type=str, default='semi_aves')
    # parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('--num_classes', type=int, default=12)
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--input_size', default=48, type=int, 
            help='input image size')
    parser.add_argument('--unlabel', default='in', type=str, 
            choices=['in','inout'], help='U_in or U_in + U_out')
    parser.add_argument('--trainval', action='store_true', default=True,
            help='use {train+val,test,test} for {train,val,test}')
    parser.add_argument('--input_h5', type=str, default='/mnt/d/DCEC/30m_data_tiles_2023.h5',
                    help='Path to input .h5 file containing data, i, j, label')
    parser.add_argument('--subset_size', type=int, default=6024, choices=[500, 1000, 2000, 4000, 6024],
                    help='Number of labeled samples to use (500, 1000, 2000, 4000, 6024)')
    
    '''
    multi-GPUs & Distrbitued Training
    '''
    
    ## args for distributed training (from https://github.com/pytorch/examples/blob/master/imagenet/main.py)
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=1, type=int,
                        help='**node rank** for distributed training')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:10001', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--predict', action='store_true',
                    help='If set, skip training and run full-map prediction only')
    parser.add_argument('--blueprint_raster', type=str, default = './MKInput_30m_f16.tif',
                    help='Path to reference GeoTIFF used to define output size, CRS, transform')
    parser.add_argument('--nodata', type=int, default=-128,
                    help='No-data value written to the output GeoTIFF')
        
    args = parser.parse_args()
    main(args)
