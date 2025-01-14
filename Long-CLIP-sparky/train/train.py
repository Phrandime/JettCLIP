import torch
#from utils import concat_all_gather, is_dist_avail_and_initialized, accuracy
#the original concat_all_gather is abandoned because of no gradient backward
from utils import is_dist_avail_and_initialized, accuracy
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm

import sys
sys.path.append("..")

from sharegpt4v import share4v_val_dataset, share4v_train_dataset
from model import longclip

from torch.utils.data.distributed import DistributedSampler
from scheduler import cosine_lr
import argparse
import os
import subprocess
import collections
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from datetime import datetime
from torch.cuda.amp import GradScaler
# import warnings
# warnings.filterwarnings("ignore")


class CLIP_Clean_Train():
    def __init__(self, rank,local_rank,args):
        self.rank=rank
        self.local_rank = local_rank
        self.base_model = args.base_model
        self.model, _ = longclip.load_from_clip(self.base_model, device='cpu',download_root=args.download_root)
        self.model.train()
        self.model.logit_scale = torch.nn.Parameter(torch.ones([]) * args.log_scale)  
        self.model = self.model.cuda()
        
        self.batch_size = args.batch_size
        self.num_epoch = args.epochs
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.warmup_length = args.warmup_length
        if args.exp_name == "auto":
            self.logdir = f"longclip/lr={args.lr}_wd={args.weight_decay}_wl={args.warmup_length}_logs={args.log_scale}_64xb"
        else:
            self.logdir = args.exp_name
        self.ckptdir = self.logdir + "/ckpt/"
        os.makedirs(self.ckptdir, exist_ok=True)
        self.writer = SummaryWriter(self.logdir)

        

        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank])
           
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scaler =GradScaler()


    def train_epoch(self, dataloader, epoch, start_iter=0):
        # running_loss_long = 0.0
        # running_loss_short = 0.0
        num_batches_per_epoch = len(dataloader)

        for i, (images, texts, short_texts) in enumerate(tqdm(dataloader, disable=(self.rank != 0))):
            step = num_batches_per_epoch * epoch + i
            if step < start_iter:
                continue

            # 将数据移动到 GPU
            images = images.cuda()
            texts = longclip.tokenize(texts, truncate=True).cuda()
            short_texts = longclip.tokenize(short_texts, truncate=True).cuda()

            # 学习率调整和优化器初始化
            self.scheduler(step)
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                # 获取模型返回的特征
                features = self.model(images, texts, short_texts)

                # 提取特征
                image_features_long = features["image_features_long"]
                text_features_long = features["text_features_long"]
                image_features_short = features["image_features_short"]
                text_features_short = features["text_features_short"]

                # 分布式收集全局特征
                image_feat_all_long = torch.cat(torch.distributed.nn.all_gather(image_features_long), dim=0)
                text_feat_all_long = torch.cat(torch.distributed.nn.all_gather(text_features_long), dim=0)
                image_feat_all_short = torch.cat(torch.distributed.nn.all_gather(image_features_short), dim=0)
                text_feat_all_short = torch.cat(torch.distributed.nn.all_gather(text_features_short), dim=0)

                # 计算相似度
                sim_i2tl = torch.matmul(image_features_long, text_feat_all_long.T)
                sim_tl2i = torch.matmul(image_feat_all_long, text_features_long.T).T
                sim_i2ts = torch.matmul(image_features_short, text_feat_all_short.T)
                sim_ts2i = torch.matmul(image_feat_all_short, text_features_short.T).T

                # 对相似度进行缩放
                logit_scale = self.model.module.logit_scale.exp()
                sim_i2tl *= logit_scale
                sim_tl2i *= logit_scale
                sim_i2ts *= logit_scale
                sim_ts2i *= logit_scale

                # 计算目标
                bs = images.size(0)
                targets = torch.linspace(self.rank * bs, self.rank * bs + bs - 1, bs, dtype=torch.long).to(images.device)

                # 计算损失
                loss_long = (F.cross_entropy(sim_i2tl, targets, label_smoothing=0.1)
                            + F.cross_entropy(sim_tl2i, targets, label_smoothing=0.1)) / 2
                loss_short = (F.cross_entropy(sim_i2ts, targets, label_smoothing=0.1)
                            + F.cross_entropy(sim_ts2i, targets, label_smoothing=0.1)) / 2
                loss = loss_long + loss_short

            # 梯度反向传播和优化器更新
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        #     # 更新损失
        #     running_loss_long += loss_long.item()
        #     running_loss_short += loss_short.item()
        # 
        # return running_loss_long / num_batches_per_epoch, running_loss_short / num_batches_per_epoch
        

    @torch.no_grad()
    def test_epoch(self, dataloader):
        temp_corr_dict = dict()
        rank = torch.distributed.get_rank()

        for id, (images, text) in enumerate(tqdm(dataloader, disable=(rank != 0))):

            images = images.cuda()
            image_features = self.model.module.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            text = longclip.tokenize(text, truncate=True).cuda()
            text_feature = self.model.module.encode_text(text)
            text_feature /= text_feature.norm(dim=-1, keepdim=True)

            i = 0
            correct = 0
            total = 0

            for i in range(text_feature.shape[0]):
                text = text_feature[i]
                sim = text @ image_features.T
                sim = sim.squeeze()
                correct_i = torch.argmax(sim)

                if i==correct_i:
                    correct = correct + 1
                total = total + 1

        return correct/total
    
    def test(self, epoch=0):
        rank = torch.distributed.get_rank()
        if rank == 0:
            self.model.eval()
            testset = share4v_val_dataset()
            testloader = torch.utils.data.DataLoader(testset, batch_size=1000, num_workers=32, pin_memory=True)
            with torch.no_grad():    

                acc = self.test_epoch(testloader)
                print("=====================================")
                print(f"test mean of share4v retrieval: {acc}")
                print("=====================================")

            return
    
    def train(self, resume=False, warmup_length=200):
        trainset = share4v_train_dataset()
        train_sampler = DistributedSampler(dataset=trainset, shuffle=True)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, sampler=train_sampler, num_workers=32, pin_memory=True)

        self.scheduler = cosine_lr(self.optimizer, base_lr=self.lr, warmup_length=warmup_length, steps=self.num_epoch * len(train_loader))
        start_epoch = 0
        resume_iter = 0
        
        for epoch in range(start_epoch, self.num_epoch):
            
            self.train_epoch(train_loader, epoch, start_iter=resume_iter)
            if self.rank == 0:
                name = "longclip.pt"
                now = datetime.now()
                formatted_date = now.strftime("%m-%d--%H_%M_%S_")
                #torch.distributed.barrier()
                torch.save(self.model.module.state_dict(), '../checkpoints/'+str(self.rank)+formatted_date+name)
            # print("=====================================")
            # print(f"loss after training epoch: {epoch}")
            # print("=====================================")


def setup_distributed(backend="nccl", port=None):
    """Initialize distributed training environment.
    support both slurm and torch.distributed.launch
    see torch.distributed.init_process_group() for more details
    """
    num_gpus = torch.cuda.device_count()

    def _set_os_environment(key, val):
        if key not in os.environ:
            os.environ[key] = val

    if "SLURM_JOB_ID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world_size = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if port is not None:
            os.environ["MASTER_PORT"] = str(port)
        elif "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = "29522"
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % num_gpus)
        os.environ["RANK"] = str(rank)
    else:
        _set_os_environment("RANK", "0")
        _set_os_environment("WORLD_SIZE", "1")
        _set_os_environment("MASTER_ADDR", "localhost")
        _set_os_environment("MASTER_PORT", "29522")

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank % num_gpus)
    
    dist.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(device=f'cuda:{rank % num_gpus}')
    return rank, rank % num_gpus


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--lr', default=1e-6, type=float, help='lr.')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='wd.')
    parser.add_argument('--log_scale', default=4.6052, type=float, help='clip temperature log scale.')
    parser.add_argument("--exp_name", default="auto", type=str, help="specify experiment name.")
    parser.add_argument("--warmup_length", default=200, type=int, help="warmup_length.")
    parser.add_argument("--base_model", default="ViT-L/14", help="CLIP Base Model")
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Batch size per gpu."#112
    )
    parser.add_argument(
        "--epochs", type=int, default=2, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--resume",
        default=False,
        action='store_true',
        help="resume training from checkpoint."
    )
    parser.add_argument("--download-root", default=None, help="CLIP Base Model download root")
    args = parser.parse_args()
    rank, local_rank = setup_distributed()
    print("DDP Done")

    trainer = CLIP_Clean_Train(
        rank=rank,
        local_rank=local_rank, 
        args=args
        )
    trainer.train(resume=args.resume, warmup_length=args.warmup_length)