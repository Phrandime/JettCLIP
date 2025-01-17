import torch

import sys
sys.path.append("..")

from train.utils import is_dist_avail_and_initialized, accuracy
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from tqdm import tqdm


from sharegpt4v_kd import share4v_kd_train_dataset
from loss import DistillClipLoss
from model import longclip

from torch.utils.data.distributed import DistributedSampler
from train.scheduler import cosine_lr
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
from datetime import datetime


class CLIP_Clean_Train():
    def __init__(self, rank, world_size, local_rank, args):
        self.rank = rank
        self.world_size = world_size
        self.local_rank = local_rank
        self.base_model = args.base_model

        # TODO: load JettCLIP model after implementing it.
        self.model, _ = longclip.load_from_clip(self.base_model, device='cpu',download_root=args.download_root)
        # import ipdb;ipdb.set_trace()
        print("Model load success...")
        # self.model, _ = longclip.load('../checkpoints/longclip-B.pt', device='cpu')  # load teacher model for checking
        self.model.train()
        self.model.logit_scale = torch.nn.Parameter(torch.ones([]) * args.log_scale)  
        self.model = self.model.cuda()
        
        self.batch_size = args.batch_size
        self.num_epoch = args.epochs
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.warmup_length = args.warmup_length
        if args.exp_name == "auto":
            self.logdir = f"longclip/lr={args.lr}_wd={args.weight_decay}_wl={args.warmup_length}_logs={args.log_scale}_64xb_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            self.logdir = args.exp_name
        self.ckptdir = self.logdir + "/ckpt/"
        os.makedirs(self.ckptdir, exist_ok=True)
        self.writer = SummaryWriter(self.logdir)

        self.label_smoothing = 0.1  # 原始 Long-CLIP 里的值，mobile-clip 原始的 `DistillClipLoss` 没有 label_smoothing
        self.dist_logit_scale = torch.nn.Parameter(torch.ones([]) * args.log_scale, requires_grad=False)  # 固定值
        
        self.distill_loss = DistillClipLoss(
            distill_loss_weights = [1.0, 1.0],  # 知识蒸馏损失的权重
            teacher_dimension = [-1],          # 教师特征维度一致
            label_smoothing = self.label_smoothing,
            dist_logit_scale = self.dist_logit_scale.exp(),  # 初始 dist_logit_scale
            rank = self.rank,                  # 当前设备的 rank
            world_size = self.world_size       # 分布式训练的 world_size
        )

        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank])
           
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scaler =GradScaler()


    def train_epoch(self, dataloader, epoch, start_iter=0):
        # running_loss_long = 0.0
        # running_loss_short = 0.0
        num_batches_per_epoch = len(dataloader)

        for i, (images, texts, short_texts, teacher_features) in enumerate(tqdm(dataloader, disable=(self.rank != 0))):
            # 原来的代码没有 teacher_features
            step = num_batches_per_epoch * epoch + i
            if step < start_iter:
                continue

            # 将数据移动到 GPU
            images = images.cuda()
            texts = longclip.tokenize(texts, truncate=True).cuda()
            short_texts = longclip.tokenize(short_texts, truncate=True).cuda()
            teacher_features = {key: value.squeeze(1).cuda() for key, value in teacher_features.items()}

            # 学习率调整和优化器初始化
            self.scheduler(step)
            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast():
            # if True:
                # 获取模型返回的特征
                features = self.model(images, texts, short_texts)

                # TODO: 修改损失函数，结合 teacher_features 计算知识蒸馏损失，这里 teacher_features 是与 features 结构相同的字典
                # 建议在外部定义损失函数，在这里直接调用函数。仿照 DistillClipLoss 计算 Contrastive Relational Distillation 即可
                # Long-CLIP 的每个图像对应一个长文本和一个短文本，图像 encoding 得到 image_features_long 之后，通过 PCA 得到 image_features_short
                # image_features_long 应该和 text_features_long 对齐，image_features_short 应该和 text_features_short 对齐
                # Already finished by ChatGPT and modified by Sparky.

                # 提取特征
                image_features_long = features["image_features_long"]  # (batch_size, encoding_dim)
                text_features_long = features["text_features_long"]  # (batch_size, encoding_dim)
                image_features_short = features["image_features_short"]  # (batch_size, encoding_dim)
                text_features_short = features["text_features_short"]  # (batch_size, encoding_dim)

                # 从 teacher_features 中获取教师模型的特征
                teacher_image_features_long = teacher_features["image_features_long"]
                teacher_text_features_long = teacher_features["text_features_long"]
                teacher_image_features_short = teacher_features["image_features_short"]
                teacher_text_features_short = teacher_features["text_features_short"]

                '''
                print('image_features_long: ', image_features_long.dtype)
                print('text_features_long: ', text_features_long.dtype)
                print('image_features_short: ', image_features_short.dtype)
                print('text_features_short: ', text_features_short.dtype)
                print('teacher_image_features_long: ', teacher_image_features_long.dtype)
                print('teacher_text_features_long: ', teacher_text_features_long.dtype)
                print('teacher_image_features_short: ', teacher_image_features_short.dtype)
                print('teacher_text_features_short: ', teacher_text_features_short.dtype)

                # raise
                '''

                # 调用 DistillClipLoss 计算损失
                loss_long, distill_loss_long = self.distill_loss(
                    image_features=image_features_long,
                    text_features=text_features_long,
                    logit_scale=self.model.module.logit_scale.exp(),  # 学生的缩放因子
                    dist_image_features=teacher_image_features_long,
                    dist_text_features=teacher_text_features_long,
                    dist_logit_scale=self.dist_logit_scale  # 教师的缩放因子
                )

                loss_short, distill_loss_short = self.distill_loss(
                    image_features=image_features_short,
                    text_features=text_features_short,
                    logit_scale=self.model.module.logit_scale.exp(),
                    dist_image_features=teacher_image_features_short,
                    dist_text_features=teacher_text_features_short,
                    dist_logit_scale=self.dist_logit_scale
                )
                
                # 总损失
                # TODO: 修改系数
                loss = loss_long + loss_short + distill_loss_long * 1.5 + distill_loss_short * 0.5

                '''
                for key, val in teacher_features.items():
                    print('teacher_' + key, torch.norm(val, p=2, dim=1))
                    print('distanc_' + key, torch.norm(val - features[key], p=2, dim=1))

                print(f"loss_long: {loss_long}\nloss_short: {loss_short}\ndistill_loss_long: {distill_loss_long}\ndistill_loss_short: {distill_loss_short}\n")
                '''
                
            # print("loss: ", loss)
            # 梯度反向传播和优化器更新
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # 记录损失和 logit scale
            self.writer.add_scalar("Loss/loss_long", loss_long.item(), step)
            self.writer.add_scalar("Loss/loss_short", loss_short.item(), step)
            self.writer.add_scalar("Loss/distill_loss_long", distill_loss_long.item(), step)
            self.writer.add_scalar("Loss/distill_loss_short", distill_loss_short.item(), step)
            self.writer.add_scalar("LogitScale/student", self.model.module.logit_scale.item(), step)
            

        #     # 更新损失
        #     running_loss_long += loss_long.item()
        #     running_loss_short += loss_short.item()
        # 
        # return running_loss_long / num_batches_per_epoch, running_loss_short / num_batches_per_epoch
        
    
    def train(self, resume=False, warmup_length=200, model_name = None):
        trainset = share4v_kd_train_dataset(model_name=model_name)
        print("dataset init success...")
        train_sampler = DistributedSampler(dataset=trainset, shuffle=True)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=self.batch_size, sampler=train_sampler, num_workers=32, pin_memory=True)

        self.scheduler = cosine_lr(self.optimizer, base_lr=self.lr, warmup_length=warmup_length, steps=self.num_epoch * len(train_loader))
        start_epoch = 0
        resume_iter = 0
        
        print("Train Starting...")
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
    return rank, rank % num_gpus, world_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='params')
    parser.add_argument('--lr', default=1e-6, type=float, help='lr.')
    parser.add_argument('--weight_decay', default=1e-2, type=float, help='wd.')
    parser.add_argument('--log_scale', default=4.6052, type=float, help='clip temperature log scale.')
    parser.add_argument("--exp_name", default="auto", type=str, help="specify experiment name.")
    parser.add_argument("--warmup_length", default=200, type=int, help="warmup_length.")
    parser.add_argument("--base_model", default="ViT-B/16", help="CLIP Base Model")
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size per gpu."#112
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--resume",
        default=False,
        action='store_true',
        help="resume training from checkpoint."
    )
    parser.add_argument("--download-root", default=None, help="CLIP Base Model download root")
    args = parser.parse_args()
    rank, local_rank, world_size = setup_distributed()
    print("DDP Done")

    trainer = CLIP_Clean_Train(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank, 
        args=args
        )
    trainer.train(resume=args.resume, warmup_length=args.warmup_length, model_name = args.base_model)