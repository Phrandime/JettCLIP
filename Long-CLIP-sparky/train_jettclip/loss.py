import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss


def dot_ensemble_features(feat_a, feat_b, logit_scale, dims):
    """Compute sum_t Softmax(a_t @ b_t) for between features from an ensemble model."""
    num_members = len(dims)
    dims = np.cumsum([0] + dims)
    logits = [
        logit_scale * (feat_a[:, dims[i]:dims[i+1]] @ feat_b[dims[i]:dims[i+1], :])
        for i in range(num_members)
    ]
    logits = sum([F.softmax(logit, dim=1) for logit in logits]) / num_members
    return logits


class DistillClipLoss(ClipLoss):

    def __init__(
        self,
        *args,
        teacher_dimension = [-1],
        distill_loss_weights = [1.0, 1.0],
        average_after_softmax = False,
        dist_logit_scale = None,
        label_smoothing = None,
        method = "CRD",
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        if method not in ["CRD", "FD"]:
            raise ValueError(f"KD method '{method}' is not implemented.")

        self.dist_logit_scale = dist_logit_scale
        self.teacher_dimension = teacher_dimension
        self.distill_loss_weights = distill_loss_weights
        self.average_after_softmax = average_after_softmax
        self.label_smoothing=label_smoothing
        self.method = method

    def get_logits_dist(self, image_features, text_features, logit_scale):
        dims = self.teacher_dimension
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = dot_ensemble_features(image_features, all_text_features.T, logit_scale, dims)
                logits_per_text = dot_ensemble_features(text_features, all_image_features.T, logit_scale, dims)
            else:
                logits_per_image = dot_ensemble_features(all_image_features, all_text_features.T, logit_scale, dims)
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = dot_ensemble_features(image_features, text_features.T, logit_scale, dims)
            logits_per_text = dot_ensemble_features(text_features, image_features.T, logit_scale, dims)
        
        return logits_per_image, logits_per_text

    def dist_loss(self, teacher_logits, student_logits):
        if self.average_after_softmax:
            raise NotImplementedError
            return -(teacher_logits * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)
        else:
            return (teacher_logits.softmax(dim=1) * (teacher_logits.log_softmax(dim=1) - student_logits.log_softmax(dim=1))).sum(dim=1).mean(dim=0)
            # return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale=None,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        if self.label_smoothing:
            contrastive_loss = (
                F.cross_entropy(logits_per_image, labels, label_smoothing=self.label_smoothing) +
                F.cross_entropy(logits_per_text, labels, label_smoothing=self.label_smoothing)
            ) / 2 * self.distill_loss_weights[0]
        else:
            contrastive_loss = (
                F.cross_entropy(logits_per_image, labels) +
                F.cross_entropy(logits_per_text, labels)
            ) / 2 * self.distill_loss_weights[0]

        if self.method == "CRD":
            if self.dist_logit_scale is not None:
                dist_logit_scale = self.dist_logit_scale

            if self.average_after_softmax:
                dist_logits_per_image, dist_logits_per_text = \
                    self.get_logits_dist(dist_image_features, dist_text_features, dist_logit_scale)
            else:
                dist_logits_per_image, dist_logits_per_text = \
                    self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)
            
            distill_loss = (
                self.dist_loss(dist_logits_per_image, logits_per_image) +
                self.dist_loss(dist_logits_per_text, logits_per_text)
            ) / 2 * self.distill_loss_weights[1]
        
        elif self.method == "FD":
            distill_loss = (
                torch.sum((image_features - dist_image_features) ** 2, dim=1).mean() +
                torch.sum((text_features - dist_text_features) ** 2, dim=1).mean()
            )

        # Calculate accuracy for I2T and T2I
        with torch.no_grad():
            i2t_correct = (logits_per_image.argmax(dim=1) == labels).float().mean().item()
            t2i_correct = (logits_per_text.argmax(dim=1) == labels).float().mean().item()

        if output_dict:
            return {
                "contrastive_loss": contrastive_loss,
                "distill_loss": distill_loss,
                "accuracy_i2t": i2t_correct,  # Image-to-Text accuracy
                "accuracy_t2i": t2i_correct   # Text-to-Image accuracy
            }

        return contrastive_loss, distill_loss, i2t_correct, t2i_correct
