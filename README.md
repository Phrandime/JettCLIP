# JettCLIP

"Multimodal Learning" Course Project

## Long-CLIP

`train/train.py` 执行训练

- 261 行 `trainset = share4v_train_dataset()` 加载数据集，数据集同时包含 `(images, texts, short_text)`

- `train_epoch` 中 `loss_long,loss_short = self.model(images, texts,short_text,self.rank)` ， `loss=loss_long+loss_short` ，没有使用 OpenCLIP 给的模块

- 36 行 `self.model, _ = longclip.load_from_clip(self.base_model, device='cpu',download_root=args.download_root)` 然后再对 `model` 进行一些处理

`model/longclip.py` 加载模型参数

- 136 行 `def load_from_clip(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", jit: bool = False, download_root: str = None):`

- 这里 `self.base_model` 对应参数 `name` ，似乎用于决定选用的 `state_dict` ，但是 `name in _MODELS` 似乎只是视觉，没看到文本参数哪里来的

- 228 行 `model = build_model(state_dict or model.state_dict(), load_from_clip = True).to(device)`

`model/model_longclip.py` 定义模型架构

- 512 行 `def build_model(state_dict: dict, load_from_clip: bool):`

- `state_dict` 至少用于设定模型参数，

- 537 行 
```python
    model = CLIP(
        embed_dim,
        image_resolution, vision_layers, vision_width, vision_patch_size,
        context_length, vocab_size, transformer_width, transformer_heads, transformer_layers, load_from_clip
    )
```

- 243 行 `class CLIP(nn.Module):`

- 定义的文本、图像编码器为：

```python
    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text): 
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        
        x = x + (self.positional_embedding.to(x.device) * self.mask1.to(x.device)).type(self.dtype).to(x.device) + (self.positional_embedding_res.to(x.device) * self.mask2.to(x.device)).type(self.dtype).to(x.device) 
        
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x
```

- 这里 `self.visual` 为 ResNet 或 ViT， `self.transformer` 为 Transformer，长短文本使用同一编码器，可见没有使用 hybrid。

- 442 行 `def forward(self, image, text_long,text_short,rank):` ，可见先编码再进行后续 PCA 等处理得到 `loss_itcl, loss_itcs` （即 `loss_long, loss_short` ）

## ml-mobileclip

**Part 1**

`open_clip/src/training/main.py` 执行训练，[github](https://github.com/apple/ml-mobileclip/tree/main/training)给出的运行方式为

```bash
cd open_clip/
bash configs/run_datacomp12m.sh  # Train a ViT-B/16 on DataComp-12M without DR
bash configs/run_datacompdr12m.sh  # Train a ViT-B/16 on DataComp-12M with DR
bash configs/run_datacompdr1B.sh  # Train a ViT-B/16 on DataComp-1B with DR
```

这里 `.sh` 加载 `args` 的参数并执行 `open_clip/src/training/main.py`

- 357 行 ```data = get_data(
        args,
        (preprocess_train, preprocess_val),
        epoch=start_epoch,
        tokenizer=tokenizer,
    )``` 加载数据集

- 430 行 `loss = create_loss(args)` 定义损失函数， `create_loss` 定义自 `open_clip/src/open_clip/factory.py` ，上面三者的后两者对应 `DRClipLoss` ，它继承 `DistillClipLoss` 的方法分别计算 `loss_gt` 和 `loss_syn` 

- 436 行 `train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=writer)`

`open_clip/src/training/train.py` 定义了 `train_one_epoch`

- 64 行 `def train_one_epoch(model, data, loss, epoch, optimizer, scaler, scheduler, dist_model, args, tb_writer=None):`

从代码看， `for i, batch in enumerate(dataloader):` 中的每个 `batch` 应该是 `[images, texts, dist_image_features, dist_text_features, syn_texts, dist_syn_text_features]` ，可以看出 teacher 编码的 feature 预先存储进了数据集，因此代码中没有定义 teacher 的模型结构（注意：没有设置参数 `args.distill` ，否则为调用 teacher 模型进行编码）。我们可以模仿此法。

`open_clip/src/training/main.py` 的 223 行
```
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        image_mean=args.image_mean,
        image_std=args.image_std,
        image_interpolation=args.image_interpolation,
        image_resize_mode=args.image_resize_mode,  # only effective for inference
        aug_cfg=args.aug_cfg,
        pretrained_image=args.pretrained_image,
        output_dict=True,
        **model_kwargs,
    )
```
定义模型，Trace 到 `open_clip/src/open_clip/factory.py` 第 180 行 
```python
def create_model(
        model_name: str,
        pretrained: Optional[str] = None,
        precision: str = 'fp32',
        device: Union[str, torch.device] = 'cpu',
        jit: bool = False,
        force_quick_gelu: bool = False,
        force_custom_text: bool = False,
        force_patch_dropout: Optional[float] = None,
        force_image_size: Optional[Union[int, Tuple[int, int]]] = None,
        force_preprocess_cfg: Optional[Dict[str, Any]] = None,
        pretrained_image: bool = False,
        pretrained_hf: bool = True,
        cache_dir: Optional[str] = None,
        output_dict: Optional[bool] = None,
        require_pretrained: bool = False,
        **model_kwargs,
):
```
其中第 261 行 
```python
        if custom_text:
            if "multimodal_cfg" in model_cfg:
                model = CoCa(**model_cfg, cast_dtype=cast_dtype)
            else:
                model = CustomTextCLIP(**model_cfg, cast_dtype=cast_dtype)
        else:
            model = CLIP(**model_cfg, cast_dtype=cast_dtype)
```

但是这里的 `CLIP` 似乎是 OpenCLIP 中的模型，不是 MobileCLIP 提出的模型，也就是 `open_clip/src/training/main.py` 确实是 **Training on DataCompDR with OpenCLIP**

`eval/zeroshot_imagenet.py` 测试模型的图像分类能力，得到一个 `metrics` ，这里面最终会调用到 `mobileclip/clip.py` 中的 `CLIP` 类，也就是 MobileCLIP 提出的模型架构。但是 yyd 目前还没有找到使用 MobileCLIP 的模型架构进行训练的代码（如果熟悉 Experiments 可补充这部分）。

**Part 2**

`mobileclip/clip.py` 定义了 MobileCLIP 的 CLIP 架构，`forward` 返回 `(image_embeddings, text_embeddings, self._exponentiate_and_clip_logits())` ，没有 loss 的计算。

参数见 `mobileclip/configs/` ，`checkpoints/` 里面应该是预训练参数。

`image_encoder` 使用 MCi，网络结构见 `mobileclip/models/mci.py` ，搭建 `FastViT` 并注册进 `timm` 库，然后调库 `timm.models` 实现

`text_encoder` 使用自己搭建的 `TextTransformer` 实现 hybrid，`mobileclip/text_encoder.py` 40 行 `context_length = cfg["context_length"]` 规定文本最大长度，原本是 77，我们需要改成 248。

`reparameterize` 也在代码结构中

如果要跟 LongCLIP 结合，`image_encoder` 结构照搬即可，`text_encoder` 仅仅修改文本最大长度不一定效果好，这是 [ChatGPT](https://chatgpt.com/share/676b992c-d650-800c-ad8c-f668c2b07a0d) 分析。

## Our Plan

使用 MobileCLIP 的网络结构，`image_encoder` 完全照搬 MobileCLIP；`text_encoder` 除了修改最大长度之外，可能还需要对网络结构进行一些微调，如果效果不佳（效果不佳也可以写进报告），一种更笨的办法是搬 LongCLIP 的 `text_encoder`（即只改 `image_encoder` ，然后 inference 时加入 reparameterization）。

CLIP-loss 照搬 LongCLIP，包括 PCA 之类的后续处理。
Datasets 使用 LongCLIP 的，LongCLIP 在知识蒸馏时作为 teacher ，建议模仿 MobileCLIP 中 `open_clip/src/training/main.py` 的方法，提前保存 LongCLIP 编码结果（因为不一定一次实验就成功）。

Distill-loss 对编码的使用有两种选择：1. 只选择原始图像和长文本的编码结果；2. 除选择原始图像和长文本的编码结果外，也选择主成分提取后的图像编码和短文本编码的结果。yyd 目前建议选第 2 种。

Distill-loss 可以尝试 MSE-loss ，如果担心实验失败、时间不够，可以先试原来的（只是注意图像-文本对由一组变成了长短两组），可以参考 [CLIP-KD](https://github.com/winycg/CLIP-KD/tree/main) 的 `src/open_clip/loss.py` ，但是 yyd 觉得反正都知道公式，不如就在 MobileCLIP 的 `src/open_clip/loss` 上稍微改一下（其他代码文件相应改动一些）。

## Details

代码基于 LongCLIP 或者 MobileCLIP 甚至 OpenCLIP 修改都可以（待补充）

如果基于 MobileCLIP ，yyd 的建议是基于 `open_clip/src/training/main.py` 一步步地修改，先搬运 LongCLIP 的数据集和数据加载方式，并额外将 teacher 的编码结果加入数据集，然后修改 `loss` 为可以选择 MSE-loss 的 Distill-loss ，然后模型加载我们修改后的模型（在 `open_clip/src/training/main.py` 中修改 `import` 的路径，再改点传入参数，应该就行了）。

`image_encoder` 和 `text_encoder` 基于 MobileCLIP 进行修改，`image_encoder` 最多改些输入大小之类的参数（还没细看），`text_encoder` 参考前文。`CLIP` 模型需要搬运 LongCLIP ，把 LongCLIP 的 CLIP 中的图像、文本编码器换了就行。此外建议将 LongCLIP 的 CLIP 在 `model/model_longclip.py` 的 452 行处截断，直接返回四个特征，后面 `loss` 的计算挪到外面（因为还要计算 Distill-loss）。

暂时写到这里，大致思路应该清晰了，细节还可以补充，比如 `text_encoder` 的具体修改方式，数据集分 train 和 validate 的方法之类的。
