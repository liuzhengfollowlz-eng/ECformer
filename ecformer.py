import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(dim))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output
    
class PatchEmbed(nn.Module):
    """Slide Patch Embedding"""

    def __init__(
        self,
        in_chans=1536,
        embed_dim=768,
        norm_layer='rnorm',#
        bias=True,
        act_layer=nn.ReLU,
        dropout=0.1
    ):
        super().__init__()

        self.proj = nn.Linear(in_chans, embed_dim, bias=bias)

        self.act = act_layer()
        self.norm = nn.LayerNorm(embed_dim) if norm_layer=='lnorm' else \
                    RMSNorm(embed_dim) if norm_layer=='rnorm' else \
                    nn.Identity()
        self.drop = nn.Dropout(dropout) if dropout>0 else nn.Identity()

    def forward(self, x):
        B, L, D = x.shape
        x = self.proj(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.drop(x)
        return x
    
class Head(nn.Module):
    def __init__(self,num_classes, hidden_dim):
        super().__init__()           
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        x=self.fc(x)
        return x

class ECAttention(nn.Module):
    def __init__(
        self, 
        embed_dim,#D
        num_heads=8, 
        num_cls=1,
        qkv_bias=False, 
        qk_scale=None, 
        attn_drop=0., 
        proj_drop=0.
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = embed_dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.num_cls=num_cls

        self.qv_glb = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
        self.kv_img = nn.Linear(embed_dim, embed_dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj1 = nn.Linear(embed_dim, embed_dim)
        self.proj2 = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.forward_hook_before_softmax = nn.Identity()
        self.backward_hook_after_softmax = nn.Identity()
        self.forward_hook_after_softmax = nn.Identity()

    def forward(self, x):
        B, N, C = x.shape
        NC = self.num_cls
        NI=N-NC

        x_glb,x_img=x[:,:NC],x[:,NC:]

        #####
        qv_glb = self.qv_glb(x_glb).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q_glb,v_glb = qv_glb[0], qv_glb[1]  #(B,H,NC,d)

        kv_img = self.kv_img(x_img).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k_img,v_img = kv_img[0], kv_img[1]  #(B,H,NI,d)

        attn_glb = (q_glb @ k_img.transpose(-2, -1)) * self.scale#(B,H,NC,NI)
        attn_img = attn_glb.transpose(-2, -1) #(B,H,NI,NC)

        attn_glb = self.forward_hook_before_softmax(attn_glb)
        attn_glb = attn_glb.softmax(dim=-1)
        attn_glb = self.attn_drop(attn_glb)
        #attn_glb = self.forward_hook_after_softmax(attn_glb)
        attn_glb = self.backward_hook_after_softmax(attn_glb)
        
        x_glb = (attn_glb @ v_img).transpose(1, 2).reshape(B, NC, C)#(B,H,NC,d)->(B,NC,D)
        x_glb = self.proj1(x_glb)
        x_glb = self.proj_drop(x_glb)

        attn_img = attn_img.softmax(dim=-1)
        attn_img = self.attn_drop(attn_img)
        x_img = (attn_img @ v_glb).transpose(1, 2).reshape(B, NI, C)#(B,H,NI,d)->(B,NI,D)
        x_img = self.proj2(x_img)
        x_img = self.proj_drop(x_img)

        x=torch.cat((x_glb, x_img), dim=1)
        return x

class Block(nn.Module):
    def __init__(
        self, 
        embed_dim, 
        num_heads=8,
        num_cls=1, 
        qkv_bias=False, 
        qk_scale=None, 
        attn_drop=0.,
        proj_drop=0., 
        drop_path=0.,
    ):
        super().__init__()
        self.num_cls=num_cls

        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = ECAttention(embed_dim, num_heads=num_heads,num_cls=num_cls, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x,coords):
        B, N, C = x.shape
        NC = self.num_cls
        NI=N-NC

        x_att= self.attn(self.ln1(x))
        x = x + self.drop_path(x_att)

        return x,x_att[:,:NC]

class ECFormer(nn.Module):
    def __init__(
        self,
        num_classes=2,
        input_dim=1536,
        embed_dim=512,
        num_heads=8,
        num_cls_tokens=2,
        depth=2,
        qkv_bias=False,
        qk_scale=None,
        dropout=0.1,
        ee=[1]
    ):
        super().__init__()

        self.num_cls=num_cls_tokens

        self.patch_embed = PatchEmbed(input_dim, embed_dim) #if input_dim!=embed_dim else nn.Identity()
        self.cls_token = nn.Parameter(torch.zeros(1, num_cls_tokens, embed_dim))

        self.pe_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([
            Block(  
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    num_cls=num_cls_tokens,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,                   
                    attn_drop=dropout,
                    proj_drop=dropout,
                    drop_path=dropout,
                    )
            for i in range(depth)])

        self.ln = nn.LayerNorm(embed_dim)

        self.head_cls = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head_img = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        #exit heads
        self.exit_heads=nn.ModuleList([Head(num_classes=num_classes,hidden_dim=embed_dim) for i in range(len(ee))])
        self.exit_positions = {int(position) - 1: index for index, position in enumerate(ee)}
        
        trunc_normal_(self.cls_token, std=.02)

    def forward_features(self, x,coords):
        B,N,D=x.shape
        x = self.patch_embed(x) 

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        exit_logits=[]
        for i, block in enumerate(self.blocks):
            if i in self.exit_positions:
                x,cls_attn = block(x,coords)
                exit_logit = self.exit_heads[self.exit_positions[i]](cls_attn.mean(dim=1))
                exit_logits.append(exit_logit)
            else:
                x,cls_attn = block(x,coords)

        x=self.ln(x)
        x_cls,x_img =x[:,:self.num_cls], x[:,self.num_cls:]
        return x_cls,x_img,exit_logits
    
    def forward(self, x,coords=None,indices=None,inf=False):
        x_cls,x_img,exit_logits= self.forward_features(x,coords)
        x_img= x_img.mean(dim=1)

        cls_logit = self.head_cls(x_cls.mean(dim=1))
        img_logit = self.head_img(x_img)

        if inf:
            return (cls_logit+img_logit)/2
        
        return cls_logit,img_logit,exit_logits
    

def build_model(config):
    model_type = config.model
    if model_type == 'ecformer':
        model=ECFormer(num_classes=config.num_classes,
                input_dim=config.input_dim,
                embed_dim=config.embed_dim,
                num_heads=config.num_heads,
                depth=config.depth,
                dropout=config.dropout,
                num_cls_tokens=config.num_cls_tokens,
                ee=config.ee
                )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model