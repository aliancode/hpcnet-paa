import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Tuple
from .clip_wrapper import FrozenCLIP

class HierarchicalPromptBank(nn.Module):
    def __init__(self, embed_dim: int = 512, K_F: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.K_F = K_F

        init_descriptors = [
            "round shape", "angular shape", "smooth texture", "rough texture",
            "high contrast", "low contrast", "symmetric pattern", "asymmetric pattern",
            "vertical line", "horizontal line", "diagonal edge", "curved boundary",
            "solid color", "gradient color", "sparse pattern", "dense pattern"
        ]
        clip = FrozenCLIP()
        with torch.no_grad():
            fp_anchors_raw = clip.encode_text_batch(init_descriptors)
        self.register_buffer('fp_anchors', fp_anchors_raw)
        self.fp_prompts = nn.Parameter(torch.randn(K_F, embed_dim) * 0.02)
        self.cp_alphas = nn.ParameterList()
        self.cp_residuals = nn.ParameterList()
        self.ip_betas = nn.ParameterDict()
        self.ip_residuals = nn.ParameterDict()

    def add_compositional_prompt(self):
        alpha = torch.randn(self.K_F) * 0.02
        self.cp_alphas.append(nn.Parameter(alpha))
        self.cp_residuals.append(nn.Parameter(torch.zeros(self.embed_dim)))

    def add_instance_prompt(self, class_id: str):
        beta = torch.randn(len(self.cp_alphas)) * 0.02 if len(self.cp_alphas) > 0 else torch.zeros(0)
        self.ip_betas[class_id] = nn.Parameter(beta)
        self.ip_residuals[class_id] = nn.Parameter(torch.zeros(self.embed_dim))

    def compute_all_prompts(self):
        device = self.fp_prompts.device
        if len(self.cp_alphas) == 0:
            cp_bank = torch.empty(0, self.embed_dim, device=device)
        else:
            alphas = torch.stack(list(self.cp_alphas))
            residuals = torch.stack(list(self.cp_residuals))
            cp_bank = torch.matmul(alphas, self.fp_prompts) + residuals
        ip_bank = []
        for cls in self.ip_betas:
            beta = self.ip_betas[cls]
            delta = self.ip_residuals[cls]
            if len(beta) == 0:
                ip = delta
            else:
                ip = torch.matmul(beta, cp_bank) + delta
            ip_bank.append(ip)
        ip_bank = torch.stack(ip_bank) if ip_bank else torch.empty(0, self.embed_dim, device=device)
        return cp_bank, ip_bank

class SparsePromptRouter(nn.Module):
    def __init__(self, embed_dim: int, top_k: int = 56):
        super().__init__()
        self.top_k = top_k
        self.W_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_k = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, img_emb: torch.Tensor, prompt_bank: torch.Tensor):
        if prompt_bank.size(0) == 0:
            return torch.empty(img_emb.size(0), 0, device=img_emb.device), torch.empty(0, dtype=torch.long, device=img_emb.device)
        Q = self.W_q(img_emb)
        K = self.W_k(prompt_bank)
        logits = torch.matmul(Q, K.T) / math.sqrt(img_emb.size(-1))
        top_vals, top_idxs = torch.topk(logits, k=min(self.top_k, logits.size(1)), dim=-1)
        attn = torch.zeros_like(logits).scatter_(-1, top_idxs, F.softmax(top_vals, dim=-1))
        return attn, top_idxs

class HPCNet(nn.Module):
    def __init__(self, initial_classes: List[str], embed_dim: int = 512, K_F: int = 16):
        super().__init__()
        self.clip = FrozenCLIP()
        self.prompt_bank = HierarchicalPromptBank(embed_dim=embed_dim, K_F=K_F)
        self.router = SparsePromptRouter(embed_dim=embed_dim)
        self.class_names = list(initial_classes)
        for cls in initial_classes:
            self.prompt_bank.add_instance_prompt(cls)

    def forward(self, images, active_classes: List[str]):
        img_emb = self.clip.encode_image(images)
        cp_bank, ip_bank = self.prompt_bank.compute_all_prompts()
        routing_weights, top_idxs = self.router(img_emb, ip_bank)
        composed_prompt = torch.matmul(routing_weights, ip_bank)
        text_embs = self.clip.encode_text_batch(active_classes)
        logits = torch.cosine_similarity(
            img_emb.unsqueeze(1),
            (text_embs + composed_prompt.unsqueeze(1)).unsqueeze(0),
            dim=-1
        )
        return logits, cp_bank, ip_bank, top_idxs

def consistency_loss(self, top_idxs, cp_bank, ip_bank):
    if cp_bank.size(0) == 0 or ip_bank.size(0) == 0:
        return torch.tensor(0.0, device=ip_bank.device)
    loss = 0.0
    # Enforce sparsity in composition vectors (Assumption A2)
    for cls in self.ip_betas:
        beta = self.ip_betas[cls]
        if len(beta) > 0:
            loss += torch.norm(beta, p=1)  # L1 sparsity on β
    for alpha in self.cp_alphas:
        loss += torch.norm(alpha, p=1)    # L1 sparsity on α
    return loss * 0.01

    def add_new_class(self, class_id: str):
        self.class_names.append(class_id)
        self.prompt_bank.add_instance_prompt(class_id)
