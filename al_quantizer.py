import os
import math
import time
import numpy as np
from sklearn.cluster import KMeans
from multiprocessing import Pool
from tqdm import tqdm

import torch
import torch.nn as nn
import transformers

from quant import *

DEBUG = False

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def kmeans_fit(row_data):
    """
    row_data: (weights_np, sample_weight, n_cluster, random_seed)
      - weights_np: shape [ncols, 1]
      - sample_weight: shape [ncols]
    """
    weights_np, sample_weight, n_cluster, random_seed = row_data
    kmeans = KMeans(
        n_clusters=n_cluster,
        init=np.linspace(weights_np.min(), weights_np.max(), num=n_cluster)[:, None] if n_cluster <= 8 else 'k-means++',
        n_init='auto',
        random_state=random_seed,
        max_iter=100,
        tol=1e-6,
    ).fit(weights_np, sample_weight=sample_weight)
    return kmeans.cluster_centers_.reshape(-1)

# Use available CPU cores for kmeans
pool = Pool(len(os.sched_getaffinity(0)))

class LeanQuant:
    """
    LeanQuant with activation-aware + loss-error-aware (Hessian) objective during search.
    - Columns correspond to input channels (after internal transpose for HF Conv1D).
    - set_input_moments() lets you provide per-input-channel second moment (sqrt(E[x^2]) recommended).
    """
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device

        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            # flatten kernels; columns = input patch dims
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            # HF Conv1D stores [in, out]; transpose so columns = in (input channels)
            W = W.t()

        self.rows = W.shape[0]
        self.columns = W.shape[1]

        # Hessian accumulator
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

        # placeholders for codebook export (kmeans path)
        self.lut = None
        self.quant_grid = None
        self.quantized_codes = None

        # activation moments (per input channel/column); and strength
        self.act_moment = None        # torch.Tensor [columns] on CPU or CUDA
        self.awq_lambda = 0.0         # float

    # ===== Activation-aware hooks =====
    def set_input_moments(self, moment: torch.Tensor, awq_lambda: float = 0.01):
        """
        moment: 1D tensor of length = in_features (i.e., number of columns after internal transforms).
        awq_lambda: strength for activation-aware term (>=0). If 0, disables.
        """
        if moment is None:
            self.act_moment = None
            self.awq_lambda = 0.0
            return
        m = moment.detach().float().view(-1).cpu()
        if m.numel() != self.columns:
            raise ValueError(f"Activation moment length mismatch: got {m.numel()}, expected {self.columns}.")
        # normalize to mean=1 to keep scale stable across layers
        m = m / (m.mean() + 1e-9)
        self.act_moment = m  # keep on CPU; we'll to(device) when needed
        self.awq_lambda = float(max(0.0, awq_lambda))

    # ===== GPTQ Hessian accumulation =====
    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)

        # EMA-like update for H = E[x x^T]
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp

        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    # ===== Main quantization (GPTQ + activation-aware search) =====
    def fasterquant(
        self, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False, args=None,
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        # Prepare Hessian and prune dead columns
        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        # Optionally precompute per-group quantizers
        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)], weight=True)
                groups.append(quantizer)

        # Offload large H to secondary GPU if needed
        if H.shape[0] >= args.offload_threshold:
            secondary_device = torch.device('cuda:1')
            H = H.to(secondary_device)

        # Activation-order heuristic as in GPTQ
        if actorder:
            perm_H = torch.argsort(torch.diag(H), descending=True)
            perm = perm_H.to(W.device)
            W = W[:, perm]
            H = H[perm_H][:, perm_H]
            invperm = torch.argsort(perm)

        # Damping + Cholesky to get H^{-1/2} type factors used in updates
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=H.device)
        H[diag, diag] += damp

        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)  # Hinv chol (upper) for block updates

        if H.shape[0] >= args.offload_threshold:
            H = H.to(self.dev)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)
        Q_codes = Q.to(torch.uint8).cpu()
        Hinv = H
        torch.cuda.empty_cache()

        # ===== Build activation column weights (on device of W) =====
        # col_weight[j] = 1 + awq_lambda * act_moment[j]_{mean-normalized}
        if (self.act_moment is not None) and (getattr(args, "awq_lambda", 0.0) > 0.0):
            awq_lambda = float(args.awq_lambda)
            act_m = self.act_moment.to(W.device)
            col_weight = (1.0 + awq_lambda * act_m).clamp(min=1e-6)
        else:
            col_weight = None

        # ===== If using k-means centroid search per row, inject BOTH Hessian & Activation weights =====
        if isinstance(args.exponent, float):
            # Per-column Hessian weight base: diag(Hinv)^{-exponent}
            base_w = (torch.diagonal(Hinv) ** (-args.exponent)).clamp(min=1e-12)
            if col_weight is not None:
                # Combine with activation weighting
                sample_weight_t = (base_w * col_weight).to('cpu').numpy()
            else:
                sample_weight_t = base_w.to('cpu').numpy()

            kmeans_tasks = []
            W_np = W.cpu().numpy()
            for j in range(W_np.shape[0]):
                # Each row sees the same per-column sample_weight
                kmeans_tasks.append((W_np[j, :, None], sample_weight_t, 2 ** args.wbits, args.kmeans_seed))
            kmeans_results = list(tqdm(pool.imap(kmeans_fit, kmeans_tasks), total=len(kmeans_tasks)))
            centroids = torch.from_numpy(np.stack(kmeans_results)).reshape(W.shape[0], 2 ** args.wbits).to(W.device)
        else:
            centroids = None

        # ===== Column-blocked GPTQ solve with activation-aware loss accounting =====
        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            # prefetch block col weights (scalar per column) for loss scaling
            if col_weight is not None:
                block_col_w = col_weight[i1:i2]  # shape [count]
            else:
                block_col_w = None

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                # dynamic per-group quant params if grouping
                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                # choose q
                if isinstance(centroids, torch.Tensor):
                    # nearest centroid (pre-searched with activation+H weights)
                    codes = torch.argmin((centroids - w[:, None]).abs(), dim=1, keepdim=True)
                    Q_codes[:, i1 + i] = codes.flatten().to(torch.uint8).cpu()
                    q = torch.gather(centroids, 1, codes).flatten()
                else:
                    # standard per-channel quantization
                    q = quantize(
                        w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                    ).flatten()

                # activation-aware + loss-error-aware loss accounting
                sqerr = (w - q) ** 2  # per-row
                if block_col_w is not None:
                    col_scalar = block_col_w[i]  # scalar for this column
                    Losses1[:, i] = (sqerr * col_scalar) / (d ** 2)
                else:
                    Losses1[:, i] = sqerr / (d ** 2)

                # classic GPTQ error feedback
                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1
                Q1[:, i] = q

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            # propagate to the rest of columns (block update)
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        # undo permutation if actorder
        if actorder:
            Q = Q[:, invperm]
            Q_codes = Q_codes[:, invperm.cpu()]

        # export grid/codes if kmeans used
        if isinstance(args.save_path, str) and isinstance(centroids, torch.Tensor):
            nrows, ncols = Q_codes.shape
            idx = torch.arange(0, ncols, 2)[None, :].repeat(nrows, 1).to(Q_codes.device)
            self.quantized_codes = torch.bitwise_or(
                torch.bitwise_left_shift(Q_codes.gather(1, idx), 4),
                Q_codes.gather(1, idx + 1)
            )
            self.quant_grid = centroids.cpu()

        # write back to real weight with original layout
        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        print('norm of difference', torch.norm(self.layer.weight.data - Q).item())
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)

        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
