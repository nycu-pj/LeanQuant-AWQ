import os
os.environ["OMP_NUM_THREADS"] = "1"  # this is necessary to parallelize the kmeans
import time

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from safetensors.torch import save_file

from al_quantizer import *
from modelutils import *
from quant import *
from leanquant import replace_with_quantizers

# activation_aware_gptq.py (drop-in)
import torch
from typing import Dict, Any, Callable, Optional, List, Tuple
import torch.nn as nn

def _is_linear_like(m: nn.Module) -> bool:
    return isinstance(m, nn.Linear) or m.__class__.__name__ == "Conv1D" or (hasattr(m, "weight") and m.weight.ndim == 2)

@torch.no_grad()
def collect_input_second_moments(model: nn.Module,
                                 calib_loader,
                                 device: torch.device,
                                 nsamples: int = 128,
                                 predicate: Optional[Callable[[nn.Module], bool]] = None,
                                 max_tokens: Optional[int] = None) -> Dict[int, torch.Tensor]:
    """
    回傳 dict: { id(module) -> moment_vector (shape [C_in], dtype=float32 on cpu) }
    用 sqrt(E[x^2]) 作為 importance (可改用 max 或 var)。
    predicate(module) 決定哪些 layer 要蒐集（預設選 linear-like).
    """
    if predicate is None:
        predicate = lambda m: _is_linear_like(m)

    modules = [m for m in model.modules() if predicate(m)]
    # prepare storage
    stats = {id(m): {"sum2": None, "count": 0, "module": m} for m in modules}

    # register pre-hooks to capture input x (before linear)
    handles = []
    def make_hook(mid):
        def hook(m, inp):
            x = inp[0]
            if x is None:
                return
            # x shape: [B, T?, C_in] or [B, C_in]
            with torch.no_grad():
                v = x.detach()
                if v.ndim == 3:
                    # collapse batch and seq
                    v2 = v.view(-1, v.shape[-1])
                elif v.ndim == 2:
                    v2 = v
                else:
                    v2 = v.view(-1, v.shape[-1])
                s2 = (v2 ** 2).sum(dim=0).to('cpu')  # sum of squares per channel
                rec = stats[mid]
                if rec["sum2"] is None:
                    rec["sum2"] = s2
                else:
                    rec["sum2"] += s2
                rec["count"] += v2.shape[0]
        return hook

    for m in modules:
        h = m.register_forward_pre_hook(make_hook(id(m)))
        handles.append(h)

    # Ensure model is on device for forward passes
    model.eval().to(device)
    # Temporarily disable cache to reduce memory usage if present
    use_cache_flag = getattr(getattr(model, 'config', object()), 'use_cache', None)
    if use_cache_flag is not None:
        model.config.use_cache = False

    seen = 0
    with torch.no_grad():
        for batch in calib_loader:
            # support dict or tensor/tuple
            if isinstance(batch, dict):
                # try to cut tokens if requested
                b = {}
                for k, v in batch.items():
                    if torch.is_tensor(v):
                        if max_tokens is not None and v.ndim >= 2:
                            v = v[..., :max_tokens]
                        b[k] = v.to(device)
                    else:
                        b[k] = v
                # forward
                try:
                    _ = model(**b)
                except TypeError:
                    # some HF versions may not accept kwargs like use_cache etc.
                    _ = model(**b)
                bs = b[next(iter(b))].shape[0]
            else:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                if max_tokens is not None and x.ndim >= 2:
                    x = x[..., :max_tokens]
                x = x.to(device)
                try:
                    _ = model(x)
                except TypeError:
                    _ = model(x)
                bs = x.shape[0]
            seen += bs
            # small safety: free cache to avoid fragmentation during calibration
            torch.cuda.empty_cache()
            if seen >= nsamples:
                break

    # remove hooks
    for h in handles:
        h.remove()

    # restore use_cache if we changed it
    if use_cache_flag is not None:
        model.config.use_cache = use_cache_flag

    # finalize moments: sqrt( sum2 / N )
    result = {}
    for mid, rec in stats.items():
        if rec["sum2"] is None:
            continue
        count = max(1, rec["count"])
        mean2 = rec["sum2"] / count
        moment = torch.sqrt(mean2).float()  # shape [C_in]
        result[mid] = moment  # CPU tensor
    return result

@torch.no_grad()
def quantize_with_activation_aware_gptq(module: nn.Module,
                                        input_moment: Optional[torch.Tensor],
                                        quantize_fn: Callable[[nn.Module, dict], Any],
                                        inplace: bool = True,
                                        scale_eps: float = 1e-6,
                                        quantize_kwargs: Optional[dict] = None) -> Any:
    """
    Fallback wrapper (kept for compatibility). In the current pipeline we inject moments
    into LeanQuant instances via set_input_moments(), so this wrapper may not be used.
    """
    if quantize_kwargs is None:
        quantize_kwargs = {}

    if input_moment is None:
        # no AWQ info -> fallback
        return quantize_fn(module, quantize_kwargs)

    # ensure input_moment on same device as weight for scaling, but do computation in fp32
    w = module.weight.data
    dev = w.device
    mvec = input_moment.to(dev).to(w.dtype)
    # avoid zeros
    mvec = torch.clamp(mvec, min=scale_eps)

    # scale columns: W_scaled[:, j] = W[:, j] * mvec[j]
    # For HF Conv1D (weight stored [in, out]) we multiply rows instead.
    is_conv1d = (module.__class__.__name__ == "Conv1D")
    if is_conv1d:
        if w.shape[0] != mvec.numel():
            raise RuntimeError("input_moment length mismatch for Conv1D")
        w.mul_(mvec.view(-1, 1))
    else:
        if w.shape[1] != mvec.numel():
            raise RuntimeError("input_moment length mismatch for Linear-like")
        w.mul_(mvec.view(1, -1))

    # call original quantizer on scaled weight
    result = quantize_fn(module, quantize_kwargs)

    # Heuristic unscale attempt (best-effort)
    try:
        wq = module.weight.data  # after quantize_fn possibly changed
        if is_conv1d:
            wq.div_(mvec.view(-1, 1))
        else:
            wq.div_(mvec.view(1, -1))
    except Exception:
        # If module was replaced by a quantized wrapper that doesn't expose .weight or weight is int,
        # leave result as-is (user may need to fold inverse into quantizer metadata).
        pass
    return result


def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model

@torch.no_grad()
def llama_sequential(model, dataloader, dev):
    print('Starting ...')

    # collect activation moments first (AWQ) so we can inject into LeanQuant objects
    # note: calibration will temporarily disable use_cache to reduce memory usage
    try:
        print('Collecting activation moments for AWQ...')
        # default max tokens for calibration — adjust if you want shorter tokens
        max_calib_tokens = 256
        moments = collect_input_second_moments(model, dataloader, dev, nsamples=args.nsamples, max_tokens=max_calib_tokens)
        print(f'Collected moments for {len(moments)} modules.')
    except Exception as e:
        # if calibration fails, fallback to empty moments dict and continue
        print('Warning: AWQ calibration failed, continuing without activation moments:', e)
        moments = {}

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    if hasattr(model.model, 'rotary_emb'):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(args.n_layers if isinstance(args.n_layers, int) else len(layers)):
        layer = layers[i].to(dev)
        quantizers = {}
        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ['self_attn.k_proj', 'self_attn.v_proj', 'self_attn.q_proj'],
                ['self_attn.o_proj'],
                ['mlp.up_proj', 'mlp.gate_proj'],
                ['mlp.down_proj']
            ]
        else:
            sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}

            leanquant = {}
            for name in subset:
                leanquant[name] = LeanQuant(subset[name])
                # inject activation moment if available
                mid = id(subset[name])
                if mid in moments:
                    try:
                        leanquant[name].set_input_moments(moments[mid], awq_lambda=args.awq_lambda)
                    except Exception as e:
                        print(f'Warning: unable to set input moments for {name}:', e)
                leanquant[name].quantizer = Quantizer()
                leanquant[name].quantizer.configure(
                    args.wbits, perchannel=True, sym=args.sym, mse=False
                )

            def add_batch(name):
                def tmp(_, inp, out):
                    leanquant[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print('Quantizing ...')

                # fasterquant will now use input_moment injected into the LeanQuant instance (if present)
                leanquant[name].fasterquant(
                    blocksize=args.block_size, percdamp=args.percdamp, groupsize=args.groupsize, actorder=args.act_order, static_groups=args.static_groups, args=args,
                )
                if isinstance(args.exponent, float):
                    quantizers[name] = (leanquant[name].quant_grid, leanquant[name].quantized_codes)
                leanquant[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layer = layer.cpu()
        replace_with_quantizers(layer, quantizers)
        layers[i] = layer
        del layer
        del leanquant
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    if isinstance(args.save_path, str):
        save_file(model.state_dict(), args.save_path)

    model.config.use_cache = use_cache

    return quantizers

@torch.no_grad()
def llama_eval(model, testenc, dev):
    print('Evaluating ...')

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    if hasattr(model.model, 'rotary_emb'):
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen):((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.nearest:
            subset = find_layers(layer)
            for name in subset:
                quantizer = Quantizer()
                quantizer.configure(
                    args.wbits, perchannel=True, sym=False, mse=False
                )
                W = subset[name].weight.data
                quantizer.find_params(W, weight=True)
                subset[name].weight.data = quantize(
                    W, quantizer.scale, quantizer.zero, quantizer.maxq
                ).to(next(iter(layer.parameters())).dtype)

        for j in range(nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[
            :, (i * model.seqlen):((i + 1) * model.seqlen)
        ][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(ppl.item())

    model.config.use_cache = use_cache


if __name__ == '__main__':
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument(
        'model', type=str,
        help='LlaMa model to load; pass location of hugginface converted checkpoint.'
    )
    parser.add_argument(
        'dataset', type=str, choices=['wikitext2', 'ptb', 'c4', 'ptb-new', 'c4-new', 'c4-full'],
        help='Where to extract calibration data from.'
    )
    parser.add_argument(
        '--seed',
        type=int, default=0, help='Seed for sampling the calibration data.'
    )
    parser.add_argument(
        '--nsamples', type=int, default=128,
        help='Number of calibration data samples.'
    )
    parser.add_argument(
        '--percdamp', type=float, default=.01,
        help='Percent of the average Hessian diagonal to use for dampening.'
    )
    parser.add_argument(
        '--nearest', action='store_true',
        help='Whether to run the RTN baseline.'
    )
    parser.add_argument(
        '--wbits', type=int, default=16, choices=[2, 3, 4, 16],
        help='#bits to use for quantization; use 16 for evaluating base model.'
    )
    parser.add_argument(
        '--groupsize', type=int, default=-1,
        help='Groupsize to use for quantization; default uses full row.'
    )
    parser.add_argument(
        '--sym', action='store_true',
        help='Whether to perform symmetric quantization.'
    )
    parser.add_argument(
        '--new-eval', action='store_true',
        help='Whether to use the new PTB and C4 eval.'
    )
    parser.add_argument(
        '--act-order', action='store_true',
        help='Whether to apply the activation order GPTQ heuristic'
    )
    parser.add_argument(
        '--true-sequential', action='store_true',
        help='Whether to run in true sequential model.'
    )
    parser.add_argument(
        '--static-groups', action='store_true',
        help='Whether to use static groups; recommended when using `--actorder` for more efficient inference.'
    )
    parser.add_argument(
        '--n_layers', type=int, default=None,
    )
    parser.add_argument(
        '--block_size', type=int, default=128,
    )
    parser.add_argument(
        '--exponent', type=float, default=None,
    )
    parser.add_argument(
        '--kmeans_seed', type=int, default=0,
    )
    parser.add_argument(
        '--offload_threshold', type=int, default=53248,
    )
    parser.add_argument(
        '--save_path', type=str, default=None,
    )
    parser.add_argument('--awq-lambda', type=float, default=0.0, help='Weight for activation-aware term in GPTQ search (0 disables).')

    args = parser.parse_args()
    print(args)

    model = get_llama(args.model)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    if args.wbits < 16 and not args.nearest:
        tick = time.time()
        quantizers = llama_sequential(model, dataloader, DEV)
        print(f'quant_time={time.time() - tick}')

    datasets = ['wikitext2']
    if args.new_eval:
        datasets = ['wikitext2']
    for dataset in datasets:
        dataloader, testloader = get_loaders(
            dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
        )
        print(dataset)
        llama_eval(model, testloader, DEV)
