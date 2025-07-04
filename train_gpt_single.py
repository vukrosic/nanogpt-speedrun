import os
import sys
with open(sys.argv[0]) as f:
    code = f.read() # read the code of this file ASAP, for logging
import uuid
import time
import copy
import glob
import signal
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
torch.empty(1, device="cuda", requires_grad=True).backward() # prevents a bug on some systems
from torch import Tensor, nn
import torch.nn.functional as F
# use of FlexAttention contributed by @KoszarskyB
from torch.nn.attention.flex_attention import BlockMask, flex_attention

print(f"[DEBUG] Initial setup complete, CUDA available: {torch.cuda.is_available()}")

# -----------------------------------------------------------------------------
# Custom operators: FP8 matmul by @YouJiacheng

@torch.library.custom_op("nanogpt::mm", mutates_args=())
def mm_op(x: Tensor, w: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor, Tensor]:
    @torch.compile
    def impl(x: Tensor, w: Tensor):
        assert x.is_contiguous() and w.is_contiguous()
        x_f8 = x.div(x_s).to(torch.float8_e4m3fn)
        w_f8 = w.div(w_s).to(torch.float8_e4m3fn)
        out = torch._scaled_mm(
            x_f8,
            w_f8.T,
            out_dtype=torch.bfloat16,
            scale_a=x.new_tensor(x_s, dtype=torch.float32),
            scale_b=x.new_tensor(w_s, dtype=torch.float32),
            use_fast_accum=True,
        )
        return out, x_f8, w_f8

    return impl(x, w)

@mm_op.register_fake
def _(x: Tensor, w: Tensor, *_):
    assert x.ndim == w.ndim == 2
    assert x.shape[1] == w.shape[1]
    assert x.device == w.device
    assert x.is_contiguous() and w.is_contiguous()
    return x @ w.T, x.to(torch.float8_e4m3fn), w.to(torch.float8_e4m3fn)

@torch.library.custom_op("nanogpt::mm_backward", mutates_args=())
def mm_backward_op(g: Tensor, x_f8: Tensor, w_f8: Tensor, x_s: float, w_s: float, grad_s: float) -> tuple[Tensor, Tensor]:
    @torch.compile
    def impl(grad: Tensor, x_f8: Tensor, w_f8: Tensor):
        assert grad.is_contiguous()
        x_inv_s = grad.new_tensor(x_s, dtype=torch.float32)
        w_inv_s = grad.new_tensor(w_s, dtype=torch.float32)
        grad_inv_s = grad.new_tensor(grad_s, dtype=torch.float32)
        grad_f8 = grad.div(grad_s).to(torch.float8_e5m2)
        grad_x = torch._scaled_mm(
            grad_f8,
            w_f8.T.contiguous().T,
            out_dtype=torch.bfloat16,
            scale_a=grad_inv_s,
            scale_b=w_inv_s,
            use_fast_accum=False,
        )
        grad_w = torch._scaled_mm(
            x_f8.T.contiguous(),
            grad_f8.T.contiguous().T,
            out_dtype=torch.float32,
            scale_a=x_inv_s,
            scale_b=grad_inv_s,
            use_fast_accum=False,
        ).T
        return grad_x, grad_w

    return impl(g, x_f8, w_f8)

@mm_backward_op.register_fake
def _(g: Tensor, x_f8: Tensor, w_f8: Tensor, *_):
    return x_f8.to(torch.bfloat16), w_f8.T.contiguous().T.to(torch.float32)

def backward(ctx, grad_out: Tensor, *_):
    x_f8, w_f8 = ctx.saved_tensors
    x_s, w_s, grad_s = ctx.scales
    grad_x, grad_w = torch.ops.nanogpt.mm_backward(
        grad_out, x_f8, w_f8, x_s, w_s, grad_s
    )
    return grad_x, grad_w, None, None, None

def setup_context(ctx: torch.autograd.function.FunctionCtx, inputs, output):
    *_, x_s, w_s, grad_s = inputs
    _, x_f8, w_f8 = output
    ctx.save_for_backward(x_f8, w_f8)
    ctx.scales = x_s, w_s, grad_s
    ctx.set_materialize_grads(False)

mm_op.register_autograd(backward, setup_context=setup_context)

print(f"[DEBUG] Custom operators registered")

# -----------------------------------------------------------------------------
# Simplified Muon optimizer (single GPU)

@torch.compile
def zeropower_via_newtonschulz5(G: Tensor, steps: int) -> Tensor:
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X

    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                    
                g = p.grad
                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf: Tensor = state["momentum_buffer"]
                buf.lerp_(g, 1 - group["momentum"])
                g = g.lerp_(buf, group["momentum"]) if group["nesterov"] else buf
                g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                p.add_(g, alpha=-group["lr"] * max(1, p.size(-2) / p.size(-1))**0.5)

print(f"[DEBUG] Muon optimizer defined")

# -----------------------------------------------------------------------------
# PyTorch nn.Module definitions for the model

def norm(x: Tensor):
    return F.rms_norm(x, (x.size(-1),))

class CastedLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, use_fp8=False, x_s=1.0, w_s=1.0, grad_s=1.0):
        super().__init__(in_features, out_features, bias=False)
        self.use_fp8 = use_fp8
        self.x_s = x_s
        self.w_s = w_s
        self.grad_s = grad_s

    def reset_parameters(self) -> None:
        std = 0.5 * (self.in_features ** -0.5)
        bound = (3 ** 0.5) * std
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

    def forward(self, x: Tensor):
        if self.use_fp8 and self.training:
            _x = x.flatten(0, -2)
            out: Tensor = torch.ops.nanogpt.mm(_x, self.weight, x_s=self.x_s, w_s=self.w_s, grad_s=self.grad_s)[0]
            return out.reshape(*x.shape[:-1], -1)
        else:
            return F.linear(x, self.weight.type_as(x))

class Rotary(nn.Module):
    def __init__(self, dim: int, max_seq_len: int):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim//4, dtype=torch.float32)
        angular_freq = torch.cat([angular_freq, angular_freq.new_zeros(dim//4)])
        t = torch.arange(max_seq_len, dtype=torch.float32)
        theta = torch.einsum("i,j -> ij", t, angular_freq)
        self.cos = nn.Buffer(theta.cos(), persistent=False)
        self.sin = nn.Buffer(theta.sin(), persistent=False)

    def forward(self, x_BTHD: Tensor):
        assert self.cos.size(0) >= x_BTHD.size(-3)
        cos, sin = self.cos[None, :x_BTHD.size(-3), None, :], self.sin[None, :x_BTHD.size(-3), None, :]
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        y1 = x1 * cos + x2 * sin
        y2 = x1 * (-sin) + x2 * cos
        return torch.cat((y1, y2), 3).type_as(x_BTHD)

class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        hdim = num_heads * head_dim
        std = 0.5 * (dim ** -0.5)
        bound = (3 ** 0.5) * std
        self.qkv_w = nn.Parameter(torch.empty(3, hdim, dim).uniform_(-bound, bound))
        self.lambdas = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.rotary = Rotary(head_dim, max_seq_len)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_()

    def forward(self, x: Tensor, ve: Tensor | None, block_mask: BlockMask):
        B, T = x.size(0), x.size(1)
        assert B == 1, "Must use batch size = 1 for FlexAttention"
        q, k, v = F.linear(x, self.qkv_w.flatten(end_dim=1).type_as(x)).view(B, T, 3 * self.num_heads, self.head_dim).chunk(3, dim=-2)
        q, k = norm(q), norm(k)
        q, k = self.rotary(q), self.rotary(k)
        if ve is not None:
            v = self.lambdas[0] * v + self.lambdas[1] * ve.view_as(v)
        else:
            v = self.lambdas[0] * v
        y = flex_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), block_mask=block_mask, scale=0.12).transpose(1, 2)
        y = y.contiguous().view(B, T, self.num_heads * self.head_dim)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        hdim = 4 * dim
        self.c_fc = CastedLinear(dim, hdim)
        self.c_proj = CastedLinear(hdim, dim)
        self.c_proj.weight.detach().zero_()

    def forward(self, x: Tensor):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, max_seq_len: int, layer_idx: int):
        super().__init__()
        self.attn = CausalSelfAttention(dim, num_heads, max_seq_len) if layer_idx != 7 else None
        self.mlp = MLP(dim)
        self.lambdas = nn.Parameter(torch.tensor([1., 0.]))

    def forward(self, x: Tensor, ve: Tensor | None, x0: Tensor, block_mask: BlockMask):
        x = self.lambdas[0] * x + self.lambdas[1] * x0
        if self.attn is not None:
            x = x + self.attn(norm(x), ve, block_mask)
        x = x + self.mlp(norm(x))
        return x

print(f"[DEBUG] Model layers defined")

# -----------------------------------------------------------------------------
# The main model

def next_multiple_of_n(v: float | int, *, n: int):
    return next(x for x in range(n, int(v) + 1 + n, n) if x >= v)

class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, num_heads: int, model_dim: int, max_seq_len: int):
        super().__init__()
        print(f"[DEBUG] Initializing GPT model: vocab_size={vocab_size}, num_layers={num_layers}, num_heads={num_heads}, model_dim={model_dim}, max_seq_len={max_seq_len}")
        self.embed = nn.Embedding(vocab_size, model_dim)
        self.value_embeds = nn.ModuleList([nn.Embedding(vocab_size, model_dim) for _ in range(3)])
        self.blocks = nn.ModuleList([Block(model_dim, num_heads, max_seq_len, i) for i in range(num_layers)])
        self.lm_head = CastedLinear(model_dim, next_multiple_of_n(vocab_size, n=128),
                                    use_fp8=False, x_s=(model_dim**0.5)/448, w_s=24/448, grad_s=1/448)  # Disabled FP8 for T4
        self.lm_head.weight.detach().zero_()
        assert num_layers % 2 == 0
        self.skip_weights = nn.Parameter(torch.ones(num_layers//2))
        print(f"[DEBUG] GPT model initialized")

    def create_blockmasks(self, input_seq: Tensor, sliding_window_num_blocks: Tensor):
        print(f"[DEBUG] Creating blockmasks for seq_len={len(input_seq)}, window_blocks={sliding_window_num_blocks.item()}")
        t_start = time.perf_counter()
        
        BLOCK_SIZE = 128
        docs = (input_seq == 50256).cumsum(0)

        def document_causal(b, h, q_idx, kv_idx):
            causal_mask = q_idx >= kv_idx
            document_mask = docs[q_idx] == docs[kv_idx]
            return causal_mask & document_mask

        def dense_to_ordered(dense_blockmask: Tensor):
            num_blocks = dense_blockmask.sum(dim=-1, dtype=torch.int32)
            indices = dense_blockmask.argsort(dim=-1, descending=False, stable=True).flip(-1).to(torch.int32)
            return num_blocks[None, None].contiguous(), indices[None, None].contiguous()

        assert len(input_seq) % BLOCK_SIZE == 0
        NUM_BLOCKS = len(input_seq) // BLOCK_SIZE
        print(f"[DEBUG] NUM_BLOCKS={NUM_BLOCKS}")
        
        block_idx = torch.arange(NUM_BLOCKS, dtype=torch.int32, device="cuda")
        causal_blockmask_any = block_idx[:, None] >= block_idx
        causal_blockmask_all = block_idx[:, None] > block_idx
        docs_low = docs.view(-1, BLOCK_SIZE)[:, 0].contiguous()
        docs_high = docs.view(-1, BLOCK_SIZE)[:, -1].contiguous()
        document_blockmask_any = (docs_low[:, None] <= docs_high) & (docs_high[:, None] >= docs_low)
        document_blockmask_all = (docs_low[:, None] == docs_high) & (docs_high[:, None] == docs_low)
        blockmask_any = causal_blockmask_any & document_blockmask_any
        blockmask_all = causal_blockmask_all & document_blockmask_all
        partial_kv_num_blocks, partial_kv_indices = dense_to_ordered(blockmask_any & ~blockmask_all)
        full_kv_num_blocks, full_kv_indices = dense_to_ordered(blockmask_all)
        
        print(f"[DEBUG] Dense operations complete, creating BlockMask...")
        
        def build_bm(window_size_blocks: Tensor) -> BlockMask:
            return BlockMask.from_kv_blocks(
                torch.clamp_max(partial_kv_num_blocks, torch.clamp_min(window_size_blocks - full_kv_num_blocks, 1)),
                partial_kv_indices,
                torch.clamp_max(full_kv_num_blocks, window_size_blocks - 1),
                full_kv_indices,
                BLOCK_SIZE=BLOCK_SIZE,
                mask_mod=document_causal,
            )
        
        long_bm = build_bm(sliding_window_num_blocks)
        short_bm = build_bm(sliding_window_num_blocks // 2)
        
        t_end = time.perf_counter()
        print(f"[DEBUG] Blockmasks created in {(t_end - t_start) * 1000:.1f}ms")
        return long_bm, short_bm

    def forward(self, input_seq: Tensor, target_seq: Tensor, sliding_window_num_blocks: Tensor):
        print(f"[DEBUG] Model forward: input_seq.shape={input_seq.shape}")
        assert input_seq.ndim == 1

        print(f"[DEBUG] Creating value embeddings...")
        ve = [value_embed(input_seq) for value_embed in self.value_embeds]
        ve = [ve[0], ve[1], ve[2]] + [None] * (len(self.blocks) - 6) + [ve[0], ve[1], ve[2]]
        assert len(ve) == len(self.blocks)

        print(f"[DEBUG] Creating block masks...")
        long_bm, short_bm = self.create_blockmasks(input_seq, sliding_window_num_blocks)
        block_masks = [long_bm, short_bm, short_bm, short_bm, long_bm, short_bm, short_bm, long_bm, short_bm, short_bm, short_bm, long_bm]
        assert len(block_masks) == len(self.blocks)

        print(f"[DEBUG] Running embeddings...")
        x = x0 = norm(self.embed(input_seq)[None])

        print(f"[DEBUG] Running transformer blocks...")
        skip_connections = []
        n = len(self.skip_weights)
        for i in range(len(self.blocks)):
            print(f"[DEBUG] Block {i}/{len(self.blocks)}")
            if i >= n:
                x = x + self.skip_weights[i - n] * skip_connections.pop()
            x = self.blocks[i](x, ve[i], x0, block_masks[i])
            if i < n:
                skip_connections.append(x)

        print(f"[DEBUG] Running final norm and lm_head...")
        x = norm(x)
        logits = self.lm_head(x).float()
        logits = 30 * torch.sigmoid(logits / (7.5 * x.size(-1)**0.5))
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_seq, reduction='sum' if self.training else 'mean')
        print(f"[DEBUG] Forward complete, loss={loss.item():.4f}")
        return loss

print(f"[DEBUG] GPT class defined")

# -----------------------------------------------------------------------------
# Simple Data Loader (single GPU)

def _load_data_shard(file: Path):
    header = torch.from_file(str(file), False, 256, dtype=torch.int32)
    assert header[0] == 20240520, "magic number mismatch in the data .bin file"
    assert header[1] == 1, "unsupported version"
    num_tokens = int(header[2])
    with file.open("rb", buffering=0) as f:
        tokens = torch.empty(num_tokens, dtype=torch.uint16, pin_memory=True)
        f.seek(256 * 4)
        nbytes = f.readinto(tokens.numpy())
        assert nbytes == 2 * num_tokens, "number of tokens read does not match header"
    return tokens

def simple_data_generator(filename_pattern: str, seq_len: int):
    files = [Path(file) for file in sorted(glob.glob(filename_pattern))]
    file_iter = iter(files)
    tokens, pos = _load_data_shard(next(file_iter)), 0
    while True:
        if pos + seq_len + 1 >= len(tokens):
            tokens, pos = _load_data_shard(next(file_iter)), 0
        buf = tokens[pos:pos + seq_len + 1]
        inputs = buf[:-1].to(device="cuda", dtype=torch.int32, non_blocking=True)
        targets = buf[1:].to(device="cuda", dtype=torch.int64, non_blocking=True)
        pos += seq_len
        yield inputs, targets

print(f"[DEBUG] Data loader defined")

# -----------------------------------------------------------------------------
# Timeout handler for compilation

class CompilationTimeout(Exception):
    pass

def timeout_handler(signum, frame):
    raise CompilationTimeout("Model compilation timed out")

# -----------------------------------------------------------------------------
# int main

@dataclass
class Hyperparameters:
    # data
    train_files = "data/fineweb10B/fineweb_train_*.bin"
    val_files = "data/fineweb10B/fineweb_val_*.bin"
    val_tokens = 1048576  # Reduced for T4
    train_seq_len = 4096  # Much smaller for T4 - was 48*1024
    val_seq_len = 8192    # Smaller for T4 - was 4*64*1024
    # optimization
    num_iterations = 1000  # Reduced
    cooldown_frac = 0.4
    # architecture
    vocab_size = 50257
    # evaluation and logging
    val_loss_every = 100  # More frequent validation
    save_checkpoint = False

args = Hyperparameters()

# Single GPU setup
assert torch.cuda.is_available()
device = torch.device("cuda:0")
torch.cuda.set_device(device)
master_process = True

print(f"[DEBUG] Using device: {device}")
print(f"[DEBUG] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

# begin logging
run_id = uuid.uuid4()
os.makedirs("logs", exist_ok=True)
logfile = f"logs/{run_id}.txt"
print(f"[DEBUG] Logging to: {logfile}")

def print0(s, console=False):
    with open(logfile, "a") as f:
        if console:
            print(s)
        print(s, file=f)

# begin by printing this file (the Python code)
print0(code)
print0("="*100)
print0(f"Running Python {sys.version}")
print0(f"Running PyTorch {torch.version.__version__} compiled for CUDA {torch.version.cuda}")
print0("="*100)

########################################
#    Construct model and optimizer     #
########################################

print0("[DEBUG] Creating model...", console=True)
# Smaller model for T4
model: nn.Module = GPT(vocab_size=args.vocab_size, num_layers=8, num_heads=4, model_dim=512,  # Reduced from 12/6/768
                       max_seq_len=max(args.train_seq_len, args.val_seq_len)).cuda()

print0("[DEBUG] Converting embeddings to bfloat16...", console=True)
for m in model.modules():
    if isinstance(m, nn.Embedding):
        m.bfloat16()

print0("[DEBUG] Setting up optimizers...", console=True)
# collect the parameters to optimize
hidden_matrix_params = [p for n, p in model.blocks.named_parameters() if p.ndim >= 2 and "embed" not in n]
embed_params = [p for n, p in model.named_parameters() if "embed" in n]
scalar_params = [p for p in model.parameters() if p.ndim < 2]
head_params = [model.lm_head.weight]

# init the optimizer(s) with reduced learning rates
adam_params = [dict(params=head_params, lr=0.1), dict(params=embed_params, lr=0.3), dict(params=scalar_params, lr=0.02)]
optimizer1 = torch.optim.Adam(adam_params, betas=(0.8, 0.95), eps=1e-10, fused=True)
optimizer2 = Muon(hidden_matrix_params, lr=0.025, momentum=0.95)
optimizers = [optimizer1, optimizer2]
for opt in optimizers:
    for group in opt.param_groups:
        group["initial_lr"] = group["lr"]

print0("[DEBUG] Setting up learning rate schedules...", console=True)
# learning rate schedule
def get_lr(step: int):
    x = step / args.num_iterations
    assert 0 <= x < 1
    if x < 1 - args.cooldown_frac:
        return 1.0
    else:
        w = (1 - x) / args.cooldown_frac
        return w * 1.0 + (1 - w) * 0.1

# attention window size schedule
@lru_cache(1)
def get_window_size_blocks_helper(window_size: int):
    return torch.tensor(window_size // 128, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)

def get_window_size_blocks(step: int):
    x = step / args.num_iterations
    assert 0 <= x <= 1
    # Smaller window for T4
    window_size = next_multiple_of_n(512 * x, n=128)  # Much smaller than original 1728
    return get_window_size_blocks_helper(window_size)

# Debug and compilation settings
print0("[DEBUG] Setting up compilation debugging...", console=True)
import torch._dynamo
torch._dynamo.config.verbose = True

print0("[DEBUG] About to compile model - this may take a while...", console=True)
compilation_start = time.perf_counter()

# Make compilation optional for debugging
USE_COMPILE = True  # Set to False to disable compilation for debugging
COMPILATION_TIMEOUT = 300  # 5 minutes timeout

if USE_COMPILE:
    try:
        # Set up timeout
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(COMPILATION_TIMEOUT)
        
        print0(f"[DEBUG] Starting compilation with {COMPILATION_TIMEOUT}s timeout...", console=True)
        model: nn.Module = torch.compile(model, dynamic=False)
        
        # Cancel timeout
        signal.alarm(0)
        
        compilation_time = time.perf_counter() - compilation_start
        print0(f"[DEBUG] Model compiled successfully in {compilation_time:.1f}s", console=True)
        
    except CompilationTimeout:
        signal.alarm(0)  # Cancel timeout
        print0("[ERROR] Model compilation timed out! Falling back to non-compiled model.", console=True)
        USE_COMPILE = False
    except Exception as e:
        signal.alarm(0)  # Cancel timeout
        print0(f"[ERROR] Model compilation failed: {e}. Falling back to non-compiled model.", console=True)
        USE_COMPILE = False

if not USE_COMPILE:
    print0("[DEBUG] Using non-compiled model", console=True)

########################################
#            Warmup kernels            #
########################################

print0("[DEBUG] Starting warmup phase...", console=True)
warmup_steps = 3  # Further reduced warmup for debugging
initial_state = dict(model=copy.deepcopy(model.state_dict()),
                     optimizers=[copy.deepcopy(opt.state_dict()) for opt in optimizers])

for warmup_step in range(warmup_steps):
    print0(f"[DEBUG] Warmup step {warmup_step + 1}/{warmup_steps}...", console=True)
    warmup_start = time.perf_counter()
    
    inputs = targets = torch.randint(0, args.vocab_size, size=(args.train_seq_len,), device="cuda")
    print0(f"[DEBUG] Created random tensors in {(time.perf_counter() - warmup_start) * 1000:.1f}ms", console=True)
    
    forward_start = time.perf_counter()
    loss = model(inputs.to(torch.int32), targets, get_window_size_blocks(0))
    print0(f"[DEBUG] Forward pass took {(time.perf_counter() - forward_start) * 1000:.1f}ms", console=True)
    
    backward_start = time.perf_counter()
    loss.backward()
    print0(f"[DEBUG] Backward pass took {(time.perf_counter() - backward_start) * 1000:.1f}ms", console=True)
    
    opt_start = time.perf_counter()
    for opt in optimizers:
        opt.step()
    print0(f"[DEBUG] Optimizer step took {(time.perf_counter() - opt_start) * 1000:.1f}ms", console=True)
    
    model.zero_grad(set_to_none=True)
    print0(f"[DEBUG] Warmup step {warmup_step + 1} completed in {(time.perf_counter() - warmup_start) * 1000:.1f}ms", console=True)

print0("[DEBUG] Restoring initial state...", console=True)
model.load_state_dict(initial_state["model"])
for opt, opt_state in zip(optimizers, initial_state["optimizers"]):
    opt.load_state_dict(opt_state)
del initial_state

print0("[DEBUG] Warmup complete!", console=True)

########################################
#        Training and validation       #
########################################

print0("[DEBUG] Starting training...", console=True)
train_loader = simple_data_generator(args.train_files, args.train_seq_len)
training_time_ms = 0
torch.cuda.synchronize()
t0 = time.perf_counter()

train_steps = args.num_iterations
for step in range(train_steps + 1):
    last_step = (step == train_steps)

    # --------------- VALIDATION SECTION -----------------
    if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
        torch.cuda.synchronize()
        training_time_ms += 1000 * (time.perf_counter() - t0)
        print0(f"[DEBUG] Starting validation at step {step}...", console=True)
        model.eval()
        val_steps = args.val_tokens // args.val_seq_len
        val_loader = simple_data_generator(args.val_files, args.val_seq_len)
        val_loss = 0
        with torch.no_grad():
            for val_step in range(val_steps):
                if val_step % 10 == 0:
                    print0(f"[DEBUG] Validation step {val_step}/{val_steps}", console=True)
                inputs, targets = next(val_loader)
                val_loss += model(inputs, targets, get_window_size_blocks(step))
        val_loss /= val_steps
        del val_loader
        print0(f"step:{step}/{train_steps} val_loss:{val_loss:.4f} train_time:{training_time_ms:.0f}ms step_avg:{training_time_ms/max(step, 1):.2f}ms", console=True)
        model.train()
        torch.cuda.synchronize()
        t0 = time.perf_counter()

    if last_step:
        if args.save_checkpoint:
            log = dict(step=step, code=code, model=model.state_dict(), optimizers=[opt.state_dict() for opt in optimizers])
            os.makedirs(f"logs/{run_id}", exist_ok=True)
            torch.save(log, f"logs/{run_id}/state_step{step:06d}.pt")
        break

    # --------------- TRAINING SECTION -----------------
    print0(f"[DEBUG] Training step {step + 1}/{train_steps}...", console=True)
    step_start = time.perf_counter()
    
    data_start = time.perf_counter()
    inputs, targets = next(train_loader)
    print0(f"[DEBUG] Data loading took {(time.perf_counter() - data_start) * 1000:.1f}ms", console=True)
    
    forward_start = time.perf_counter()
    loss = model(inputs, targets, get_window_size_blocks(step))
    print0(f"[DEBUG] Forward pass took {(time.perf_counter() - forward_start) * 1000:.1f}ms", console=True)
    
    backward_start = time.perf_counter()
    loss.backward()
    print0(f"[DEBUG] Backward pass took {(time.perf_counter() - backward_start) * 1000:.1f}ms", console=True)

    # set optimization hyperparameters
    for opt in optimizers:
        for group in opt.param_groups:
            group["lr"] = group["initial_lr"] * get_lr(step)
    for group in optimizer2.param_groups:
        frac = min(step / 300, 1)
        group["momentum"] = (1 - frac) * 0.85 + frac * 0.95
    
    # step the optimizers
    opt_start = time.perf_counter()
    for opt in optimizers:
        opt.step()
    print0(f"[DEBUG] Optimizer step took {(time.perf_counter() - opt_start) * 1000:.1f}ms", console=True)
    
    model.zero_grad(set_to_none=True)
    
    step_time = (time.perf_counter() - step_start) * 1000
    print0(f"[DEBUG] Full step took {step_time:.1f}ms", console=True)
    
    # logging
    approx_training_time_ms = training_time_ms + 1000 * (time.perf_counter() - t0)
    print0(f"step:{step+1}/{train_steps} train_time:{approx_training_time_ms:.0f}ms step_avg:{approx_training_time_ms/(step + 1):.2f}ms", console=True)

print0(f"peak memory allocated: {torch.cuda.max_memory_allocated() // 1024 // 1024} MiB "
       f"reserved: {torch.cuda.max_memory_reserved() // 1024 // 1024} MiB", console=True)