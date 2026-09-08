import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import time


# =============================================================================
# Methods
# =============================================================================

def avg_pool_similarity(t_feat, v_feat):
    """Compute text-video similarity matrix via average pooling + L2 norm + einsum."""
    t_pooled = t_feat.mean(dim=1)
    v_pooled = v_feat.mean(dim=1)
    t_pooled = F.normalize(t_pooled, p=2, dim=1)
    v_pooled = F.normalize(v_pooled, p=2, dim=1)
    sim_matrix = torch.einsum('bd,cd->bc', t_pooled, v_pooled)
    return sim_matrix


class WeightedInteraction(nn.Module):
    """Learnable weighted interaction for text-video feature alignment."""

    def __init__(self, embed_size, nums_size):
        super().__init__()
        self.t_weight = nn.Linear(embed_size, 1)
        self.v_weight = nn.Linear(embed_size, 1)

    def forward(self, t_feat, v_feat):
        t_attn = F.softmax(self.t_weight(t_feat).squeeze(-1), dim=1)
        v_attn = F.softmax(self.v_weight(v_feat).squeeze(-1), dim=1)
        t_pooled = torch.einsum('bn,bne->be', t_attn, t_feat)
        v_pooled = torch.einsum('bn,bne->be', v_attn, v_feat)
        t_pooled = F.normalize(t_pooled, p=2, dim=1)
        v_pooled = F.normalize(v_pooled, p=2, dim=1)
        sim_matrix = torch.einsum('bd,cd->bc', t_pooled, v_pooled)
        return sim_matrix


class MaxWeightingInteraction(nn.Module):
    """Max-Weighting interaction for text-video feature alignment."""

    def __init__(self, embed_size, nums_size):
        super().__init__()
        self.t_weight = nn.Linear(embed_size, 1)
        self.v_weight = nn.Linear(embed_size, 1)

    def forward(self, t_feat, v_feat):
        t_feat = F.normalize(t_feat, p=2, dim=-1)
        v_feat = F.normalize(v_feat, p=2, dim=-1)
        t_w = F.softmax(self.t_weight(t_feat).squeeze(-1), dim=-1)
        v_w = F.softmax(self.v_weight(v_feat).squeeze(-1), dim=-1)
        sims = torch.einsum('tnd,vmd->tvnm', t_feat, v_feat)
        sims_t2v, _ = sims.max(dim=-1)
        sims_t2v = torch.einsum('tvn,tn->tv', sims_t2v, t_w)
        sims_v2t, _ = sims.max(dim=-2)
        sims_v2t = torch.einsum('tvm,vm->tv', sims_v2t, v_w)
        sim_matrix = (sims_t2v + sims_v2t) / 2
        return sim_matrix


class LogSumExpInteraction(nn.Module):
    """LogSumExp Pooling interaction for text-video feature alignment."""

    def __init__(self, embed_size, nums_size, init_temp=1.0):
        super().__init__()
        self.t_weight = nn.Linear(embed_size, 1)
        self.v_weight = nn.Linear(embed_size, 1)
        self.temperature = nn.Parameter(torch.tensor(init_temp, dtype=torch.float32))

    def _logsumexp_pool(self, x, dim):
        tau = F.softplus(self.temperature) + 1e-6
        return torch.logsumexp(x * tau, dim=dim) / tau

    def forward(self, t_feat, v_feat):
        t_feat = F.normalize(t_feat, p=2, dim=-1)
        v_feat = F.normalize(v_feat, p=2, dim=-1)
        t_w = F.softmax(self.t_weight(t_feat).squeeze(-1), dim=-1)
        v_w = F.softmax(self.v_weight(v_feat).squeeze(-1), dim=-1)
        sims = torch.einsum('tnd,vmd->tvnm', t_feat, v_feat)
        sims_t2v = self._logsumexp_pool(sims, dim=-1)
        sims_t2v = torch.einsum('tvn,tn->tv', sims_t2v, t_w)
        sims_v2t = self._logsumexp_pool(sims, dim=-2)
        sims_v2t = torch.einsum('tvm,vm->tv', sims_v2t, v_w)
        sim_matrix = (sims_t2v + sims_v2t) / 2
        return sim_matrix


class DTWInteraction(nn.Module):
    """Dynamic Time Warping (DTW) interaction for text-video alignment."""

    def __init__(self, embed_size, nums_size):
        super().__init__()
        self.embed_size = embed_size
        self.nums_size = nums_size

    def forward(self, t_feat, v_feat):
        t_feat = F.normalize(t_feat, p=2, dim=-1)
        v_feat = F.normalize(v_feat, p=2, dim=-1)
        B_t, N_t, D = t_feat.shape
        B_v, N_v, _ = v_feat.shape
        sims = torch.einsum('tnd,vmd->tvnm', t_feat, v_feat)
        dp = torch.zeros(B_t, B_v, N_t, N_v, device=t_feat.device)
        for i in range(N_t):
            for j in range(N_v):
                if i == 0 and j == 0:
                    dp[:, :, i, j] = sims[:, :, i, j]
                elif i == 0:
                    dp[:, :, i, j] = dp[:, :, i, j - 1] + sims[:, :, i, j]
                elif j == 0:
                    dp[:, :, i, j] = dp[:, :, i - 1, j] + sims[:, :, i, j]
                else:
                    dp[:, :, i, j] = sims[:, :, i, j] + torch.max(
                        torch.stack([dp[:, :, i - 1, j - 1], dp[:, :, i - 1, j], dp[:, :, i, j - 1]], dim=-1), dim=-1
                    )[0]
        sim_matrix = dp[:, :, -1, -1] / ((N_t + N_v) / 2.0)
        return sim_matrix


class SoftDTWInteraction(nn.Module):
    """Soft Dynamic Time Warping (Soft-DTW) interaction for text-video alignment."""

    def __init__(self, embed_size, nums_size, gamma=1.0):
        super().__init__()
        self.embed_size = embed_size
        self.nums_size = nums_size
        self.gamma = gamma

    def _soft_min(self, a, b, c):
        vals = torch.stack([a, b, c], dim=-1)
        weights = F.softmax(-vals / self.gamma, dim=-1)
        return (vals * weights).sum(dim=-1)

    def forward(self, t_feat, v_feat):
        t_feat = F.normalize(t_feat, p=2, dim=-1)
        v_feat = F.normalize(v_feat, p=2, dim=-1)
        B_t, N_t, D = t_feat.shape
        B_v, N_v, _ = v_feat.shape
        sims = torch.einsum('tnd,vmd->tvnm', t_feat, v_feat)
        R = torch.full((B_t, B_v, N_t, N_v), float('-inf'), device=t_feat.device)
        for i in range(N_t):
            for j in range(N_v):
                if i == 0 and j == 0:
                    R[:, :, i, j] = sims[:, :, i, j]
                elif i == 0:
                    R[:, :, i, j] = sims[:, :, i, j] + R[:, :, i, j - 1]
                elif j == 0:
                    R[:, :, i, j] = sims[:, :, i, j] + R[:, :, i - 1, j]
                else:
                    R[:, :, i, j] = sims[:, :, i, j] + self._soft_min(
                        R[:, :, i - 1, j - 1], R[:, :, i - 1, j], R[:, :, i, j - 1]
                    )
        sim_matrix = R[:, :, -1, -1] / ((N_t + N_v) / 2.0)
        return sim_matrix


class TopKPoolInteraction(nn.Module):
    """Top-k Pooling interaction for text-video feature alignment."""

    def __init__(self, embed_size, nums_size, k=3):
        super().__init__()
        self.embed_size = embed_size
        self.nums_size = nums_size
        self.k = k
        self.t_weight = nn.Linear(embed_size, 1)
        self.v_weight = nn.Linear(embed_size, 1)

    def forward(self, t_feat, v_feat):
        B_t, N_t, D = t_feat.shape
        B_v, N_v, _ = v_feat.shape
        k_v = min(self.k, N_v)
        k_t = min(self.k, N_t)
        t_feat = F.normalize(t_feat, p=2, dim=-1)
        v_feat = F.normalize(v_feat, p=2, dim=-1)
        t_w = F.softmax(self.t_weight(t_feat).squeeze(-1), dim=-1)
        v_w = F.softmax(self.v_weight(v_feat).squeeze(-1), dim=-1)
        sims = torch.einsum('tnd,vmd->tvnm', t_feat, v_feat)
        sims_t2v, _ = torch.topk(sims, k=k_v, dim=-1, largest=True)
        sims_t2v = sims_t2v.mean(dim=-1)
        sims_t2v = torch.einsum('tvn,tn->tv', sims_t2v, t_w)
        sims_v2t = sims.transpose(-2, -1)
        sims_v2t, _ = torch.topk(sims_v2t, k=k_t, dim=-1, largest=True)
        sims_v2t = sims_v2t.mean(dim=-1)
        sims_v2t = torch.einsum('tvm,vm->tv', sims_v2t, v_w)
        sim_matrix = (sims_t2v + sims_v2t) / 2
        return sim_matrix


class OTInteraction(nn.Module):
    """Optimal Transport (OT) interaction for text-video alignment."""

    def __init__(self, embed_size, nums_size, reg=0.05, num_iter=20):
        super().__init__()
        self.embed_size = embed_size
        self.nums_size = nums_size
        self.reg = reg
        self.num_iter = num_iter
        self.t_weight = nn.Linear(embed_size, 1)
        self.v_weight = nn.Linear(embed_size, 1)

    def forward(self, t_feat, v_feat):
        B_t, N_t, D = t_feat.shape
        B_v, N_v, _ = v_feat.shape
        t_feat = F.normalize(t_feat, p=2, dim=-1)
        v_feat = F.normalize(v_feat, p=2, dim=-1)
        t_w = F.softmax(self.t_weight(t_feat).squeeze(-1), dim=-1)
        v_w = F.softmax(self.v_weight(v_feat).squeeze(-1), dim=-1)
        sims = torch.einsum('tnd,vmd->tvnm', t_feat, v_feat)
        cost = 1.0 - sims
        K = torch.exp(-cost / self.reg).clamp(min=1e-8)
        u = torch.ones(B_t, B_v, N_t, device=t_feat.device)
        v_vec = torch.ones(B_t, B_v, N_v, device=t_feat.device)
        for _ in range(self.num_iter):
            u = 1.0 / (K @ v_vec.unsqueeze(-1)).squeeze(-1).clamp(min=1e-8)
            v_vec = 1.0 / (K.transpose(-2, -1) @ u.unsqueeze(-1)).squeeze(-1).clamp(min=1e-8)
        P = u.unsqueeze(-1) * K * v_vec.unsqueeze(-2)
        ot_cost = (P * cost).sum(dim=(-2, -1))
        sim_matrix = 1.0 / (1.0 + ot_cost)
        return sim_matrix


# =============================================================================
# Benchmark
# =============================================================================

def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def estimate_mflops(method_name, batch_size, nums_size, embed_size):
    B, N, D = batch_size, nums_size, embed_size
    B2 = B * B
    if method_name == "avg_pool_similarity":
        return (B * N * D + B * D + B2 * D) / 1e6
    elif method_name in ("WeightedInteraction",):
        return (4 * B * N * D + 2 * B * N * D + 2 * B * D + B2 * D) / 1e6
    elif method_name in ("MaxWeightingInteraction", "TopKPoolInteraction", "LogSumExpInteraction"):
        return (2 * B * N * D + B2 * N * N * D + 2 * B2 * N + 2 * B2 * N) / 1e6
    elif method_name == "DTWInteraction":
        return (2 * B * N * D + B2 * N * N * D + B2 * N * N * 15) / 1e6
    elif method_name == "SoftDTWInteraction":
        return (2 * B * N * D + B2 * N * N * D + B2 * N * N * 40) / 1e6
    elif method_name == "OTInteraction":
        return (2 * B * N * D + B2 * N * N * D + 40 * B2 * N * N) / 1e6
    return 0.0


def benchmark_method(method_name, method, dataloader, device):
    param_count = count_parameters(method) if isinstance(method, nn.Module) else 0
    batch_size = next(iter(dataloader))[0].shape[0]
    mflops = estimate_mflops(method_name, batch_size, 12, 512)

    for t_batch, v_batch in dataloader:
        t_batch = t_batch.to(device, non_blocking=True)
        v_batch = v_batch.to(device, non_blocking=True)
        _ = method(t_batch, v_batch)
        break
    torch.cuda.synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    torch.cuda.synchronize(device)
    start = time.perf_counter()

    with torch.no_grad():
        for t_batch, v_batch in dataloader:
            t_batch = t_batch.to(device, non_blocking=True)
            v_batch = v_batch.to(device, non_blocking=True)
            _ = method(t_batch, v_batch)

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    peak_mb = 0.0
    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    print(f"  [{method_name}] Time: {elapsed:.4f} s | MFLOPs: {mflops:.2f} | Params: {param_count:,} | Peak: {peak_mb:.1f} MB")

    return {"time": elapsed, "mflops": mflops, "params": param_count, "peak_mb": peak_mb}


def format_table(results):
    sorted_items = results.items()

    w_name = 30
    w_time = 10
    w_mflops = 10
    w_params = 12
    w_peak = 10

    header = (
        f"{'Method':{w_name}} | {'Time':>{w_time}} | {'MFLOPs':>{w_mflops}} | "
        f"{'Params':>{w_params}} | {'Peak Mem':>{w_peak}}"
    )
    sep = "-" * len(header)

    print("\n" + "=" * len(header))
    print("Benchmark Report")
    print("=" * len(header))
    print(header)
    print(sep)

    for name, r in sorted_items:
        time_str = f"{r['time']:.4f}s"
        mflops_str = f"{r['mflops']:.2f}"
        params_str = f"{r['params']:,}"
        peak_str = f"{r['peak_mb']:.1f}MB"
        line = (
            f"{name:{w_name}} | {time_str:>{w_time}} | {mflops_str:>{w_mflops}} | "
            f"{params_str:>{w_params}} | {peak_str:>{w_peak}}"
        )
        print(line)

    print("=" * len(header))


def main():
    total_samples = 32
    nums_size_t, nums_size_v = 64, 64
    embed_size = 512
    batch_size = 32
    num_workers = 4
    pin_memory = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Total samples: {total_samples}, batch_size: {batch_size}\n")

    t_data = torch.rand(total_samples, nums_size_t, embed_size)
    v_data = torch.rand(total_samples, nums_size_v, embed_size)
    dataset = TensorDataset(t_data, v_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    results = {}

    print("Benchmarking avg_pool_similarity ...")
    results["avg_pool_similarity"] = benchmark_method("avg_pool_similarity", avg_pool_similarity, dataloader, device)

    print("Benchmarking WeightedInteraction ...")
    results["WeightedInteraction"] = benchmark_method("WeightedInteraction", WeightedInteraction(embed_size, nums_size_t).to(device).eval(), dataloader, device)

    print("Benchmarking MaxWeightingInteraction ...")
    results["MaxWeightingInteraction"] = benchmark_method("MaxWeightingInteraction", MaxWeightingInteraction(embed_size, nums_size_t).to(device).eval(), dataloader, device)

    print("Benchmarking TopKPoolInteraction ...")
    results["TopKPoolInteraction"] = benchmark_method("TopKPoolInteraction", TopKPoolInteraction(embed_size, nums_size_t, k=3).to(device).eval(), dataloader, device)

    print("Benchmarking LogSumExpInteraction ...")
    results["LogSumExpInteraction"] = benchmark_method("LogSumExpInteraction", LogSumExpInteraction(embed_size, nums_size_t).to(device).eval(), dataloader, device)

    print("Benchmarking DTWInteraction ...")
    results["DTWInteraction"] = benchmark_method("DTWInteraction", DTWInteraction(embed_size, nums_size_t).to(device).eval(), dataloader, device)

    print("Benchmarking SoftDTWInteraction ...")
    results["SoftDTWInteraction"] = benchmark_method("SoftDTWInteraction", SoftDTWInteraction(embed_size, nums_size_t, gamma=1.0).to(device).eval(), dataloader, device)

    print("Benchmarking OTInteraction ...")
    results["OTInteraction"] = benchmark_method("OTInteraction", OTInteraction(embed_size, nums_size_t, reg=0.05, num_iter=20).to(device).eval(), dataloader, device)

    format_table(results)

if __name__ == "__main__":
    main()
