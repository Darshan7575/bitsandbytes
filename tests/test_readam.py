import pytest
import torch

import bitsandbytes.functional as F

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="ReAdam kernels require CUDA")

BETA1, BETA2, EPS, LR, DECAY_RATIO = 0.9, 0.999, 1e-8, 1e-2, 10.0


def _new_state(shape, optim_bits, device):
    n_rows, n_cols = shape[-2], shape[-1]
    lead = shape[:-2]
    packed = 8 // optim_bits
    numel = torch.Size(shape).numel()
    return {
        "state1": torch.zeros((numel + packed - 1) // packed, dtype=torch.uint8, device=device),
        "row": torch.zeros((*lead, n_rows, 1), dtype=torch.float32, device=device),
        "col": torch.zeros((*lead, 1, n_cols), dtype=torch.float32, device=device),
        "mean": torch.zeros((*lead, 1), dtype=torch.float32, device=device),
    }


def _step(p, g, st, optim_bits, step, weight_decay, fused, scratch):
    prev = F.READAM_FUSED
    F.READAM_FUSED = fused
    try:
        F.optimizer_update_readam(
            g,
            p,
            st["state1"],
            st["row"],
            st["col"],
            optim_bits,
            BETA1,
            BETA2,
            EPS,
            step,
            LR,
            weight_decay,
            DECAY_RATIO,
            state2_mean=st["mean"],
            stats_scratch=scratch,
        )
    finally:
        F.READAM_FUSED = prev


# (512, 768): full tiles; (4, 33, 256): batch slices whose rows do not align with the 16-row tiles;
# (1000, 520): partial column tile.
@pytest.mark.parametrize("shape", [(512, 768), (4, 33, 256), (1000, 520)], ids=str)
@pytest.mark.parametrize("optim_bits", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16], ids=str)
@pytest.mark.parametrize("weight_decay", [0.0, 0.01])
def test_readam_fused_matches_unfused(shape, optim_bits, dtype, weight_decay):
    """Each step starts both paths from identical state, so differences come only from the step itself."""
    device = "cuda"
    torch.manual_seed(0)
    assert F.readam_fused_supported(torch.empty(shape, device=device), optim_bits)

    p_ref = torch.randn(shape, device=device, dtype=dtype)
    st_ref = _new_state(shape, optim_bits, device)
    n_rows, n_cols = shape[-2], shape[-1]
    batch = p_ref.numel() // (n_rows * n_cols)
    scratch = torch.zeros(batch * (n_rows + n_cols) + 7, dtype=torch.float32, device=device)

    # the unfused kernel rounds p to the parameter dtype twice when weight_decay > 0, the fused one once: allow 1 ulp
    rtol, atol = {torch.float32: (1e-5, 1e-6), torch.float16: (1e-3, 1e-4), torch.bfloat16: (8e-3, 1e-3)}[dtype]
    for step in range(1, 6):
        g = torch.randn(shape, device=device, dtype=dtype) * (0.1 * step)
        if step == 3:
            g[..., 0, :8] = 0.0  # exact zeros
            g.view(-1)[5] = float("nan")  # skipped by both paths for the parameter update

        p_fused = p_ref.clone()
        st_fused = {k: v.clone() for k, v in st_ref.items()}

        _step(p_fused, g, st_fused, optim_bits, step, weight_decay, True, scratch)
        _step(p_ref, g, st_ref, optim_bits, step, weight_decay, False, None)
        torch.cuda.synchronize()

        assert torch.count_nonzero(scratch) == 0, "fused kernel must leave the scratch buffer zeroed"

        torch.testing.assert_close(p_fused.float(), p_ref.float(), rtol=rtol, atol=atol)
        # U_t may land on the other side of a rounding boundary when S differs in the last bit
        mismatch = (st_fused["state1"] != st_ref["state1"]).float().mean().item()
        assert mismatch < 2e-3, f"step {step}: {mismatch:.2%} of state1 bytes differ"

        if step == 3:
            # the unfused statistics propagate the NaN, the fused kernel skips it; resync and check finiteness
            assert torch.isfinite(st_fused["row"]).all() and torch.isfinite(st_fused["col"]).all()
            for k in ("row", "col", "mean"):
                st_ref[k].copy_(st_fused[k])
            continue

        for k in ("row", "col", "mean"):
            torch.testing.assert_close(st_fused[k], st_ref[k], rtol=2e-5, atol=1e-12, msg=f"step {step}: {k}")


@pytest.mark.parametrize("optim_bits", [2, 4, 8])
def test_readam_1d_updates_whole_tensor(optim_bits):
    """Regression test: the 1D launch used floor(n / block) blocks and skipped the tail of the tensor."""
    device = "cuda"
    n = 5000  # not a multiple of 256 * (8 / optim_bits)
    p = torch.randn(n, device=device)
    p0 = p.clone()
    g = torch.randn(n, device=device).abs_() + 0.1  # nonzero everywhere
    state1 = torch.zeros((n * optim_bits + 7) // 8, dtype=torch.uint8, device=device)
    state2 = torch.zeros(n, dtype=torch.uint8, device=device)
    qmap2 = F.create_dynamic_map(signed=False).to(device)
    absmax2 = torch.zeros((n + 255) // 256, dtype=torch.float32, device=device)

    F.optimizer_update_readam_1D(
        g, p, state1, state2, optim_bits, BETA1, BETA2, EPS, 1, LR, qmap2, absmax2, 0.0, DECAY_RATIO
    )
    torch.cuda.synchronize()
    assert (p != p0).all(), f"{(p == p0).sum().item()} of {n} elements were not updated"
