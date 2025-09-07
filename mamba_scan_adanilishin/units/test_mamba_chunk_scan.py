import torch
import pytest

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined as ref_mamba_chunk_scan_combined
from mamba_scan_adanilishin import mamba_chunk_scan_combined as my_mamba_chunk_scan_combined


@pytest.mark.parametrize("batch,seqlen,nheads,headdim,ngroups,dstate,chunk_size", [
    (2, 64, 4, 8, 2, 16, 16),
])
def test_mamba_chunk_scan_equivalence(batch, seqlen, nheads, headdim, ngroups, dstate, chunk_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    x = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=dtype) * 0.1
    dt = torch.rand(batch, seqlen, nheads, device=device, dtype=dtype) * 0.1
    A = torch.randn(nheads, device=device, dtype=dtype) * 0.1
    B = torch.randn(batch, seqlen, ngroups, dstate, device=device, dtype=dtype) * 0.1
    C = torch.randn(batch, seqlen, ngroups, dstate, device=device, dtype=dtype) * 0.1
    D = torch.randn(nheads, headdim, device=device, dtype=dtype) * 0.1
    z = torch.randn(batch, seqlen, nheads, headdim, device=device, dtype=dtype) * 0.1
    dt_bias = torch.randn(nheads, device=device, dtype=dtype) * 0.1

    print("Shapes:")
    print("x:", x.shape, "dt:", dt.shape, "A:", A.shape, "B:", B.shape, "C:", C.shape, "D:", D.shape, "z:", z.shape)

    try:
        out_ref = ref_mamba_chunk_scan_combined(
            x, dt, A, B, C, chunk_size,
            D=D, z=z, dt_bias=dt_bias, dt_softplus=True,
            return_final_states=False,
        )
        torch.cuda.synchronize()
    except Exception as e:
        pytest.fail(f"Reference function crashed: {e}")

    try:
        out_my = my_mamba_chunk_scan_combined(
            x, dt, A, B, C, chunk_size,
            D=D, z=z, dt_bias=dt_bias, dt_softplus=True,
            return_final_states=False,
        )
        torch.cuda.synchronize()
    except Exception as e:
        pytest.fail(f"My function crashed: {e}")

    assert torch.allclose(out_ref, out_my, rtol=1e-4, atol=1e-5), \
        f"Outputs differ! max diff = {(out_ref - out_my).abs().max().item()}"
