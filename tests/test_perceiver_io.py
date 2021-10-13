import torch
from perceiver_pytorch.perceiver_io import PerceiverIO

model = PerceiverIO(
    input_dim=32,  # dimension of sequence to be encoded
    num_input_axes=1,
    output_dim=100,  # dimension of final logits
    num_output_axes=None,
    queries_dim=32,  # dimension of decoder queries
    network_depth=6,  # depth of net
    num_latents=256,
    # number of latents, or induced set points, or centroids. different papers giving it different names
    latent_dim=512,  # latent dimension
    num_cross_att_heads=1,  # number of heads for cross attention. paper said 1
    num_self_att_heads=8,  # number of heads for latent self attention, 8
    cross_head_dim=64,  # number of dimensions per cross attention head
    latent_head_dim=64,  # number of dimensions per latent self attention head
    weight_tie_layers=False  # whether to weight tie layers (optional, as indicated in the diagram)
)


def test_perceiver_io_classification():

    seq = torch.randn(1, 512, 32)
    queries = torch.randn(128, 32)

    logits = model(seq, queries=queries)  # (1, 128, 100) - (batch, decoder seq, logits dim)
    assert logits.shape == (1, 128, 100)
