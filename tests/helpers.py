# global
from perceiver_pytorch.perceiver_io import PerceiverIO


def get_perceiver_io(input_dim, num_input_axes, output_dim, num_output_axes, learn_query=False, num_queries=None):

    return PerceiverIO(
        input_dim=input_dim,  # dimension of sequence to be encoded
        num_input_axes=num_input_axes,  # number of input axes
        output_dim=output_dim,  # dimension of final logits
        queries_dim=1024,  # dimension of decoder queries
        network_depth=6,  # depth of net
        num_latents=512,
        # number of latents, or induced set points, or centroids. different papers giving it different names
        latent_dim=512,  # latent dimension
        num_cross_att_heads=1,  # number of heads for cross attention. paper said 1
        num_self_att_heads=8,  # number of heads for latent self attention, 8
        cross_head_dim=64,  # number of dimensions per cross attention head
        latent_head_dim=64,  # number of dimensions per latent self attention head
        weight_tie_layers=False,  # whether to weight tie layers (optional, as indicated in the diagram)
        learn_query=learn_query,  # whether or not to learn a fixed query vector internally
        query_shape=[num_queries]  # the number of queries to use when retreiving the output
    )
