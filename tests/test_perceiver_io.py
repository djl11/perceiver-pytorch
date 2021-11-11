import torch
import pytest
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


@pytest.mark.parametrize(
    "batch_size", [1, 3])
@pytest.mark.parametrize(
    "img_dims", [[16, 16], [32, 32]])
@pytest.mark.parametrize(
    "learn_query", [True, False])
def test_perceiver_io_classification(batch_size, img_dims, learn_query):

    # params
    input_dim = 3
    num_input_axes = 2
    output_dim = 10
    num_output_axes = 1

    # inputs
    img = torch.randn([batch_size] + img_dims + [3])
    queries = None if learn_query else torch.randn(batch_size, 1, 1024)

    # model call
    model = get_perceiver_io(input_dim=input_dim,
                             num_input_axes=num_input_axes,
                             output_dim=output_dim,
                             num_output_axes=num_output_axes,
                             learn_query=learn_query,
                             num_queries=1)

    # output
    output = model(img, queries=queries)

    # cardinality test
    assert output.shape == (batch_size, 1, output_dim)


@pytest.mark.parametrize(
    "batch_size", [1, 3])
@pytest.mark.parametrize(
    "img_dims", [[16, 16], [32, 32]])
def test_perceiver_io_flow(batch_size, img_dims):

    # params
    input_dim = 3
    num_input_axes = 3
    output_dim = 2
    num_output_axes = 2

    # inputs
    img = torch.randn([batch_size, 2] + img_dims + [3])
    queries = torch.randn([batch_size] + img_dims + [1024])

    # model call
    model = get_perceiver_io(input_dim=input_dim,
                             num_input_axes=num_input_axes,
                             output_dim=output_dim,
                             num_output_axes=num_output_axes)

    # output
    output = model(img, queries=queries)

    # cardinality test
    assert output.shape == tuple([batch_size] + img_dims + [output_dim])
