# global
import torch
import pytest

# local
from tests.helpers import get_perceiver_io


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
