# global
import ivy
import time
import torch
import pytest
ivy.set_framework('torch')

# local
from tests.helpers import get_perceiver_io


def test_perceiver_io_classification_training(compile_graph):

    # params
    input_dim = 3
    num_input_axes = 2
    output_dim = 10
    num_output_axes = 1
    batch_size = 1
    img_dims = [32, 32]
    learn_query = True

    # inputs
    img = torch.randn([batch_size] + img_dims + [3]).cuda()
    queries = None if learn_query else torch.randn(batch_size, 1, 1024).cuda()

    # model call
    model = get_perceiver_io(input_dim=input_dim,
                             num_input_axes=num_input_axes,
                             output_dim=output_dim,
                             num_output_axes=num_output_axes,
                             learn_query=learn_query,
                             num_queries=1).cuda()

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # train
    start_time = time.perf_counter()
    print('training started!')
    for _ in range(1000):
        optimizer.zero_grad()
        output = model(img * torch.randn(size=img.shape).cuda(), queries=queries)
        loss = torch.mean(output)
        loss.backward()
        optimizer.step()
    print('training took {} seconds'.format(time.perf_counter() - start_time))


@pytest.mark.parametrize(
    "compile_graph", [True])
def test_perceiver_io_classification_ivy_training(compile_graph):

    # params
    input_dim = 3
    num_input_axes = 2
    output_dim = 10
    num_output_axes = 1
    batch_size = 1
    img_dims = [32, 32]
    learn_query = True

    # inputs
    img = torch.randn([batch_size] + img_dims + [3]).cuda()
    queries = None if learn_query else torch.randn(batch_size, 1, 1024).cuda()

    # model call
    model = ivy.to_ivy_module(
        get_perceiver_io(input_dim=input_dim,
                         num_input_axes=num_input_axes,
                         output_dim=output_dim,
                         num_output_axes=num_output_axes,
                         learn_query=learn_query,
                         num_queries=1).cuda())
    if compile_graph:
        model.compile_graph(img, queries=queries, v=model.v)

    # optimizer
    optimizer = ivy.SGD(lr=0.001, compile_on_next_step=compile_graph)

    # loss
    def loss_fn(v):
        output = model(img * torch.randn(size=img.shape).cuda(), queries=queries, v=v)
        return ivy.reduce_mean(output)

    # train
    start_time = time.perf_counter()
    print('training started!')
    for _ in range(1000):
        loss, grads = ivy.execute_with_gradients(loss_fn, model.v)
        model.v = optimizer.step(model.v, grads)
    print('training took {} seconds'.format(time.perf_counter() - start_time))
