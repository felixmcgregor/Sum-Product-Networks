import torch


class MPN(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):

        ctx.save_for_backward(input)
        max_child = torch.max(input)
        input[input < max_child] = 0

        return input

    @staticmethod
    def backward(ctx, grad_output):

        input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        max_child = torch.max(input)
        grad_input[input < max_child] = 0
        grad_input[input == max_child] = 1

        return grad_input


N, D_in, H, D_out = 10, 5, 5, 1

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

x = torch.tensor([1., 2., 3.], requires_grad=True)
# Create random Tensors for weights.
w1 = torch.tensor([1., 1., 1.], requires_grad=True)

learning_rate = 1e-6
for t in range(10):
    # To apply our Function, we use Function.apply method. We alias this as 'mpn'.
    mpn = MPN.apply

    # forward pass
    y_pred = mpn(x + w1)

    # Compute and print loss
    loss = y_pred[0]

    # Use autograd to compute the backward pass.
    loss.backward()

    # Update weights using gradient descent
    with torch.no_grad():
        x += x.grad
        w1 += w1.grad

        x.grad.zero_()
        w1.grad.zero_()
    print("x", x)
    print("w", w1)
    print()


'''
def backward(tensors, grad_tensors=None, retain_graph=None, create_graph=False, grad_variables=None):
    """Computes the sum of gradients of given tensors w.r.t. graph leaves.

    The graph is differentiated using the chain rule. If any of ``tensors``
    are non-scalar (i.e. their data has more than one element) and require
    gradient, the function additionally requires specifying ``grad_tensors``.
    It should be a sequence of matching length, that contains gradient of
    the differentiated function w.r.t. corresponding tensors (``None`` is an
    acceptable value for all tensors that don't need gradient tensors).

    This function accumulates gradients in the leaves - you might need to zero
    them before calling it.

    Arguments:
        tensors (sequence of Tensor): Tensors of which the derivative will be
            computed.
        grad_tensors (sequence of (Tensor or None)): Gradients w.r.t.
            each element of corresponding tensors. None values can be specified for
            scalar Tensors or ones that don't require grad. If a None value would
            be acceptable for all grad_tensors, then this argument is optional.
        retain_graph (bool, optional): If ``False``, the graph used to compute the grad
            will be freed. Note that in nearly all cases setting this option to ``True``
            is not needed and often can be worked around in a much more efficient
            way. Defaults to the value of ``create_graph``.
        create_graph (bool, optional): If ``True``, graph of the derivative will
            be constructed, allowing to compute higher order derivative products.
            Defaults to ``False``.
    """
    if grad_variables is not None:
        warnings.warn("'grad_variables' is deprecated. Use 'grad_tensors' instead.")
        if grad_tensors is None:
            grad_tensors = grad_variables
        else:
            raise RuntimeError("'grad_tensors' and 'grad_variables' (deprecated) "
                               "arguments both passed to backward(). Please only "
                               "use 'grad_tensors'.")

    tensors = (tensors,) if isinstance(tensors, torch.Tensor) else tuple(tensors)

    if grad_tensors is None:
        grad_tensors = [None] * len(tensors)
    elif isinstance(grad_tensors, torch.Tensor):
        grad_tensors = [grad_tensors]
    else:
        grad_tensors = list(grad_tensors)

    grad_tensors = _make_grads(tensors, grad_tensors)
    if retain_graph is None:
        retain_graph = create_graph

    Variable._execution_engine.run_backward(
        tensors, grad_tensors, retain_graph, create_graph,
        allow_unreachable=True)  # allow_unreachable flag
'''
