class AvgPool2d(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  def forward(self: __torch__.torch.nn.modules.pooling.___torch_mangle_179.AvgPool2d,
    argument_1: Tensor) -> Tensor:
    input = torch.avg_pool2d(argument_1, [9, 9], [4, 4], [4, 4])
    return input
