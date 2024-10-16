class AvgPool2d(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  def forward(self: __torch__.torch.nn.modules.pooling.AvgPool2d,
    argument_1: Tensor) -> Tensor:
    input = torch.avg_pool2d(argument_1, [5, 5], [2, 2], [2, 2])
    return input
class AdaptiveAvgPool2d(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  def forward(self: __torch__.torch.nn.modules.pooling.AdaptiveAvgPool2d,
    argument_1: Tensor) -> Tensor:
    input = torch.adaptive_avg_pool2d(argument_1, [1, 1])
    return input
