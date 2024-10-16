class _BatchNormXd(Module):
  __parameters__ = ["weight", "bias", ]
  __buffers__ = ["running_mean", "running_var", "num_batches_tracked", ]
  weight : Tensor
  bias : Tensor
  running_mean : Tensor
  running_var : Tensor
  num_batches_tracked : Tensor
  training : bool
  _is_full_backward_hook : Optional[bool]
  def forward(self: __torch__.mmengine.model.utils.___torch_mangle_115._BatchNormXd,
    argument_1: Tensor) -> Tensor:
    running_var = self.running_var
    running_mean = self.running_mean
    bias = self.bias
    weight = self.weight
    _0 = torch.batch_norm(argument_1, weight, bias, running_mean, running_var, False, 0.10000000000000001, 1.0000000000000001e-05, True)
    return _0
