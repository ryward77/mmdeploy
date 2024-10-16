class Conv2d(Module):
  __parameters__ = ["weight", ]
  __buffers__ = []
  weight : Tensor
  training : bool
  _is_full_backward_hook : Optional[bool]
