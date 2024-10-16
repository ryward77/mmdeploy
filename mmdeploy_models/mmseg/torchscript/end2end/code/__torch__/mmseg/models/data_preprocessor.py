class SegDataPreProcessor(Module):
  __parameters__ = []
  __buffers__ = ["mean", "std", ]
  mean : Tensor
  std : Tensor
  training : bool
  _is_full_backward_hook : Optional[bool]
