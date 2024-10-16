class ModuleList(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  __annotations__["0"] = __torch__.mmseg.models.losses.tversky_loss.TverskyLoss
  __annotations__["1"] = __torch__.mmseg.models.losses.tversky_loss.___torch_mangle_221.TverskyLoss
