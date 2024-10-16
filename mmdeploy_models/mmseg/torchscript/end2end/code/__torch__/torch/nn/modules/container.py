class Sequential(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  __annotations__["0"] = __torch__.mmseg.models.utils.basic_block.BasicBlock
  __annotations__["1"] = __torch__.mmseg.models.utils.basic_block.___torch_mangle_18.BasicBlock
  def forward(self: __torch__.torch.nn.modules.container.Sequential,
    argument_1: Tensor) -> Tensor:
    _1 = getattr(self, "1")
    _0 = getattr(self, "0")
    _2 = (_1).forward((_0).forward(argument_1, ), )
    return _2
class ModuleList(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  __annotations__["0"] = __torch__.torch.nn.modules.container.___torch_mangle_64.Sequential
  __annotations__["1"] = __torch__.torch.nn.modules.container.___torch_mangle_85.Sequential
  __annotations__["2"] = __torch__.torch.nn.modules.container.___torch_mangle_100.Sequential
