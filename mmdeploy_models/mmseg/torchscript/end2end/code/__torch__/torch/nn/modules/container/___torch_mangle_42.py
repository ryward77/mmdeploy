class Sequential(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  __annotations__["0"] = __torch__.mmcv.cnn.bricks.conv_module.ConvModule
  __annotations__["1"] = __torch__.mmcv.cnn.bricks.conv_module.___torch_mangle_3.ConvModule
  __annotations__["2"] = __torch__.torch.nn.modules.container.Sequential
  __annotations__["3"] = __torch__.torch.nn.modules.activation.___torch_mangle_19.ReLU
  __annotations__["4"] = __torch__.torch.nn.modules.container.___torch_mangle_40.Sequential
  __annotations__["5"] = __torch__.torch.nn.modules.activation.___torch_mangle_41.ReLU
  def forward(self: __torch__.torch.nn.modules.container.___torch_mangle_42.Sequential,
    inputs: Tensor) -> Tensor:
    _5 = getattr(self, "5")
    _4 = getattr(self, "4")
    _3 = getattr(self, "3")
    _2 = getattr(self, "2")
    _1 = getattr(self, "1")
    _0 = getattr(self, "0")
    _6 = (_1).forward((_0).forward(inputs, ), )
    _7 = (_4).forward((_3).forward((_2).forward(_6, ), ), )
    return (_5).forward(_7, )
