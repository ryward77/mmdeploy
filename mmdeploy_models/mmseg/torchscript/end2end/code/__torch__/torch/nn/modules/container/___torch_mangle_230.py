class Sequential(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  __annotations__["0"] = __torch__.mmcv.cnn.bricks.conv_module.___torch_mangle_227.ConvModule
  __annotations__["1"] = __torch__.mmengine.model.utils.___torch_mangle_228._BatchNormXd
  __annotations__["2"] = __torch__.torch.nn.modules.activation.___torch_mangle_229.ReLU
  def forward(self: __torch__.torch.nn.modules.container.___torch_mangle_230.Sequential,
    argument_1: Tensor) -> Tensor:
    _2 = getattr(self, "2")
    _1 = getattr(self, "1")
    _0 = getattr(self, "0")
    _3 = (_1).forward((_0).forward(argument_1, ), )
    return (_2).forward(_3, )
