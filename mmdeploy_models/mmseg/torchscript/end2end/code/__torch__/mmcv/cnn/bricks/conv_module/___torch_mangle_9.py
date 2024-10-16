class ConvModule(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  conv : __torch__.torch.nn.modules.conv.___torch_mangle_7.Conv2d
  bn : __torch__.torch.nn.modules.batchnorm.___torch_mangle_8.BatchNorm2d
  def forward(self: __torch__.mmcv.cnn.bricks.conv_module.___torch_mangle_9.ConvModule,
    argument_1: Tensor) -> Tensor:
    bn = self.bn
    conv = self.conv
    _0 = (bn).forward((conv).forward(argument_1, ), )
    return _0
