class ConvModule(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  conv : __torch__.torch.nn.modules.conv.___torch_mangle_44.Conv2d
  bn : __torch__.torch.nn.modules.batchnorm.___torch_mangle_45.BatchNorm2d
  activate : __torch__.torch.nn.modules.activation.___torch_mangle_46.ReLU
  def forward(self: __torch__.mmcv.cnn.bricks.conv_module.___torch_mangle_47.ConvModule,
    argument_1: Tensor) -> Tensor:
    activate = self.activate
    bn = self.bn
    conv = self.conv
    _0 = (bn).forward((conv).forward(argument_1, ), )
    return (activate).forward(_0, )
