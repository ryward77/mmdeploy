class ConvModule(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  conv : __torch__.torch.nn.modules.conv.___torch_mangle_204.Conv2d
  bn : __torch__.torch.nn.modules.batchnorm.___torch_mangle_205.BatchNorm2d
  activate : __torch__.torch.nn.modules.activation.___torch_mangle_206.ReLU
  def forward(self: __torch__.mmcv.cnn.bricks.conv_module.___torch_mangle_207.ConvModule,
    input: Tensor) -> Tensor:
    conv = self.conv
    activate = self.activate
    bn = self.bn
    _0 = (activate).forward((bn).forward(input, ), )
    return (conv).forward(_0, )
