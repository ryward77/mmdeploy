class ConvModule(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  conv : __torch__.torch.nn.modules.conv.___torch_mangle_11.Conv2d
  bn : __torch__.mmengine.model.utils.___torch_mangle_12._BatchNormXd
  activate : __torch__.torch.nn.modules.activation.___torch_mangle_13.ReLU
  def forward(self: __torch__.mmcv.cnn.bricks.conv_module.___torch_mangle_14.ConvModule,
    argument_1: Tensor) -> Tensor:
    activate = self.activate
    bn = self.bn
    conv = self.conv
    _0 = (bn).forward((conv).forward(argument_1, ), )
    return (activate).forward(_0, )
