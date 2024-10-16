class ConvModule(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  conv : __torch__.torch.nn.modules.conv.Conv2d
  bn : __torch__.mmengine.model.utils._BatchNormXd
  activate : __torch__.torch.nn.modules.activation.ReLU
  def forward(self: __torch__.mmcv.cnn.bricks.conv_module.ConvModule,
    inputs: Tensor) -> Tensor:
    activate = self.activate
    bn = self.bn
    conv = self.conv
    _0 = (bn).forward((conv).forward(inputs, ), )
    return (activate).forward(_0, )
