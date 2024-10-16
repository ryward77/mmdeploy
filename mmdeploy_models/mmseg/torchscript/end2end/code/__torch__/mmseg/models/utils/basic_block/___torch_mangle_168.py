class Bottleneck(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  conv1 : __torch__.mmcv.cnn.bricks.conv_module.___torch_mangle_157.ConvModule
  conv2 : __torch__.mmcv.cnn.bricks.conv_module.___torch_mangle_161.ConvModule
  conv3 : __torch__.mmcv.cnn.bricks.conv_module.___torch_mangle_164.ConvModule
  downsample : __torch__.torch.nn.modules.container.___torch_mangle_167.Sequential
  def forward(self: __torch__.mmseg.models.utils.basic_block.___torch_mangle_168.Bottleneck,
    argument_1: Tensor) -> Tensor:
    downsample = self.downsample
    conv3 = self.conv3
    conv2 = self.conv2
    conv1 = self.conv1
    _0 = (conv2).forward((conv1).forward(argument_1, ), )
    x_s = torch.add_((conv3).forward(_0, ), (downsample).forward(argument_1, ))
    return x_s
