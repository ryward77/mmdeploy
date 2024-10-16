class BasicBlock(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  conv1 : __torch__.mmcv.cnn.bricks.conv_module.___torch_mangle_6.ConvModule
  conv2 : __torch__.mmcv.cnn.bricks.conv_module.___torch_mangle_9.ConvModule
  act : __torch__.torch.nn.modules.activation.___torch_mangle_10.ReLU
  def forward(self: __torch__.mmseg.models.utils.basic_block.BasicBlock,
    argument_1: Tensor) -> Tensor:
    act = self.act
    conv2 = self.conv2
    conv1 = self.conv1
    _0 = (conv2).forward((conv1).forward(argument_1, ), )
    input = torch.add_(_0, argument_1)
    return (act).forward(input, )
class Bottleneck(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  conv1 : __torch__.mmcv.cnn.bricks.conv_module.___torch_mangle_89.ConvModule
  conv2 : __torch__.mmcv.cnn.bricks.conv_module.___torch_mangle_93.ConvModule
  conv3 : __torch__.mmcv.cnn.bricks.conv_module.___torch_mangle_96.ConvModule
  downsample : __torch__.torch.nn.modules.container.___torch_mangle_99.Sequential
  def forward(self: __torch__.mmseg.models.utils.basic_block.Bottleneck,
    argument_1: Tensor) -> Tensor:
    downsample = self.downsample
    conv3 = self.conv3
    conv2 = self.conv2
    conv1 = self.conv1
    _1 = (conv2).forward((conv1).forward(argument_1, ), )
    input = torch.add_((conv3).forward(_1, ), (downsample).forward(argument_1, ))
    return input
