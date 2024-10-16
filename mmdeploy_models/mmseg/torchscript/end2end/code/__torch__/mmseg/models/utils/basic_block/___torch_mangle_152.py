class BasicBlock(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  conv1 : __torch__.mmcv.cnn.bricks.conv_module.___torch_mangle_148.ConvModule
  conv2 : __torch__.mmcv.cnn.bricks.conv_module.___torch_mangle_151.ConvModule
  def forward(self: __torch__.mmseg.models.utils.basic_block.___torch_mangle_152.BasicBlock,
    argument_1: Tensor) -> Tensor:
    conv2 = self.conv2
    conv1 = self.conv1
    _0 = (conv2).forward((conv1).forward(argument_1, ), )
    return torch.add_(_0, argument_1)
