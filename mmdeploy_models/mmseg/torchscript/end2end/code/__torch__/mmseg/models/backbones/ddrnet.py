class DDRNet(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  stem : __torch__.torch.nn.modules.container.___torch_mangle_42.Sequential
  relu : __torch__.torch.nn.modules.activation.___torch_mangle_43.ReLU
  context_branch_layers : __torch__.torch.nn.modules.container.ModuleList
  compression_1 : __torch__.mmcv.cnn.bricks.conv_module.___torch_mangle_103.ConvModule
  down_1 : __torch__.mmcv.cnn.bricks.conv_module.___torch_mangle_106.ConvModule
  compression_2 : __torch__.mmcv.cnn.bricks.conv_module.___torch_mangle_109.ConvModule
  down_2 : __torch__.torch.nn.modules.container.___torch_mangle_117.Sequential
  spatial_branch_layers : __torch__.torch.nn.modules.container.___torch_mangle_170.ModuleList
  spp : __torch__.mmseg.models.utils.ppm.DAPPM
  def forward(self: __torch__.mmseg.models.backbones.ddrnet.DDRNet,
    inputs: Tensor) -> Tensor:
    spp = self.spp
    context_branch_layers = self.context_branch_layers
    _2 = getattr(context_branch_layers, "2")
    spatial_branch_layers = self.spatial_branch_layers
    _20 = getattr(spatial_branch_layers, "2")
    down_2 = self.down_2
    compression_2 = self.compression_2
    spatial_branch_layers0 = self.spatial_branch_layers
    _1 = getattr(spatial_branch_layers0, "1")
    context_branch_layers0 = self.context_branch_layers
    _10 = getattr(context_branch_layers0, "1")
    down_1 = self.down_1
    compression_1 = self.compression_1
    relu = self.relu
    spatial_branch_layers1 = self.spatial_branch_layers
    _0 = getattr(spatial_branch_layers1, "0")
    context_branch_layers1 = self.context_branch_layers
    _00 = getattr(context_branch_layers1, "0")
    stem = self.stem
    _3 = ops.prim.NumToTensor(torch.size(inputs, 2))
    _4 = torch.floor_divide(_3, CONSTANTS.c0)
    _5 = int(_4)
    _6 = int(_4)
    _7 = int(_4)
    _8 = ops.prim.NumToTensor(torch.size(inputs, 3))
    _9 = torch.floor_divide(_8, CONSTANTS.c0)
    _11 = int(_9)
    _12 = int(_9)
    _13 = int(_9)
    _14 = (stem).forward(inputs, )
    _15 = (_00).forward(_14, )
    _16 = (_0).forward(_14, )
    _17 = (compression_1).forward((relu).forward(_15, ), )
    _18 = (down_1).forward((relu).forward1(_16, ), )
    input = torch.add_(_15, _18)
    _19 = torch.upsample_bilinear2d(_17, [_7, _13], False, None)
    input0 = torch.add_(_16, _19)
    _21 = (_10).forward((relu).forward2(input, ), )
    _22 = (_1).forward((relu).forward3(input0, ), )
    _23 = (compression_2).forward((relu).forward4(_21, ), )
    _24 = (down_2).forward((relu).forward5(_22, ), )
    input1 = torch.add_(_21, _24)
    _25 = torch.upsample_bilinear2d(_23, [_6, _12], False, None)
    input2 = torch.add_(_22, _25)
    _26 = (_20).forward((relu).forward6(input2, ), )
    _27 = (_2).forward((relu).forward7(input1, ), )
    x_c = torch.upsample_bilinear2d((spp).forward(_27, ), [_5, _11], False, None)
    return torch.add(_26, x_c)
