class DAPPM(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  scales : __torch__.mmengine.model.base_module.ModuleList
  processes : __torch__.mmengine.model.base_module.___torch_mangle_212.ModuleList
  compression : __torch__.mmcv.cnn.bricks.conv_module.___torch_mangle_216.ConvModule
  shortcut : __torch__.mmcv.cnn.bricks.conv_module.___torch_mangle_220.ConvModule
  def forward(self: __torch__.mmseg.models.utils.ppm.DAPPM,
    argument_1: Tensor) -> Tensor:
    shortcut = self.shortcut
    compression = self.compression
    processes = self.processes
    _3 = getattr(processes, "3")
    scales = self.scales
    _4 = getattr(scales, "4")
    processes0 = self.processes
    _2 = getattr(processes0, "2")
    scales0 = self.scales
    _30 = getattr(scales0, "3")
    processes1 = self.processes
    _1 = getattr(processes1, "1")
    scales1 = self.scales
    _20 = getattr(scales1, "2")
    processes2 = self.processes
    _0 = getattr(processes2, "0")
    scales2 = self.scales
    _10 = getattr(scales2, "1")
    scales3 = self.scales
    _00 = getattr(scales3, "0")
    _5 = (_00).forward(argument_1, )
    _6 = (_10).forward(argument_1, )
    _7 = ops.prim.NumToTensor(torch.size(argument_1, 2))
    _8 = int(_7)
    _9 = ops.prim.NumToTensor(torch.size(argument_1, 3))
    feat_up = torch.upsample_bilinear2d(_6, [_8, int(_9)], False, None)
    input = torch.add(feat_up, _5)
    _11 = (_0).forward(input, )
    _12 = (_20).forward(argument_1, )
    _13 = ops.prim.NumToTensor(torch.size(argument_1, 2))
    _14 = int(_13)
    _15 = ops.prim.NumToTensor(torch.size(argument_1, 3))
    feat_up0 = torch.upsample_bilinear2d(_12, [_14, int(_15)], False, None)
    input0 = torch.add(feat_up0, _11)
    _16 = (_1).forward(input0, )
    _17 = (_30).forward(argument_1, )
    _18 = ops.prim.NumToTensor(torch.size(argument_1, 2))
    _19 = int(_18)
    _21 = ops.prim.NumToTensor(torch.size(argument_1, 3))
    feat_up1 = torch.upsample_bilinear2d(_17, [_19, int(_21)], False, None)
    input1 = torch.add(feat_up1, _16)
    _22 = (_2).forward(input1, )
    _23 = (_4).forward(argument_1, )
    _24 = ops.prim.NumToTensor(torch.size(argument_1, 2))
    _25 = int(_24)
    _26 = ops.prim.NumToTensor(torch.size(argument_1, 3))
    feat_up2 = torch.upsample_bilinear2d(_23, [_25, int(_26)], False, None)
    input2 = torch.add(feat_up2, _22)
    _27 = [_5, _11, _16, _22, (_3).forward(input2, )]
    input3 = torch.cat(_27, 1)
    input4 = torch.add((compression).forward(input3, ), (shortcut).forward(argument_1, ))
    return input4
