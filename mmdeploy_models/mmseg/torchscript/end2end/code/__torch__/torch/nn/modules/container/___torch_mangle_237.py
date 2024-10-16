class Sequential(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  __annotations__["0"] = __torch__.mmcv.cnn.bricks.conv_module.___torch_mangle_234.ConvModule
  __annotations__["1"] = __torch__.mmengine.model.utils.___torch_mangle_235._BatchNormXd
  __annotations__["2"] = __torch__.torch.nn.modules.activation.___torch_mangle_236.ReLU
