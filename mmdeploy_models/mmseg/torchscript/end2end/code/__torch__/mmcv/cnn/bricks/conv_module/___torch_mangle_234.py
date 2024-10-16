class ConvModule(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  conv : __torch__.torch.nn.modules.conv.___torch_mangle_231.Conv2d
  bn : __torch__.mmengine.model.utils.___torch_mangle_232._BatchNormXd
  activate : __torch__.torch.nn.modules.activation.___torch_mangle_233.ReLU
