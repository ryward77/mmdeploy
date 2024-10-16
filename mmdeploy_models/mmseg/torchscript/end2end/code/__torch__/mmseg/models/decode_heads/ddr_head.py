class DDRHead(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  loss_decode : __torch__.torch.nn.modules.container.___torch_mangle_222.ModuleList
  conv_seg : __torch__.torch.nn.modules.conv.___torch_mangle_223.Conv2d
  head : __torch__.torch.nn.modules.container.___torch_mangle_230.Sequential
  aux_head : __torch__.torch.nn.modules.container.___torch_mangle_237.Sequential
  aux_cls_seg : __torch__.torch.nn.modules.conv.___torch_mangle_238.Conv2d
