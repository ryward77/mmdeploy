class EncoderDecoder(Module):
  __parameters__ = []
  __buffers__ = []
  training : bool
  _is_full_backward_hook : Optional[bool]
  data_preprocessor : __torch__.mmseg.models.data_preprocessor.SegDataPreProcessor
  backbone : __torch__.mmseg.models.backbones.ddrnet.DDRNet
  decode_head : __torch__.mmseg.models.decode_heads.ddr_head.DDRHead
  def forward(self: __torch__.mmseg.models.segmentors.encoder_decoder.EncoderDecoder,
    inputs: Tensor) -> Tensor:
    decode_head = self.decode_head
    conv_seg = decode_head.conv_seg
    decode_head0 = self.decode_head
    head = decode_head0.head
    backbone = self.backbone
    _0 = ops.prim.NumToTensor(torch.size(inputs, 2))
    _1 = int(_0)
    _2 = ops.prim.NumToTensor(torch.size(inputs, 3))
    _3 = int(_2)
    _4 = (head).forward((backbone).forward(inputs, ), )
    seg_logit = torch.upsample_bilinear2d((conv_seg).forward(_4, ), [_1, _3], False, None)
    return torch.argmax(seg_logit, 1, True)
