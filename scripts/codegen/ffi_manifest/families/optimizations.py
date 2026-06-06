"""Opt-in fused / specialised primitives (SDPA, RMSNorm, SwiGLU, fused initialisers, etc.)."""

from .._entry import Entry


ENTRIES = {
    "polyak_blend": Entry(args=('d', 's', 's'), ret='i', slice='UserExecutorOptimizations', idris_method='primPolyakBlend', mlx='direct'),
    "tensor_create_param_1d_const_streamed": Entry(args=('i', 'd', 'i', 'i'), ret='T', slice='UserExecutorOptimizations', idris_method='primCreateParam1dConstStreamed', mlx='direct'),
    "tensor_create_param_1d_normal_streamed": Entry(args=('i', 'd', 'd', 'i', 'i'), ret='T', slice='UserExecutorOptimizations', idris_method='primCreateParam1dNormalStreamed', mlx='direct'),
    "tensor_create_param_2d_const_streamed": Entry(args=('i', 'i', 'd', 'i', 'i'), ret='T', slice='UserExecutorOptimizations', idris_method='primCreateParam2dConstStreamed', mlx='direct'),
    "tensor_create_param_2d_normal_streamed": Entry(args=('i', 'i', 'd', 'd', 'i', 'i'), ret='T', slice='UserExecutorOptimizations', idris_method='primCreateParam2dNormalStreamed', mlx='direct'),
    "tensor_create_param_3d_const_streamed": Entry(args=('i', 'i', 'i', 'd', 'i', 'i'), ret='T', slice='UserExecutorOptimizations', idris_method='primCreateParam3dConstStreamed', mlx='direct'),
    "tensor_create_param_3d_normal_streamed": Entry(args=('i', 'i', 'i', 'd', 'd', 'i', 'i'), ret='T', slice='UserExecutorOptimizations', idris_method='primCreateParam3dNormalStreamed', mlx='direct'),
    "tensor_create_param_4d_const_streamed": Entry(args=('i', 'i', 'i', 'i', 'd', 'i', 'i'), ret='T', slice='UserExecutorOptimizations', idris_method='primCreateParam4dConstStreamed', mlx='direct'),
    "tensor_create_param_4d_normal_streamed": Entry(args=('i', 'i', 'i', 'i', 'd', 'd', 'i', 'i'), ret='T', slice='UserExecutorOptimizations', idris_method='primCreateParam4dNormalStreamed', mlx='direct'),
    "tensor_cross_attention": Entry(args=('T', 'T', 'T', 'T', 'd'), ret='T', slice='UserExecutorOptimizations', idris_method='primCrossAttention'),
    "tensor_rms_norm_2d": Entry(args=('T', 'T', 'd'), ret='T', slice='UserExecutorOptimizations', idris_method='primRmsNorm2d', mlx='direct'),
    "tensor_sdpa_2d": Entry(args=('T', 'T', 'T', 'i', 'i', 'i', 'i'), ret='T', slice='UserExecutorOptimizations', idris_method='primSdpa2d', mlx='direct'),
    "tensor_swi_glu_2d": Entry(args=('T', 'T'), ret='T', slice='UserExecutorOptimizations', idris_method='primSwiGlu2d', c_symbol='tensor_swiglu_2d', mlx='direct'),
    "tensor_tile_2d": Entry(args=('T', 'i', 'i'), ret='T', slice='UserExecutorOptimizations', idris_method='primTile2d'),
}
