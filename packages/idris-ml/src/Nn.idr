||| The v1 models-as-records surface. `import Nn` brings the whole layer
||| library (Module/Params/Seq/Init/Group/Recurrent + all ~19 ported
||| layers). Coexists with the legacy `Layer/` until the example sweep; see
||| design-decisions.md "models-as-records: the `Nn` surface".
module Nn

import public Nn.Module
import public Nn.Seq
import public Nn.Init
import public Nn.Group
import public Nn.Recurrent

import public Nn.Linear
import public Nn.LinearMixed
import public Nn.Activation
import public Nn.LayerNorm
import public Nn.Dropout
import public Nn.Residual
import public Nn.Embedding
import public Nn.RmsNorm
import public Nn.LoraLinear
import public Nn.SwiGLU
import public Nn.BatchNorm
import public Nn.Conv
import public Nn.BitLinear
import public Nn.Attention
import public Nn.Transformer

import public Nn.Lstm
import public Nn.Gru
import public Nn.Ntm
import public Nn.Dnc
