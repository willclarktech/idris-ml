||| The models-as-records surface. `import Nn` brings the whole layer library
||| (Module/Params/Seq/Init/Group/Recurrent + all ~19 ported layers). Models are
||| single-owner linear resources threaded through `L IO`; see design-decisions.md
||| "models-as-records: the `Nn` surface" + docs/develop/linear-types-and-effects.md.
module Nn

import public Nn.Activation
import public Nn.Attention
import public Nn.BatchNorm
import public Nn.BitLinear
import public Nn.Conv
import public Nn.Dnc
import public Nn.Dropout
import public Nn.Embedding
import public Nn.Group
import public Nn.Gru
import public Nn.Init
import public Nn.LayerNorm
import public Nn.Linear
import public Nn.LinearMixed
import public Nn.LoraLinear
import public Nn.Lstm
import public Nn.Module
import public Nn.Ntm
import public Nn.Pool
import public Nn.PosEncoding
import public Nn.Recurrent
import public Nn.Residual
import public Nn.RmsNorm
import public Nn.Seq
import public Nn.SwiGLU
import public Nn.Transformer
