||| The models-as-records surface. `import Nn` brings the whole layer library
||| (Module/Params/Seq/Init/Group/Recurrent + all ~19 ported layers). Models are
||| single-owner linear resources threaded through `L IO`; see design-decisions.md
||| "models-as-records: the `Nn` surface" + docs/develop/linear-types-and-effects.md.
module Ml.Nn

import public Ml.Nn.Activation
import public Ml.Nn.Attention
import public Ml.Nn.BatchNorm
import public Ml.Nn.BitLinear
import public Ml.Nn.Conv
import public Ml.Nn.Dnc
import public Ml.Nn.Dropout
import public Ml.Nn.Embedding
import public Ml.Nn.Group
import public Ml.Nn.Gru
import public Ml.Nn.Init
import public Ml.Nn.LayerNorm
import public Ml.Nn.Linear
import public Ml.Nn.LinearMixed
import public Ml.Nn.LoraLinear
import public Ml.Nn.Lstm
import public Ml.Nn.Module
import public Ml.Nn.Ntm
import public Ml.Nn.Pool
import public Ml.Nn.PosEncoding
import public Ml.Nn.Recurrent
import public Ml.Nn.Residual
import public Ml.Nn.RmsNorm
import public Ml.Nn.Seq
import public Ml.Nn.SwiGLU
import public Ml.Nn.Transformer
