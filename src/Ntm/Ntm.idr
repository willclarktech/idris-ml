module Ntm.Ntm

import Network
import Tensor

import Ntm.Memory


public export
record Ntm where
  constructor MkNtm
  controller : Network i hs o ty
  memory : Matrix i o ty
  readHead : ReadHead
  writeHead : WriteHead
