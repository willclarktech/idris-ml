"""Neural-network primitives — activations, losses, embeddings, attention, cell ops."""

from .._entry import Entry


ENTRIES = {
    "tensor_batch_norm": Entry(args=('T', 'T', 'T', 'T', 'T', 'i', 'i', 'i', 'd', 'd'), ret='T', slice='UserExecutorNN', idris_method='primBatchNorm'),
    "tensor_bce_with_logits": Entry(args=('T', 'T'), ret='T', slice='UserExecutorNN', idris_method='primBceWithLogits'),
    "tensor_cosine_similarity": Entry(args=('T', 'T', 'i'), ret='T', slice='UserExecutorNN', idris_method='primCosineSimilarity'),
    "tensor_dropout": Entry(args=('T', 'd', 'i', 'i'), ret='T', slice='UserExecutorNN', idris_method='primDropout'),
    "tensor_embedding_2d": Entry(args=('T', 'T', 'i', 'i'), ret='T', slice='UserExecutorNN', idris_method='primEmbedding2d'),
    "tensor_embedding": Entry(args=('T', 'T', 'i', 'i'), ret='T', slice='UserExecutorNN', idris_method='primEmbedding'),
    "tensor_expand_mask": Entry(args=('T', 'i'), ret='T', slice='UserExecutorNN', idris_method='primExpandMask'),
    "tensor_gelu": Entry(args=('T',), ret='T', slice='UserExecutorNN', idris_method='primGelu'),
    "tensor_gru_cell": Entry(args=('T', 'T', 'T', 'i'), ret='T', slice='UserExecutorNN', idris_method='primGruCell'),
    "tensor_layer_norm_2d": Entry(args=('T', 'T', 'T', 'd'), ret='T', slice='UserExecutorNN', idris_method='primLayerNorm2d'),
    "tensor_leaky_relu": Entry(args=('T', 'd'), ret='T', slice='UserExecutorNN', idris_method='primLeakyRelu'),
    "tensor_log_softmax_2d": Entry(args=('T',), ret='T', slice='UserExecutorNN', idris_method='primLogSoftmax2d'),
    "tensor_log_softmax": Entry(args=('T', 'i'), ret='T', slice='UserExecutorNN', idris_method='primLogSoftmax'),
    "tensor_lstm_gates_pair": Entry(args=('T', 'T', 'i'), ret='R', slice='UserExecutorNN', idris_method='primLstmGatesPair'),
    "tensor_masked_fill": Entry(args=('T', 'T', 'd'), ret='T', slice='UserExecutorNN', idris_method='primMaskedFill'),
    "tensor_pair_first": Entry(args=('R',), ret='T', slice='UserExecutorNN', idris_method='primPairFirst'),
    "tensor_pair_second": Entry(args=('R',), ret='T', slice='UserExecutorNN', idris_method='primPairSecond'),
    "tensor_silu": Entry(args=('T',), ret='T', slice='UserExecutorNN', idris_method='primSilu'),
    "tensor_softmax_2d": Entry(args=('T',), ret='T', slice='UserExecutorNN', idris_method='primSoftmax2d'),
    "tensor_softmax_3d": Entry(args=('T',), ret='T', slice='UserExecutorNN', idris_method='primSoftmax3d'),
    "tensor_softmax": Entry(args=('T', 'i'), ret='T', slice='UserExecutorNN', idris_method='primSoftmax'),
    "tensor_softplus": Entry(args=('T',), ret='T', slice='UserExecutorNN', idris_method='primSoftplus'),
}
