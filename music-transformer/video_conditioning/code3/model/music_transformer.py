import json
from typing import Optional

from tokenizers import Tokenizer
from torch import Tensor, cat, nn, zeros
from torch.nn.modules.normalization import LayerNorm

from model.positional_encoding import PositionalEncoding
from model.rpr import TransformerEncoderLayerRPR, TransformerEncoderRPR
from utilities.constants import TOKEN_PAD, VOCAB_SIZE
from utilities.device import get_device


class MusicTransformer(nn.Module):
    """The implementation of Music Transformer.

    Music Transformer reproduction from https://arxiv.org/abs/1809.04281.
    Arguments allow for tweaking the transformer architecture
    (https://arxiv.org/abs/1706.03762) and the rpr argument toggles
    Relative Position Representations (RPR - https://arxiv.org/abs/1803.02155).

    Supports training and generation using Pytorch's nn.Transformer class
    with DummyDecoder to make a decoder-only transformer architecture.

    For RPR support, there is modified Pytorch 1.2.0 code in model/rpr.py.
    """

    def __init__(  # noqa: WPS210
        self,
        n_layers: int = 6,
        num_heads: int = 8,
        d_model: int = 512,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_sequence: int = 2048,
        features_embedding_dim: int = 36,
        rpr: bool = False,
    ) -> None:
        """Inits MusicTransformer.

        Default parameters are taken from section 4.2 of the original article:
        https://arxiv.org/abs/1809.04281

        Args:
            n_layers (int): A number of layers in the encoder.
            num_heads (int): A number of heads used in Multi-Head attention.
            d_model (int): A token embedding size.
            dim_feedforward (int): A dimension of the feedforward network model
                used in nn.Transformer.
            dropout (float): A dropout value in Positional Encoding and in
                encoder layers.
            max_sequence (int): A maximum length of a tokenized composition.
            rpr (bool): A boolean value indicating whether to use Relative
                Positional Encoding or not.
        """
        super().__init__()

        self.dummy = DummyDecoder()
        self.nlayers = n_layers
        self.nhead = num_heads
        self.d_model = d_model
        self.d_ff = dim_feedforward
        self.dropout = dropout
        self.max_seq = max_sequence

        # Features info
        # with open('/data/transformer-production/code3/dataset/additional_features_state.json') as fp:
        with open('dataset/additional_features_state.json') as fp:
            state = json.load(fp)
            self.local_features_indices = state['local_features_indices']
            self.global_features_indices = state['global_features_indices']
            self.features_vocab_sizes = state['vocab_sizes']
            tokenizers = {
                tokenizer_name: Tokenizer.from_file(tokenizer_path)
                for tokenizer_name, tokenizer_path in state['tokenizers'].items()
            }

        # Features Embeddings
        self.features_embedding_dim = features_embedding_dim
        self.features_embeddings = nn.ModuleDict(self.init_embeddings(tokenizers))
        del tokenizers  # noqa: WPS420

        if self.features_embeddings:
            self.tokens_embedding_dim = self.d_model - self.features_embedding_dim
        else:
            print('No additional features')
            self.tokens_embedding_dim = self.d_model

        # Input embedding
        self.embedding = nn.Embedding(
            VOCAB_SIZE,
            self.tokens_embedding_dim,
            padding_idx=TOKEN_PAD,
        )

        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            self.d_model,
            dropout=self.dropout,
            max_len=self.max_seq,
        )

        # Define encoder as None for Base Transformer
        encoder = None

        # else define encoder as TransformerEncoderRPR for RPR Transformer
        if rpr:
            encoder_norm = LayerNorm(self.d_model)
            encoder_layer = TransformerEncoderLayerRPR(
                self.d_model,
                self.nhead,
                dim_feedforward=self.d_ff,
                p_dropout=self.dropout,
                er_len=self.max_seq,
            )
            encoder = TransformerEncoderRPR(
                encoder_layer,
                self.nlayers,
                norm=encoder_norm,
            )

        # To make a decoder-only transformer we need to use masked encoder
        # layers and DummyDecoder to essentially just return the encoder output
        self.transformer = nn.Transformer(
            d_model=self.d_model,
            nhead=self.nhead,
            num_encoder_layers=self.nlayers,
            num_decoder_layers=0,
            dropout=self.dropout,
            dim_feedforward=self.d_ff,
            custom_decoder=self.dummy, # ???
            custom_encoder=encoder,
        )

        self.Wout = nn.Linear(self.d_model, VOCAB_SIZE)

    def init_embeddings(self, tokenizers):  # noqa: WPS210
        """Creates embedding layers for each feature.

        Returns:
            dict of embedding layer for each feature
        """
        features_embeddings = {}
        for local_feature in self.local_features_indices.keys():
            vocab_size = self.features_vocab_sizes[local_feature]
            embedding = nn.Embedding(
                vocab_size,
                self.features_embedding_dim,
                padding_idx=vocab_size - 1,
            )
            features_embeddings[local_feature] = embedding

        for global_feature, tokenizer in tokenizers.items():
            vocab_size = tokenizer.get_vocab_size()
            embedding = nn.Embedding(
                vocab_size,
                self.features_embedding_dim,
            )
            features_embeddings[global_feature] = embedding
        return features_embeddings

    def forward(  # noqa: WPS210
        self,
        x: Tensor,
        local_features: Optional[Tensor],
        global_features: Optional[Tensor],
        use_mask: bool = True,
    ) -> Tensor:
        """Takes an input sequence and outputs predictions via seq2seq method.

        A prediction at one index is the "next" prediction given all information
        seen previously.

        Args:
            x (Tensor): A tensor of tokenized input compositions of
                dimension (batch_size, self.max_seq).
            local_features (Optional[Tensor]): An optional tensor of
                local features
            global_features (Optional[Tensor]): An optional tensor of
                local features
            use_mask (bool): A boolean indicating whether to use a mask for the
                encoder or not.

        Returns:
            A tensor of dimension (batch_size, self.max_seq, VOCAB_SIZE), where
            in the position [i, j, :] are the logits of the distribution of the
            `j+1`-th token of the `i`-th composition from the batch.
        """
        if use_mask:
            mask = self.transformer.generate_square_subsequent_mask(
                x.shape[1],
            ).to(get_device())
        else:
            mask = None

        x = self.embedding(x)
        if self.features_embeddings:
            additional_features = zeros((x.shape[0], x.shape[1], self.features_embedding_dim)).to(get_device())

            for feature, indices in self.local_features_indices.items():
                begin, end = indices
                embedding = self.features_embeddings[feature]
                feature_values = local_features[..., begin:end].squeeze()
                encoded = embedding(feature_values)
                additional_features += encoded

            for feature, indices in self.global_features_indices.items():  # noqa: WPS440
                begin, end = indices
                embedding = self.features_embeddings[feature]
                feature_values = global_features[..., begin:end]
                encoded = embedding(feature_values)
                additional_features += encoded

            x = cat((x, additional_features), dim=-1)

        # Input shape is (max_seq, batch_size, d_model)
        x = x.permute(1, 0, 2)
        x = self.positional_encoding(x)

        # Since there are no true decoder layers, the tgt is unused
        # Pytorch wants src and tgt to have some equal dims however
        x_out = self.transformer(src=x, tgt=x, src_mask=mask)

        # Back to (batch_size, max_seq, d_model)
        x_out = x_out.permute(1, 0, 2)

        y = self.Wout(x_out)

        del mask  # noqa: WPS420

        # They are trained to predict the next note in sequence
        # we don't need the last one
        return y


class DummyDecoder(nn.Module):
    """A dummy decoder that returns its input.

    Used to make the Pytorch transformer into a decoder-only architecture
    (stacked encoders with dummy decoder fits the bill).
    """

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Returns the input (memory).

        Args:
            tgt (Tensor): A sequence to the decoder.
            memory (Tensor): A sequence from the last layer of the encoder.
            tgt_mask (Optional[Tensor]): A mask for the tgt sequence.
            memory_mask (Optional[Tensor]): A mask for the memory sequence.
            tgt_key_padding_mask (Optional[Tensor]): A mask for the tgt keys per
                batch.
            memory_key_padding_mask (Optional[Tensor]): A mask for the memory
                keys per batch.

        Returns:
            The `memory` tensor from input.
        """
        return memory
