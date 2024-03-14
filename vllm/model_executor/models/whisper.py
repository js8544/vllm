from torch import nn
import torch
from typing import List, Tuple, Optional
from transformers import WhisperConfig
from transformers.activations import GELUActivation
from vllm.model_executor.input_metadata import InputMetadata
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.enc_dec_attention import EncoderAttention, DecoderAttention
from vllm.model_executor.layers.sampler import Sampler
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.model_executor.weight_utils import hf_model_weights_iterator, default_weight_loader

KVCache = Tuple[torch.Tensor, torch.Tensor]


class WhisperAttention(nn.Module):
    def __init__(
        self,
        config: WhisperConfig,
        num_heads: int,
        is_decoder: bool,
        bias: bool = True,
        is_cross: bool = False,
    ):
        super().__init__()
        self.d_model = config.d_model
        self.num_heads = num_heads
        self.key_value_proj_dim = self.d_model
        self.head_dim = self.d_model // self.num_heads
        self.is_decoder = is_decoder
        if (self.head_dim * num_heads) != self.d_model:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.d_model}"
                f" and `num_heads`: {num_heads})."
            )

        self.scaling = self.head_dim**-0.5
        self.k_proj = ColumnParallelLinear(self.d_model, self.d_model, bias=False)
        self.v_proj = ColumnParallelLinear(self.d_model, self.d_model, bias=bias)
        self.q_proj = ColumnParallelLinear(self.d_model, self.d_model, bias=bias)
        self.out_proj = RowParallelLinear(self.d_model, self.d_model, bias=bias)

        if self.is_decoder or is_cross:
            raise NotImplementedError("Decoder attention not implemented yet.")
            """
            self.attn = DecoderAttention(self.num_heads, self.head_dim, 1)
            """
        else:
            # Encoder attention
            self.attn = EncoderAttention(self.num_heads, self.head_dim, 1)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Optional[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:

        bsz, seq_len, _ = hidden_states.size()
        q, _ = self.q_proj(hidden_states)
        q = q * self.scaling # could be potentially done elsewhere

        if not self.is_decoder:
            # Encoding step. This means that the transformer blocks
            # only employ self-attention and there is no KV cache
            # available to be used
            if kv_cache[0][0] is not None:
                 raise ValueError("Encoder self-attention step. The KV cache should not be populated.")
            k, _ = self.k_proj(hidden_states)
            v, _ = self.v_proj(hidden_states)

            # reshape the tensors to the shape required by the EncoderAttention
            proj_shape = (bsz, -1, self.head_dim * self.num_heads)
            q = q.reshape(*proj_shape)
            k = k.reshape(*proj_shape)
            v = v.reshape(*proj_shape)
            attn_output = self.attn(q, k, v, input_metadata)
            o, _ = self.out_proj(attn_output)

        return o


class WhisperEncoderBlock(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.d_model = config.d_model

        self.self_attn = WhisperAttention(
            config=config,
            num_heads=config.encoder_attention_heads,
            is_decoder=False,
            is_cross=False,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.d_model)
        self.activation_fn = GELUActivation()
        self.fc1 = nn.Linear(self.d_model, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.d_model)
        self.final_layer_norm = nn.LayerNorm(self.d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        kv_cache: Optional[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:

        residual = hidden_states
        
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, kv_cache, input_metadata)
        hidden_states = residual + hidden_states
        
        residual = hidden_states

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        
        hidden_states = residual + hidden_states

        return hidden_states


class WhisperEncoder(nn.Module):
    def __init__(self, config: WhisperConfig):
        super().__init__()
        self.d_model = config.d_model
        self.num_mel_bins = config.num_mel_bins
        self.max_source_positions = config.max_source_positions
        
        self.conv1 = nn.Conv1d(
            self.num_mel_bins, self.d_model, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv1d(
            self.d_model, self.d_model, kernel_size=3, stride=2, padding=1
        )

        self.embed_positions = nn.Embedding(self.max_source_positions, self.d_model)
        self.layers = nn.ModuleList(
            [WhisperEncoderBlock(config) for i in range(config.encoder_layers)]
        )
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        input_features: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:

        expected_seq_length = (
            self.max_source_positions * self.conv1.stride[0] * self.conv2.stride[0]
        )
        if input_features.shape[-1] != expected_seq_length:
            raise ValueError(
                f"Whisper expects the mel input features to be of length {expected_seq_length}, "
                f"but found {input_features.shape[-1]}. Make sure to pad the "
                f"input mel features to {expected_seq_length}."
            )

        inputs_embeds = nn.functional.gelu(self.conv1(input_features))
        inputs_embeds = nn.functional.gelu(self.conv2(inputs_embeds))
        inputs_embeds = inputs_embeds.permute(0, 2, 1)
        embed_pos = self.embed_positions.weight

        hidden_states = inputs_embeds + embed_pos

        for enc_block in self.layers:
            hidden_states = enc_block(hidden_states, kv_caches, input_metadata)

        return hidden_states


class WhisperForConditionalGeneration(nn.Module):
    def __init__(
        self, 
        config: WhisperConfig, 
        linear_method: Optional["LinearMethodBase"] = None # probably not needed
    ):
        super().__init__()
        self.config = config
        self.encoder = WhisperEncoder(config)
        self.sampler = Sampler(config.vocab_size)

    def forward(
        self,
        input_ids: torch.FloatTensor,
        positions: torch.Tensor,
        kv_caches: List[KVCache],
        input_metadata: InputMetadata,
    ) -> torch.Tensor:
        
        input_features = input_ids
        # it seems that before we run the actual inference, vLLM runs a forward pass
        # on initialization (for KV Cache profiling potentially?). This "mock" forward
        # pass pushes standard tokens through the forward() method. This needs to be
        # overriden
        if input_ids.dtype == torch.long:
            # inputs_ids is a standard input_ids tensor
            # replace it with a input_features tensor
            input_features = torch.zeros(1, 80, 3000, dtype=torch.bfloat16).to(input_features.device)
        return self.encoder(input_features, kv_caches, input_metadata)

    def sample(self, hidden_states: torch.Tensor, sampling_metadata: SamplingMetadata):
        # TODO: For now we are not implementing the sampling method
        return hidden_states

    def load_weights(
        self,
        model_name_or_path: str,
        cache_dir: Optional[str] = None,
        load_format: str = "auto",
        revision: Optional[str] = None,
    ):
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        print(params_dict.keys())
        print('-----------')
        for name, loaded_weight in hf_model_weights_iterator(
            model_name_or_path, cache_dir, load_format, revision
        ):  
            name = name.replace("model.", "")
            if name not in params_dict:
                print(f"{name} not in params_dict")
                continue
            #assert name in params_dict, f"{name} not in params_dict"
            param = params_dict[name]
            assert param.shape == loaded_weight.shape, (
                f"{name} shape mismatch between model and checkpoint: "
                f"{param.shape} != {loaded_weight.shape}")
            weight_loader = getattr(param, "weight_loader",
                                    default_weight_loader)
            weight_loader(param, loaded_weight)
