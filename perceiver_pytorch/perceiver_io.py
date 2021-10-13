import torch
from math import pi
from functools import wraps
from torch import nn, einsum
import torch.nn.functional as F
from torch.autograd import Variable
# noinspection PyProtectedMember
from einops import rearrange, repeat

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn

def fourier_encode(x, max_freq, num_bands = 4):
    x = x.unsqueeze(-1)
    device, dtype, orig_x = x.device, x.dtype, x

    scales = torch.linspace(1., max_freq / 2, num_bands, device = device, dtype = dtype)
    scales = scales[(*((None,) * (len(x.shape) - 1)), Ellipsis)]

    x = x * scales * pi
    x = torch.cat([x.sin(), x.cos()], dim = -1)
    x = torch.cat((x, orig_x), dim = -1)
    return x

# helper classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

    def forward(self, x, context = None, mask = None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.to_out(out)

# main class

class PerceiverIO(nn.Module):
    def __init__(
        self,
        input_dim,
        num_input_axes,
        output_dim,
        queries_dim=1024,
        network_depth=6,
        num_latents = 256,
        latent_dim=512,
        num_cross_att_heads=1,
        num_self_att_heads=8,
        cross_head_dim=64,
        latent_head_dim=64,
        weight_tie_layers=False,
        learn_query=False,
        query_shape=None,
        fourier_encode_input=False,
        num_fourier_freq_bands=None,
        max_fourier_freq=None,
        decoder_ff = False
        # ToDo: add support for variable number of cross-attends
    ):
        super().__init__()
        if fourier_encode_input and (not exists(num_fourier_freq_bands) or not exists(max_fourier_freq)):
            raise Exception('when fourier_encode_input is selected, both num_fourier_freq_bands and max_fourier_freq'
                            'must be specified, but found {} and {} respectively'.format(num_fourier_freq_bands,
                                                                                         max_fourier_freq))
        self._input_axis = num_input_axes
        self._output_dim = output_dim
        self._max_freq = max_fourier_freq
        self._num_freq_bands = num_fourier_freq_bands
        query_shape = default(query_shape, [])

        # ToDo: set the correct initializatin scheme for the query here
        self._queries = nn.Parameter(torch.randn(*query_shape, queries_dim)) if learn_query else None

        self._fourier_encode_data = fourier_encode_input
        fourier_channels = (num_input_axes * ((num_fourier_freq_bands * 2) + 1)) if fourier_encode_input else 0
        input_dim = fourier_channels + input_dim

        self._latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        self._cross_attend_blocks = nn.ModuleList([
            PreNorm(latent_dim, Attention(
                latent_dim, input_dim, heads = num_cross_att_heads, dim_head = cross_head_dim),
                    context_dim = input_dim),
            PreNorm(latent_dim, FeedForward(latent_dim))
        ])

        get_latent_attn = lambda: PreNorm(latent_dim, Attention(
            latent_dim, heads = num_self_att_heads, dim_head = latent_head_dim))
        get_latent_ff = lambda: PreNorm(latent_dim, FeedForward(latent_dim))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self._layers = nn.ModuleList([])
        cache_args = {'_cache': weight_tie_layers}

        for i in range(network_depth):
            self._layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

        self._decoder_cross_attn = PreNorm(queries_dim, Attention(
            queries_dim, latent_dim, heads = num_cross_att_heads, dim_head = cross_head_dim), context_dim = latent_dim)
        self._decoder_ff = PreNorm(queries_dim, FeedForward(queries_dim)) if decoder_ff else None

        self._to_logits = nn.Linear(queries_dim, output_dim) if exists(output_dim) else nn.Identity()

    def forward(
        self,
        data,
        mask = None,
        queries = None
    ):
        # noinspection PyTupleAssignmentBalance
        b, *axis, _, device = *data.shape, data.device

        if self._fourier_encode_data:
            # calculate fourier encoded positions in the range of [-1, 1], for all axis

            axis_pos = list(map(lambda size: torch.linspace(-1., 1., steps = size, device = device), axis))
            pos = torch.stack(torch.meshgrid(*axis_pos), dim = -1)
            enc_pos = fourier_encode(pos, self._max_freq, self._num_freq_bands)
            enc_pos = rearrange(enc_pos, '... n d -> ... (n d)')
            enc_pos = repeat(enc_pos, '... -> b ...', b = b)

            data = torch.cat((data, enc_pos), dim = -1)

        data = rearrange(data, 'b ... d -> b (...) d')

        x = repeat(self._latents, 'n d -> b n d', b = b)

        cross_attn, cross_ff = self._cross_attend_blocks

        # cross attention only happens once for Perceiver IO

        x = cross_attn(x, context = data, mask = mask) + x
        x = cross_ff(x) + x

        # layers

        for self_attn, self_ff in self._layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        if not exists(queries):
            if exists(self._queries):
                queries = repeat(self._queries, '... -> b ...', b=b)
            else:
                return x

        queries_shape = list(queries.shape)

        queries = rearrange(queries, 'b ... d -> b (...) d')

        # cross attend from decoder queries to latents
        
        latents = self._decoder_cross_attn(queries, context = x)

        # optional decoder feedforward

        if exists(self._decoder_ff):
            latents = latents + self._decoder_ff(latents)

        # final linear out

        ret = self._to_logits(latents)

        # reshape to correct number of axes
        return torch.reshape(ret, queries_shape[:-1] + [self._output_dim])
