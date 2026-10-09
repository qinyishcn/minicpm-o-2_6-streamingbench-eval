"""Pinned llama.cpp b11516 C ABI; edge vision and server decoder are separate."""
import contextlib
import ctypes as C
import hashlib
import os
import struct
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .activation import validate_layout

P, I, U, F, B, Z = C.c_void_p, C.c_int32, C.c_uint32, C.c_float, C.c_bool, C.c_size_t
MODEL_SHA = '4f635fc0c0bb88d50ccd9cf1f1e5892b5cb085ff88fe0d8e1148fd9a8a836bc2'
PROJECTOR_SHA = 'efa4f7d96aa0f838f2023fc8d28e519179b16f1106777fa9280b32628191aa3e'
TOKENIZER_SHA = 'ee2af5fc10ab7468d746c3e7e677f16e9bf56f1f3f5625f53836401d025b1ba9'
RUNTIME_SHA = {
    'llama.dll': '98e8747e58bf38d6888b753c0e1d7b21c5f16a2978fc8b5e0752f82d1bf4227f',
    'mtmd.dll': '7773002ee46f8eb65fa6e73237372143cddb1a6db6886a858d447b0b0932f3a1',
    'ggml-vulkan.dll': '639e8c7e192d4d5bec10826f68224737a7eb0a671a8da090eff315d93aa598d3',
}


class ModelParams(C.Structure):
    _fields_ = [(n, P) for n in ['devices', 'tensor_buft_overrides']] + [
        (n, I) for n in ['n_gpu_layers', 'split_mode', 'load_mode', 'lazy_mode', 'main_gpu']
    ] + [(n, P) for n in ['tensor_split', 'progress_callback', 'progress_callback_user_data', 'kv_overrides']] + [
        (n, B) for n in ['vocab_only', 'check_tensors', 'use_extra_bufts', 'no_host', 'no_alloc', 'load_mtp']
    ]


class ContextParams(C.Structure):
    _fields_ = [(n, U) for n in ['n_ctx', 'n_batch', 'n_ubatch', 'n_seq_max', 'n_rs_seq', 'n_outputs_max', 'n_outputs_max_per_seq']] + [
        (n, I) for n in ['n_threads', 'n_threads_batch', 'ctx_type', 'rope_scaling_type', 'pooling_type', 'attention_type', 'flash_attn_type']
    ] + [(n, F) for n in ['rope_freq_base', 'rope_freq_scale', 'yarn_ext_factor', 'yarn_attn_factor', 'yarn_beta_fast', 'yarn_beta_slow']] + [
        ('yarn_orig_ctx', U), ('defrag_thold', F), ('cb_eval', P), ('cb_eval_user_data', P),
        ('type_k', I), ('type_v', I), ('moe_cache_size', Z), ('abort_callback', P), ('abort_callback_data', P)
    ] + [(n, B) for n in ['embeddings', 'offload_kqv', 'no_perf', 'op_offload', 'swa_full', 'kv_unified']] + [
        ('samplers', P), ('n_samplers', Z), ('ctx_other', P)
    ]


class VisionParams(C.Structure):
    _fields_ = [('use_gpu', B), ('device', P), ('print_timings', B), ('n_threads', C.c_int),
                ('image_marker', C.c_char_p), ('media_marker', C.c_char_p), ('flash_attn_type', I),
                ('warmup', B), ('image_min_tokens', C.c_int), ('image_max_tokens', C.c_int),
                ('cb_eval', P), ('cb_eval_user_data', P), ('batch_max_tokens', I),
                ('progress_callback', P), ('progress_callback_user_data', P)]


class InputText(C.Structure):
    _fields_ = [('text', C.c_char_p), ('text_len', Z), ('add_special', B), ('parse_special', B)]


class Embedding(C.Structure):
    _fields_ = [('data', C.POINTER(F)), ('n_rows', Z), ('n_embd', Z)]


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


class NativeAPI:
    def __init__(self, runtime, vision=False):
        self.runtime = Path(runtime).resolve()
        if (C.sizeof(P), C.sizeof(ModelParams), C.sizeof(ContextParams), C.sizeof(VisionParams)) != (8,80,168,96):
            raise RuntimeError('Unexpected native ABI layout; b11516 x64 is required')
        if os.name == 'nt':
            for filename in ['llama.dll','ggml-vulkan.dll'] + (['mtmd.dll'] if vision else []):
                if file_sha256(self.runtime/filename) != RUNTIME_SHA[filename]:
                    raise ValueError(f'Native ABI identity mismatch: {filename}; use b11516 Vulkan x64')
            self._dll_directory = os.add_dll_directory(str(self.runtime))
        name = lambda x: str(self.runtime/(x+'.dll' if os.name == 'nt' else 'lib'+x+'.so'))
        self.ggml = C.CDLL(name('ggml'))
        load = self.bind(self.ggml, 'ggml_backend_load_all_from_path', None, C.c_char_p)
        load(os.fsencode(self.runtime))
        self.llama = C.CDLL(name('llama'))
        self.bind(self.llama, 'llama_backend_init', None)()
        signatures = {
            'llama_model_default_params': (ModelParams, []),
            'llama_context_default_params': (ContextParams, []),
            'llama_model_load_from_file': (P, [C.c_char_p, ModelParams]),
            'llama_model_free': (None, [P]), 'llama_model_get_vocab': (P, [P]),
            'llama_model_n_embd_inp': (I, [P]), 'llama_vocab_n_tokens': (I, [P]),
            'llama_model_meta_val_str': (I, [P, C.c_char_p, P, Z]),
            'llama_init_from_model': (P, [P, ContextParams]), 'llama_free': (None, [P]),
            'llama_get_memory': (P, [P]), 'llama_memory_clear': (None, [P, B]),
            'llama_batch_ext_init': (P, [P]), 'llama_batch_ext_free': (None, [P]),
            'llama_batch_ext_clear': (None, [P]),
            'llama_batch_ext_add_token': (I, [P, I, I]),
            'llama_batch_ext_add_embd': (I, [P, I, Embedding]),
            'llama_batch_ext_set_pos': (B, [P, I, C.POINTER(I)]),
            'llama_batch_ext_set_output_logits': (B, [P, I, B]),
            'llama_process': (I, [P, I, P]),
            'llama_sampler_init_greedy': (P, []), 'llama_sampler_free': (None, [P]),
            'llama_sampler_sample': (I, [P, P, I]), 'llama_vocab_is_eog': (B, [P, I]),
            'llama_token_to_piece': (I, [P, I, P, I, I, B]),
            'llama_get_logits_ith': (C.POINTER(F), [P, I]),
        }
        for name, (result, arguments) in signatures.items():
            self.bind(self.llama, name, result, *arguments)
        if vision:
            self.mtmd = C.CDLL(name_for_library(self.runtime, 'mtmd'))
            signatures = {
                'mtmd_context_params_default': (VisionParams, []),
                'mtmd_init_from_file': (P, [C.c_char_p, P, VisionParams]), 'mtmd_free': (None, [P]),
                'mtmd_default_marker': (C.c_char_p, []),
                'mtmd_bitmap_init': (P, [U, U, P]), 'mtmd_bitmap_free': (None, [P]),
                'mtmd_input_chunks_init': (P, []), 'mtmd_input_chunks_free': (None, [P]),
                'mtmd_input_chunks_size': (Z, [P]), 'mtmd_input_chunks_get': (P, [P, Z]),
                'mtmd_input_chunk_get_type': (I, [P]), 'mtmd_input_chunk_get_n_tokens': (Z, [P]),
                'mtmd_input_chunk_get_n_pos': (I, [P]),
                'mtmd_input_chunk_get_tokens_text': (C.POINTER(I), [P, C.POINTER(Z)]),
                'mtmd_tokenize': (I, [P, P, C.POINTER(InputText), C.POINTER(P), Z]),
                'mtmd_encode_chunk': (I, [P, P]), 'mtmd_get_output_embd': (C.POINTER(F), [P]),
                'mtmd_helper_eval_chunks': (I, [P, P, P, I, I, I, B, C.POINTER(I)]),
            }
            for name, (result, arguments) in signatures.items():
                self.bind(self.mtmd, name, result, *arguments)

    @staticmethod
    def bind(library, name, result, *arguments):
        function = getattr(library, name)
        function.restype, function.argtypes = result, arguments
        return function


def name_for_library(runtime, name):
    return str(runtime/(name+'.dll' if os.name == 'nt' else 'lib'+name+'.so'))


class EdgeVision:
    """Loads only visual tensors; the small text asset contains zero tensors."""
    def __init__(self, runtime, tokenizer, projector, use_gpu=True):
        if file_sha256(tokenizer) != TOKENIZER_SHA:
            raise ValueError('Tokenizer-only asset identity mismatch; regenerate from pinned language GGUF')
        if struct.unpack('<Q', Path(tokenizer).read_bytes()[8:16])[0] != 0:
            raise ValueError('Edge tokenizer GGUF must contain zero tensors')
        if file_sha256(projector) != PROJECTOR_SHA:
            raise ValueError('Visual projector identity mismatch')
        self.api = NativeAPI(runtime, vision=True)
        ll = self.api.llama
        params = ll.llama_model_default_params()
        params.vocab_only, params.n_gpu_layers = True, 0
        self.model = ll.llama_model_load_from_file(os.fsencode(tokenizer), params)
        if not self.model:
            raise RuntimeError('Cannot load tokenizer-only metadata')
        # vocab_only skips runtime hparams, so n_embd_inp() is zero here.
        # Read the preserved GGUF metadata instead of inferring a tensor shape.
        buffer = C.create_string_buffer(64)
        size = ll.llama_model_meta_val_str(self.model, b'qwen2.embedding_length', buffer, len(buffer))
        if size<=0:
            raise ValueError('Tokenizer metadata lacks embedding dimension')
        self.width = int(buffer.value)
        if self.width!=3584:
            raise ValueError('Unexpected MiniCPM-o 2.6 embedding dimension')
        params = self.api.mtmd.mtmd_context_params_default()
        params.use_gpu, params.warmup, params.n_threads = use_gpu, False, 4
        params.flash_attn_type, params.batch_max_tokens = 1, 256
        self.vision = self.api.mtmd.mtmd_init_from_file(os.fsencode(projector), self.model, params)
        if not self.vision:
            raise RuntimeError('Cannot initialize MiniCPM vision encoder')
        self.marker = self.api.mtmd.mtmd_default_marker().decode()

    @contextlib.contextmanager
    def chunks(self, images, prompt):
        mt = self.api.mtmd
        bitmaps = []
        chunks = mt.mtmd_input_chunks_init()
        try:
            for path in images:
                with Image.open(path) as image:
                    image = image.convert('RGB')
                    pixels = C.create_string_buffer(image.tobytes())
                    bitmap = mt.mtmd_bitmap_init(image.width, image.height, pixels)
                    if not bitmap:
                        raise RuntimeError('Cannot create image bitmap')
                    bitmaps.append(bitmap)
            # Match the no-tools Qwen2 template used by the previous llama-server run.
            content = self.marker * len(images) + prompt
            formatted = ('<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.'
                         '<|im_end|>\n<|im_start|>user\n' + content +
                         '<|im_end|>\n<|im_start|>assistant\n').encode()
            text = InputText(formatted, len(formatted), True, True)
            array = (P*len(bitmaps))(*bitmaps)
            status = mt.mtmd_tokenize(self.vision, chunks, C.byref(text), array, len(bitmaps))
            if status:
                raise RuntimeError(f'MiniCPM tokenization failed: {status}')
            yield chunks
        finally:
            mt.mtmd_input_chunks_free(chunks)
            for bitmap in bitmaps:
                mt.mtmd_bitmap_free(bitmap)

    def encode(self, images, prompt):
        begin = time.perf_counter()
        mt = self.api.mtmd
        segments, payload = [], bytearray()
        with self.chunks(images, prompt) as chunks:
            for index in range(mt.mtmd_input_chunks_size(chunks)):
                chunk = mt.mtmd_input_chunks_get(chunks, index)
                kind = mt.mtmd_input_chunk_get_type(chunk)
                if kind == 0:
                    size = Z()
                    tokens = mt.mtmd_input_chunk_get_tokens_text(chunk, C.byref(size))
                    segments.append({'kind': 'tokens', 'ids': list(tokens[:size.value])})
                elif kind == 1:
                    count = mt.mtmd_input_chunk_get_n_tokens(chunk)
                    if mt.mtmd_encode_chunk(self.vision, chunk):
                        raise RuntimeError('MiniCPM visual encode failed')
                    data = C.string_at(mt.mtmd_get_output_embd(self.vision), count*self.width*4)
                    segments.append({'kind': 'embedding', 'shape': [count, self.width], 'dtype': '<f4',
                                     'offset': len(payload), 'bytes': len(data),
                                     'n_pos': mt.mtmd_input_chunk_get_n_pos(chunk)})
                    payload.extend(data)
                else:
                    raise ValueError('Only visual input is supported in this split')
        return {'protocol': 1, 'model_sha256': MODEL_SHA, 'projector_sha256': PROJECTOR_SHA,
                'chunks': segments, 'edge_encode_seconds': time.perf_counter()-begin}, bytes(payload)

    def close(self):
        self.api.mtmd.mtmd_free(self.vision)
        self.api.llama.llama_model_free(self.model)


class LanguageDecoder:
    """Language-only server: no mtmd library or visual weights are loaded."""
    def __init__(self, runtime, model, context=4096):
        if file_sha256(model) != MODEL_SHA:
            raise ValueError('Language model identity mismatch')
        self.api = NativeAPI(runtime)
        ll = self.api.llama
        params = ll.llama_model_default_params()
        params.n_gpu_layers = -1
        self.model = ll.llama_model_load_from_file(os.fsencode(model), params)
        if not self.model:
            raise RuntimeError('Cannot load language model')
        self.width = ll.llama_model_n_embd_inp(self.model)
        self.vocab = ll.llama_model_get_vocab(self.model)
        params = ll.llama_context_default_params()
        params.n_ctx, params.n_batch, params.n_ubatch, params.n_seq_max = context, 512, 128, 1
        params.n_threads, params.n_threads_batch, params.flash_attn_type = 4, 4, 1
        self.context_size = context
        self.context = ll.llama_init_from_model(self.model, params)
        if not self.context:
            raise RuntimeError('Cannot initialize language context')
        self.batch = ll.llama_batch_ext_init(self.context)

    def reset(self):
        ll = self.api.llama
        ll.llama_memory_clear(ll.llama_get_memory(self.context), True)

    def _decode(self, values, position, embedding=False, output=False):
        ll = self.api.llama
        ll.llama_batch_ext_clear(self.batch)
        if embedding:
            values = np.ascontiguousarray(values, dtype=np.float32)
        for index, value in enumerate(values):
            if embedding:
                item = Embedding(value.ctypes.data_as(C.POINTER(F)), 1, self.width)
                idx = ll.llama_batch_ext_add_embd(self.batch, 0, item)
            else:
                idx = ll.llama_batch_ext_add_token(self.batch, 0, int(value))
            if idx < 0:
                raise RuntimeError(f'Native batch rejected input: {idx}')
            pos = I(position+index)
            if not ll.llama_batch_ext_set_pos(self.batch, idx, C.byref(pos)):
                raise RuntimeError('Native batch rejected position')
            if output and index == len(values)-1:
                ll.llama_batch_ext_set_output_logits(self.batch, idx, True)
        status = ll.llama_process(self.context, 1, self.batch)
        if status:
            raise RuntimeError(f'Language decode failed: {status}')
        return position+len(values)

    def generate(self, metadata, payload, max_tokens=16):
        if metadata.get('model_sha256') != MODEL_SHA or metadata.get('projector_sha256') != PROJECTOR_SHA:
            raise ValueError('Split model identities do not match server')
        validate_layout(metadata, payload, self.width, self.api.llama.llama_vocab_n_tokens(self.vocab),
                        self.context_size, max_tokens)
        segments = metadata['chunks']
        self.reset()
        position = 0
        for i, segment in enumerate(segments):
            embedding = segment['kind']=='embedding'
            if embedding:
                n, width = segment['shape']
                values = np.frombuffer(payload, dtype='<f4', count=n*width, offset=segment['offset']).reshape(n,width)
            else:
                values = segment['ids']
            for start in range(0,len(values),512):
                output = i==len(segments)-1 and start+512>=len(values)
                position = self._decode(values[start:start+512],position,embedding,output)
        return self.finish(position,max_tokens)

    def finish(self, position, max_tokens=16):
        ll = self.api.llama
        sampler = ll.llama_sampler_init_greedy()
        tokens, pieces = [], bytearray()
        first_logits = np.ctypeslib.as_array(ll.llama_get_logits_ith(self.context,-1),
                                            shape=(ll.llama_vocab_n_tokens(self.vocab),)).copy()
        try:
            for _ in range(max_tokens):
                token = ll.llama_sampler_sample(sampler,self.context,-1)
                if ll.llama_vocab_is_eog(self.vocab,token):
                    break
                tokens.append(token)
                buffer = C.create_string_buffer(512)
                size = ll.llama_token_to_piece(self.vocab,token,buffer,512,0,False)
                if size<0:
                    raise RuntimeError('Token piece exceeds buffer')
                pieces.extend(buffer.raw[:size])
                position = self._decode([token],position,output=True)
        finally:
            ll.llama_sampler_free(sampler)
        top = np.argsort(first_logits)[-8:][::-1]
        return {'text': pieces.decode('utf-8'), 'token_ids': tokens,
                'first_logits_sha256': hashlib.sha256(first_logits.tobytes()).hexdigest(),
                'first_logits_top8': [{'token': int(t), 'logit': float(first_logits[t])} for t in top],
                'prompt_tokens': position-len(tokens), 'generated_tokens': len(tokens)}

    def close(self):
        ll = self.api.llama
        ll.llama_batch_ext_free(self.batch)
        ll.llama_free(self.context)
        ll.llama_model_free(self.model)
