# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Forward step utilities."""

from collections.abc import Iterable

import torch

from megatron import get_args
from megatron.core import mpu
from .inference_params import InferenceParams
from .communication import (
    send_to_next_pipeline_rank,
    recv_from_prev_pipeline_rank_,
    send_list_to_next_pipeline_rank,
    recv_list_from_prev_pipeline_rank)


class ForwardStep:
    """Forward step function with all the communications.
    We use a class here to hide the inference parameters
    from the outside caller."""

    def __init__(self, model, max_batch_size=0, max_sequence_length=0, early_exit_thres=0, inference_params=None):
        """Set values so we don't need to do it multiple times."""
        # Make sure model is in eval mode.
        assert not isinstance(model, Iterable), \
            'interleaving schedule is not supported for inference'
        model.eval()
        self.model = model
        # Initialize inference parameters.
        if inference_params is None:
            self.inference_params = InferenceParams(max_batch_size,
                                                max_sequence_length, early_exit_thres)
        else:
            self.inference_params = inference_params
        # Pipelining arguments.
        args = get_args()
        self.pipeline_size_larger_than_one = (
            args.pipeline_model_parallel_size > 1)
        # Threshold of pipelining.
        self.pipelining_batch_x_seqlen = \
            args.inference_batch_times_seqlen_threshold


    def __call__(self, tokens, position_ids, attention_mask, req_ids=[]):
        """Invocation of the forward methods. Note that self.inference_params
        is being modified by the forward step."""
        # Pipelining case.
        if self.pipeline_size_larger_than_one:
            return _with_early_exit_pipelining_forward_step(self.model,
                                                     tokens,
                                                     position_ids,
                                                     attention_mask,
                                                     self.inference_params)

        return _no_pipelining_forward_step(self.model,
                                           tokens,
                                           position_ids,
                                           attention_mask,
                                           self.inference_params,
                                           req_ids=req_ids)



def _get_recv_buffer_dtype(args):
    """Receive happens between the layers."""
    if args.fp32_residual_connection:
        return torch.float
    return args.params_dtype



def _allocate_recv_buffer(batch_size, sequence_length):
    """Receive happens between the layers with size [s, b, h]."""
    if mpu.is_pipeline_first_stage():
        return None
    args = get_args()
    recv_size = (sequence_length, batch_size, args.hidden_size)
    return torch.empty(recv_size,
                       dtype=_get_recv_buffer_dtype(args),
                       device=torch.cuda.current_device())



def _forward_step_helper(model, tokens, position_ids, attention_mask,
                         inference_params, recv_buffer=None, req_ids=[]):
    """Single forward step. Update the allocate memory flag so
    only the first time the memory is allocated."""
    batch_size = tokens.size(0)
    sequence_length = tokens.size(1)
    if recv_buffer is None:
        recv_buffer = _allocate_recv_buffer(batch_size, sequence_length)

    # Receive from previous stage.
    recv_from_prev_pipeline_rank_(recv_buffer)

    # Forward pass through the model.
    model.set_input_tensor(recv_buffer)
    output_tensor, early_exit_ids = model(tokens, position_ids, attention_mask,
                          inference_params=inference_params, req_ids=req_ids)

    # Send output to the next stage.
    send_to_next_pipeline_rank(output_tensor)

    print(f"[ForwardStep: _forward_step_helper] returning output_tensor size: {output_tensor.size()}, early_exit_ids: {early_exit_ids}")

    return output_tensor, early_exit_ids



def _no_pipelining_forward_step(model, tokens, position_ids, attention_mask,
                                inference_params, recv_buffer=None, req_ids=[]):
    """If recv_buffer is none, we will allocate one on the fly."""
    # Run a simple forward pass.
    output_tensor = _forward_step_helper(model, tokens, position_ids,
                                         attention_mask, inference_params,
                                         recv_buffer=recv_buffer, req_ids=req_ids)
    # Update the sequence length offset.
    # inference_params.sequence_len_offset += tokens.size(1)

    logits = None
    if mpu.is_pipeline_last_stage():
        logits = output_tensor

    return logits



def _with_pipelining_forward_step(model, tokens, position_ids, attention_mask,
                                  inference_params, micro_batch_size):
    """No interleaving is supported."""
    sequence_length = tokens.size(1)
    batch_size = tokens.size(0)

    # Divide the batch dimension into micro batches.
    num_micro_batches, last_chunk = divmod(batch_size,
                                           micro_batch_size)
    if last_chunk > 0:
        num_micro_batches += 1

    # Preallocate memory for output logits.
    logits = None
    if mpu.is_pipeline_last_stage():
        args = get_args()
        logits = torch.empty(
            (batch_size, sequence_length, args.padded_vocab_size),
            dtype=torch.float32, device=torch.cuda.current_device())

    # Preallocate recv buffer.
    recv_buffer = _allocate_recv_buffer(micro_batch_size, sequence_length)

    for micro_batch_index in range(num_micro_batches):
        # Slice among the batch dimenion.
        start = micro_batch_index * micro_batch_size
        end = min(start + micro_batch_size, batch_size)
        this_micro_batch_size = end - start
        tokens2use = tokens[start:end, ...]
        position_ids2use = position_ids[start:end, ...]

        # Run a simple forward pass.
        if this_micro_batch_size != micro_batch_size:
            recv_buffer = None
        output = _forward_step_helper(model, tokens2use, position_ids2use,
                                      attention_mask, inference_params,
                                      recv_buffer=recv_buffer)

        # Adjust the batch size offset to account for the micro-batch.
        inference_params.batch_size_offset += this_micro_batch_size

        # Copy logits.
        if mpu.is_pipeline_last_stage():
            logits[start:end, ...] = output

    # Once we are done with all the micro-batches, we can
    # adjust the sequence length offset.
    inference_params.sequence_len_offset += sequence_length
    # and reset the batch size offset
    inference_params.batch_size_offset = 0

    return logits


def _allocate_early_exit_recv_buffers(batch_size, sequence_length):
    if mpu.is_pipeline_first_stage():
        return None
    args = get_args()
    recv_size = (sequence_length, batch_size, args.hidden_size)
    return [torch.empty(recv_size,
                       dtype=_get_recv_buffer_dtype(args),
                       device=torch.cuda.current_device()),
            torch.empty(1, dtype=torch.int8, device=torch.cuda.current_device())]


def _with_early_exit_pipelining_forward_step(model, tokens, position_ids, attention_mask,
                                  inference_params):
    """No interleaving is supported."""
    sequence_length = tokens.size(1)
    batch_size = tokens.size(0)
    assert batch_size == 1, "early exit not support batch inference yet"
    # Divide the batch dimension into micro batches.
    # Preallocate recv buffer.
    if not mpu.is_pipeline_first_stage():
        recv_buffers = _allocate_early_exit_recv_buffers(batch_size, sequence_length)
        recv_list_from_prev_pipeline_rank(recv_buffers)
        model.set_input_tensor(recv_buffers[0])
        inference_params.prev_has_early_exited = bool(recv_buffers[1])
    output_tensor = model(tokens, position_ids, attention_mask, inference_params=inference_params)
    signal_tensor = torch.tensor([int(inference_params.has_early_exited or inference_params.prev_has_early_exited)],
                                 dtype=torch.int8,
                                 device=torch.cuda.current_device())
    send_list_to_next_pipeline_rank([output_tensor, signal_tensor])
    inference_params.sequence_len_offset += sequence_length
    return output_tensor