# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.

"""Generation utilities."""

import torch
import torch.nn.functional as F

from megatron import get_args, get_tokenizer
from megatron.core import mpu
from megatron.utils import get_ltor_masks_and_position_ids
from .communication import (
    copy_from_last_to_first_pipeline_stage,
    send_token_and_probs_to_first_pipeline_stage,
    recv_token_and_probs,
    broadcast_from_last_pipeline_stage,
    broadcast_from_first_pipeline_stage,
    broadcast_from_last_to_first_pipeline_stage)
from .inference_params import InferenceParams
from .forward_step import ForwardStep
from .sampling import sample
from .beam_utils import BeamHypotheses

def score_and_return_on_first_stage(model, tokens, lengths):
    """Function for just scoring.
    Arguments:
        model: no interleaving is supported.
        tokens: prompt tokens extended to be of size [b, max_prompt_length]
        lengths: original prompt length, size: [b]tokenizer
    Note: Outside of model, other parameters only need to be available on
          rank 0.
    Outputs: 
        output_log_probs: log probability of the selected tokens. size: [b, s]
    """

    args = get_args()

    batch_size = tokens.size(0)
    max_prompt_length = lengths.max().item()
    assert max_prompt_length == tokens.size(1)
    
    if max_prompt_length > args.max_position_embeddings:
        raise ValueError("Length of prompt + tokens_to_generate longer than allowed")
    
    if max_prompt_length * batch_size > args.max_tokens_to_oom:
        raise ValueError("Too many tokens.  " + str(max_prompt_length*batch_size)+ " is greater than "+str(args.max_tokens_to_oom))

    # forward step.
    forward_step = ForwardStep(model, batch_size, max_prompt_length)

    # ===================
    # Pre-allocate memory
    # ===================

    # Log probability of the sequence (prompt + generated tokens).
    output_log_probs = None
    output_log_probs_size = (batch_size, max_prompt_length - 1)
    
    if mpu.is_pipeline_last_stage():
        output_log_probs = torch.empty(output_log_probs_size,
                                       dtype=torch.float32,
                                       device=torch.cuda.current_device())
    
    # =============
    # Run infernece
    # =============
    with torch.no_grad():
        attention_mask, position_ids = _build_attention_mask_and_position_ids(tokens)
        
        # logits will be meanigful only in the last pipeline stage.
        logits = forward_step(tokens, position_ids, attention_mask)

        if mpu.is_pipeline_last_stage():
            # Always the last stage should have an output.
            assert logits is not None
            log_probs = F.log_softmax(logits, dim=2)
            
            # Pick the tokens that we need to get the log
            # probabilities for. Note that next input token is
            # the token which we selected in the current logits,
            # so shift by 1.
            indices = torch.unsqueeze(tokens[:, 1:], 2)
            output_log_probs = torch.gather(log_probs, 2, indices).squeeze(2)
    
    # ======================================
    # Broadcast to the first pipeline stage.
    # ======================================
    output_log_probs = broadcast_from_last_to_first_pipeline_stage(
        output_log_probs_size, torch.float32, output_log_probs)
    
    return tokens, lengths, output_log_probs, logits

def generate_tokens_probs_and_return_on_first_stage(
        model, tokens, lengths,
        return_output_log_probs=False,
        top_k=0, top_p=0.0, top_p_decay=0.0, top_p_bound=0.0,
        temperature=1.0,
        use_stop_tokens_for_early_termination=True,
        stop_on_double_eol=False,
        stop_on_eol=False,
        stop_tokens=None,
        prevent_newline_after_colon=True,
        echo_prompts=False,
        early_exit_thres=1.0,
        use_early_exit=False,
        print_max_prob=False,
        exit_layers=[],
        req_ids=[],
        buffered_tokens={},):
    """Main token generation function.
    Arguments:
        model: no interleaving is supported.
        tokens: prompt tokens extended to be of size [b, max-sequence-length]
        lengths: original prompt length, size: [b]
        return_output_log_probs: flag to calculate the log probability of
            the generated tokens. Note that the log probability is the one
            from the original logit.
        top_k, top_p: top-k and top-p sampling parameters.
            Note that top-k = 1 is gready. Also, these paramters are
            exclusive meaning that:
                if top-k > 0 then we expect top-p=0.
                if top-p > 0 then we check for top-k=0.
        temperature: sampling temperature.
        use_eod_token_for_early_termination: if True, do early termination if
            all the sequences have reached this token.
        prevent_newline_after_colon: if True, it will disable generating new line \n after :
    Note: Outside of model, other parameters only need to be available on
          rank 0.
    Outputs: Note that is size is adjusted to a lower value than
             max-sequence-length if generation is terminated early.
        tokens: prompt and generated tokens. size: [b, :]
        generated_sequence_lengths: total length (including prompt) of
            the generated sequence. size: [b]
        output_log_probs: log probability of the selected tokens. size: [b, s]
    """

    args = get_args()
    tokenizer = get_tokenizer()

    batch_size = tokens.size(0)
    min_prompt_length = lengths.min().item()
    max_sequence_length = tokens.size(1)

    if max_sequence_length > args.max_position_embeddings:
        raise ValueError(f"Length of prompt + tokens_to_generate ({max_sequence_length}) longer than allowed ({args.max_position_embeddings})")
    
    if max_sequence_length * batch_size > args.max_tokens_to_oom:
        raise ValueError("Too many tokens.  " + str(max_sequence_length*batch_size)+ " is greater than "+str(args.max_tokens_to_oom))

    inference_params = InferenceParams(batch_size, max_sequence_length,
                                       top_k=top_k, top_p=top_p,
                                       temperature=temperature,
                                       top_p_bound=top_p_bound,
                                       top_p_decay=top_p_decay,
                                       early_exit_thres=early_exit_thres,
                                       use_early_exit=use_early_exit,
                                       print_max_prob=print_max_prob,
                                       exit_layers=exit_layers)
    
    print("batch_size", batch_size)
    # forward step.
    forward_step = ForwardStep(model, inference_params=inference_params)

    # Added termination_id to support the case that we want to terminate the
    # generation once that id is generated.
    if hasattr(args, 'eos_id'):
        termination_id = args.eos_id
    else:
        termination_id = tokenizer.eod

    # ===================
    # Pre-allocate memory
    # ===================

    # Log probability of the sequence (prompt + generated tokens).
    output_log_probs = None
    output_log_probs_size = (batch_size, max_sequence_length - 1)
    # Lengths of generated seuquence including including prompts.
    generated_sequence_lengths = None
    if mpu.is_pipeline_last_stage():
        if return_output_log_probs:
            output_log_probs = torch.empty(output_log_probs_size,
                                           dtype=torch.float32,
                                           device=torch.cuda.current_device())
        generated_sequence_lengths = torch.ones(
                batch_size, dtype=torch.int64,
                device=torch.cuda.current_device()) * max_sequence_length
    
    # Whether we have reached a termination id.
    is_generation_done = torch.zeros(batch_size, dtype=torch.uint8,
                                     device=torch.cuda.current_device())

    # =============
    # Run infernece
    # =============

    with torch.no_grad():
        attention_mask, position_ids = _build_attention_mask_and_position_ids(
            tokens)
        prev_context_length = 0
        full_exit_context_length = 0
        for context_length in range(min_prompt_length, max_sequence_length):

            # Pick the slice that we need to pass through the network.
            tokens2use = tokens[:, full_exit_context_length:context_length]
            positions2use = position_ids[:, full_exit_context_length:context_length]
            attention_mask2use = attention_mask[
                ..., full_exit_context_length:context_length, :context_length]

            # logits will be meanigful only in the last pipeline stage.
            logits, exited_req_ids = forward_step(tokens2use, positions2use, attention_mask2use, req_ids=req_ids)
            print(f"exited_req_ids: {exited_req_ids}")

            if mpu.is_pipeline_last_stage():
                if prevent_newline_after_colon:
                    logits[tokens2use[:, -1] == tokenizer.tokenize(':')[0], -1, tokenizer.tokenize('\n')[0]] = -1e10 # disable "\n" after ":"
                # Always the last stage should have an output.
                assert logits is not None

                # Sample.
                last_token_logits = logits[:, -1, :]
                new_sample = sample(last_token_logits,
                                    top_k=top_k,
                                    top_p=top_p,
                                    temperature=temperature,
                                    vocab_size=tokenizer.vocab_size)
                if top_p > 0.0 and top_p_decay > 0.0:
                    top_p = top_p * top_p_decay
                    if top_p_bound > 0.0:
                        top_p = max(top_p, top_p_bound)

                # If a prompt length is smaller or equal th current context
                # length, it means we have started generating tokens
                started = lengths <= context_length

                # Update the tokens.
                print(f"tokens size: {tokens.size()}")
                print(f"new sample size: {new_sample.size()}")
                # [TODO]Match the tokens to be the req_ids that EE:
                # 1. Remove the tokens with the req_ids that are not in exited_req_ids (through slicing)
                # 2. The removed tokens are stored in the dictinary buffered_tokens ({req_id: tokens}).
                # 3. If exited_req_ids contains req_ids that are not in the original req_ids, we need to fetch that tokens from buffered_tokens and append the tokens together.
                exited_idx = [i for i, req_id in enumerate(req_ids) if req_id in exited_req_ids] # the idx of the tokens that exited
                if len(exited_idx) != len(req_ids): # Not all requests exited: we need to buffer the tokens of the requests that are not exited
                    # Step 1
                    exited_tokens = tokens[exited_idx, :]

                    # Step 2
                    for i, req_id in enumerate(req_ids):
                        if req_id not in exited_req_ids:
                            buffered_tokens[req_id] = tokens[i]
                    
                    tokens = exited_tokens

                    # Step 3
                    req_ids_to_be_appended = [req_id for req_id in exited_req_ids if req_id not in req_ids]
                    if len(req_ids_to_be_appended) > 0:
                        for req_id in req_ids_to_be_appended:
                            tokens = torch.cat((tokens, buffered_tokens[req_id].unsqueeze(0)), dim=0)
                    
                tokens[started, context_length] = new_sample[started]

                # Calculate the log probabilities.
                if return_output_log_probs:
                    log_probs = F.log_softmax(logits, dim=2)
                    # Pick the tokens that we need to get the log
                    # probabilities for. Note that next input token is
                    # the token which we selected in the current logits,
                    # so shift by 1.
                    indices = torch.unsqueeze(
                        tokens[
                            :,
                            (prev_context_length + 1):(context_length + 1)],
                        2)
                    output_log_probs[:,
                                        prev_context_length:context_length] = \
                        torch.gather(log_probs, 2, indices).squeeze(2)

            # Update the tokens on the first stage so the next input to
            # the network is correct.
            copy_from_last_to_first_pipeline_stage(batch_size, torch.int64,
                                                   tokens[:, context_length])

            # Update the context length for the next token generation.
            prev_context_length = context_length
            if not inference_params.has_early_exited:
                full_exit_context_length = prev_context_length
                inference_params.sequence_len_offset += tokens2use.size(1)
            inference_params.has_early_exited = False
            inference_params.is_first_step = False

            # Check if all the sequences have hit the termination_id.
            done = None
            if mpu.is_pipeline_last_stage():
                # TODO(rprenger) These stopping methods are tokenizer dependent
                # instead tokenization should be in the inference loop so stop sequences can be used
                if stop_tokens is not None and len(stop_tokens) > 0:
                    done_token = torch.any(
                        new_sample.expand(stop_tokens.shape[0], new_sample.shape[0]) == stop_tokens.unsqueeze(dim=1), dim=0) \
                        & started.byte()

                else: 
                    done_token = (new_sample == termination_id).byte() & \
                        started.byte()
                
                just_finished = (done_token & ~is_generation_done).bool()
                generated_sequence_lengths[just_finished.view(-1)] = \
                    context_length + 1
                is_generation_done = is_generation_done | done_token
                done = torch.all(is_generation_done)
            done = broadcast_from_last_pipeline_stage(1, torch.uint8,
                                                      tensor=done)
            if use_stop_tokens_for_early_termination and done:
                break
            
    # ===================================================
    # Update the length of based on max generated length.
    # ===================================================

    tokens = tokens[:, :(context_length + 1)]
    if mpu.is_pipeline_last_stage():
        if return_output_log_probs:
            output_log_probs = output_log_probs[:, :context_length].contiguous()

    # ======================================
    # Broadcast to the first pipeline stage.
    # ======================================

    generated_sequence_lengths = broadcast_from_last_to_first_pipeline_stage(
        batch_size, torch.int64, generated_sequence_lengths)
    if return_output_log_probs:
        output_log_probs_size = (batch_size, context_length)
        output_log_probs = broadcast_from_last_to_first_pipeline_stage(
            output_log_probs_size, torch.float32, output_log_probs)
    if not echo_prompts and mpu.is_pipeline_first_stage():
        generated_sequence_lengths -= lengths
        for i, (sequence, length) in enumerate(zip(tokens, lengths)):
            tokens[i] = sequence.roll(-length.item(), dims=0)
        if return_output_log_probs:
            for i, (prob, length) in enumerate(zip(output_log_probs, lengths)):
                output_log_probs[i] = prob.roll(-(length.item() - 1), dims=0)
    return tokens, generated_sequence_lengths, output_log_probs, None, exited_req_ids, buffered_tokens

def beam_search_and_return_on_first_stage(model, tokens, lengths, beam_size, stop_token, num_return_gen, length_penalty, prevent_newline_after_colon=True):
    args = get_args()
    tokenizer = get_tokenizer()

    batch_size = tokens.size(0)
    assert(batch_size == 1)
    prompt_length = lengths.item()
    final_sequence_length = tokens.size(1)
    final_sequence_length = min(final_sequence_length, args.max_position_embeddings)
    
    # If the context is too big, this happens
    if prompt_length >= final_sequence_length:
        raise ValueError("context length + tokens_to_generate too large")

    # forward step.
    forward_step = ForwardStep(model, beam_size, final_sequence_length)

    beam_hyp = BeamHypotheses(beam_size, length_penalty)
    best_batches = None
    done = torch.zeros(1, dtype=torch.uint8, device=torch.cuda.current_device())
    scores = torch.zeros(beam_size,
                         dtype=torch.float32,
                         device=torch.cuda.current_device()).unsqueeze(1)
    scores_size_tensor, tokens_size_tensor = None, None
    # =============
    # Run infernece
    # =============
    with torch.no_grad():
        tokens = tokens.repeat(beam_size, 1)
        attention_mask, position_ids = _build_attention_mask_and_position_ids(tokens)
        prev_context_length = 0
        for context_length in range(prompt_length, final_sequence_length):

            # Pick the slice that we need to pass through the network.
            tokens2use = tokens[:, prev_context_length:context_length]
            positions2use = position_ids[:, prev_context_length:context_length]
            attention_mask2use = attention_mask[
                ..., prev_context_length:context_length, :context_length]

            # logits will be meanigful only in the last pipeline stage.
            logits = forward_step(tokens2use, positions2use, attention_mask2use)

            if mpu.is_pipeline_last_stage():
                if prevent_newline_after_colon:
                    logits[tokens2use[:, -1] == tokenizer.tokenize(':')[0], -1, tokenizer.tokenize('\n')[0]] = -1e10 # disable "\n" after ":"
                vocab_size = logits.size(2)
                log_probs = F.log_softmax(logits, dim=2)
                new_scores = log_probs[:, -1, :] + scores

                if context_length == prompt_length:  # if this is the first one
                    sorted_scores, indices = torch.sort(new_scores[0,:], descending=True)
                else:
                    sorted_scores, indices = torch.sort(new_scores.view(-1), descending=True)

                best_beam_ids = torch.div(indices[: 2 * beam_size], vocab_size).trunc().long()
                best_words = indices[:2 * beam_size] % vocab_size
                best_scores = sorted_scores[: 2 * beam_size]

                next_beams = []
                for beam_token_rank, (token_id, beam_score, beam_id) in enumerate(
                    zip(best_words, best_scores, best_beam_ids)
                ):
                    if token_id.item() == stop_token:
                        # if beam_token does not belong to top num_beams tokens, it should not be added
                        is_beam_token_worse_than_top_num_beams = beam_token_rank >= beam_size
                        if is_beam_token_worse_than_top_num_beams:
                            continue
                        beam_hyp.add(
                            tokens[beam_id].clone(),
                            beam_score,
                            context_length + 1 - prompt_length
                        )
                    else:
                        # add next predicted token since it is not eos_token
                        next_beams.append((token_id, beam_score, beam_id))

                    if len(next_beams) == beam_size:
                        break

                if beam_hyp.is_done(best_scores.max().item(), context_length + 1 - prompt_length):
                    done = torch.ones(1, dtype=torch.uint8, device=torch.cuda.current_device())
            
                best_batches = tokens.new([item[2] for item in next_beams])
                tokens = tokens[best_batches,:]
                tokens[:, context_length] = tokens.new([item[0] for item in next_beams])
                scores = scores.new([item[1] for item in next_beams]).unsqueeze(1)
          
            # torch.distributed.barrier()
            done = broadcast_from_last_pipeline_stage(1, torch.uint8, done)
            if done:
                break

            # Update the tokens on the first stage so the next input to
            # the network is correct.
            copy_from_last_to_first_pipeline_stage(tokens.size(), torch.int64,
                                                   tokens)

            # set inference key values to make it consistent with best beam index
            best_batches = broadcast_from_last_pipeline_stage(beam_size, torch.int64, best_batches)
            forward_step.inference_params.swap_key_value_dict(best_batches)

            # Update the context length for the next token generation.
            prev_context_length = context_length

        if mpu.is_pipeline_last_stage():
            # if cannot find stop token, add open beams to hyps
            if not done:
                for beam_id in range(beam_size):
                    beam_hyp.add(tokens[beam_id].clone(), scores[beam_id].squeeze(), context_length + 1 - prompt_length)

            # rank based on scores
            sorted_hyps = sorted(beam_hyp.beams, key=lambda x: x[0], reverse=True)
            num_return_gen = min(num_return_gen, len(sorted_hyps))
            scores = [sorted_hyps[i][0] for i in range(num_return_gen)]
            tokens = [sorted_hyps[i][1] for i in range(num_return_gen)]
            scores = torch.stack(scores, dim=0)
            tokens = torch.stack(tokens, dim=0)
            scores_size_tensor = torch.tensor(scores.shape, dtype=torch.int64, device=torch.cuda.current_device())
            tokens_size_tensor = torch.tensor(tokens.shape, dtype=torch.int64, device=torch.cuda.current_device())

        scores_size_tensor = broadcast_from_last_pipeline_stage(1, torch.int64, scores_size_tensor)
        tokens_size_tensor = broadcast_from_last_pipeline_stage(2, torch.int64, tokens_size_tensor)

        scores = broadcast_from_last_to_first_pipeline_stage(tuple(scores_size_tensor), torch.float32, scores)
        tokens = broadcast_from_last_to_first_pipeline_stage(tuple(tokens_size_tensor), torch.int64, tokens)

    return tokens, scores


def generate_with_pipelined_early_exit_and_return_on_first_stage(
        model, tokens, lengths,
        return_output_log_probs=False,
        top_k=0, top_p=0.0, top_p_decay=0.0, top_p_bound=0.0,
        temperature=1.0,
        use_stop_tokens_for_early_termination=True,
        stop_on_double_eol=False,
        stop_on_eol=False,
        stop_tokens=None,
        prevent_newline_after_colon=True,
        echo_prompts=False,
        early_exit_thres=1.0,
        use_early_exit=False,
        print_max_prob=False,
        exit_layers=[]
):
    """Main token generation function.
    Arguments:
        model: no interleaving is supported.
        tokens: prompt tokens extended to be of size [b, max-sequence-length]
        lengths: original prompt length, size: [b]
        return_output_log_probs: flag to calculate the log probability of
            the generated tokens. Note that the log probability is the one
            from the original logit.
        top_k, top_p: top-k and top-p sampling parameters.
            Note that top-k = 1 is gready. Also, these paramters are
            exclusive meaning that:
                if top-k > 0 then we expect top-p=0.
                if top-p > 0 then we check for top-k=0.
        temperature: sampling temperature.
        use_eod_token_for_early_termination: if True, do early termination if
            all the sequences have reached this token.
        prevent_newline_after_colon: if True, it will disable generating new line \n after :
    Note: Outside of model, other parameters only need to be available on
          rank 0.
    Outputs: Note that is size is adjusted to a lower value than
             max-sequence-length if generation is terminated early.
        tokens: prompt and generated tokens. size: [b, :]
        generated_sequence_lengths: total length (including prompt) of
            the generated sequence. size: [b]
        output_log_probs: log probability of the selected tokens. size: [b, s]
    """

    args = get_args()
    tokenizer = get_tokenizer()

    batch_size = tokens.size(0)
    min_prompt_length = lengths.min().item()
    max_sequence_length = tokens.size(1)

    if max_sequence_length > args.max_position_embeddings:
        raise ValueError(f"Length of prompt + tokens_to_generate ({max_sequence_length}) longer than allowed ({args.max_position_embeddings})")
    
    if max_sequence_length * batch_size > args.max_tokens_to_oom:
        raise ValueError("Too many tokens.  " + str(max_sequence_length*batch_size)+ " is greater than "+str(args.max_tokens_to_oom))

    inference_params = InferenceParams(batch_size, max_sequence_length,
                                       top_k=top_k, top_p=top_p,
                                       temperature=temperature,
                                       top_p_bound=top_p_bound,
                                       top_p_decay=top_p_decay,
                                       early_exit_thres=early_exit_thres,
                                       use_early_exit=use_early_exit,
                                       print_max_prob=print_max_prob,
                                       exit_layers=exit_layers)

    # forward step.
    forward_step = ForwardStep(model, inference_params=inference_params)

    # Added termination_id to support the case that we want to terminate the
    # generation once that id is generated.
    if hasattr(args, 'eos_id'):
        termination_id = args.eos_id
    else:
        termination_id = tokenizer.eod

    # ===================
    # Pre-allocate memory
    # ===================

    # Log probability of the sequence (prompt + generated tokens).
    output_log_probs = None
    output_log_probs_size = (batch_size, max_sequence_length - 1)
    # Lengths of generated seuquence including including prompts.
    generated_sequence_lengths = None
    if mpu.is_pipeline_first_stage() or mpu.is_pipeline_last_stage() or mpu.has_early_exit():
        output_log_probs = torch.empty(output_log_probs_size,
                                        dtype=torch.float32,
                                        device=torch.cuda.current_device())
    if mpu.is_pipeline_first_stage():
        generated_sequence_lengths = torch.ones(
                batch_size, dtype=torch.int64,
                device=torch.cuda.current_device()) * max_sequence_length
    
    # Whether we have reached a termination id.
    is_generation_done = torch.zeros(batch_size, dtype=torch.uint8,
                                     device=torch.cuda.current_device())

    # =============
    # Run infernece
    # =============


    with torch.no_grad():
        attention_mask, position_ids = _build_attention_mask_and_position_ids(
            tokens)
        prev_context_length = 0
        for context_length in range(min_prompt_length, max_sequence_length):

            # Pick the slice that we need to pass through the network.
            tokens2use = tokens[:, prev_context_length:context_length]
            positions2use = position_ids[:, prev_context_length:context_length]
            attention_mask2use = attention_mask[
                ..., prev_context_length:context_length, :context_length]

            # clear inference states
            inference_params.clear_early_exit_states()

            # logits will be meanigful only in the last pipeline stage.
            logits = forward_step(tokens2use, positions2use, attention_mask2use)

            if mpu.is_pipeline_last_stage() and not (inference_params.has_early_exited or inference_params.prev_has_early_exited):
                last_token_logits = logits[:, -1, :]

                # Calculate the log probabilities.
                log_probs = F.log_softmax(logits, dim=2)
                max_log_prob, token_id =  torch.max(log_probs[:, -1, :], dim=1)
                token = tokenizer.detokenize([int(token_id[-1])])
                if print_max_prob:
                    print(f"layer final: token [{token}], prob {float(torch.exp(max_log_prob[-1]))}")
                inference_params.has_early_exited = max_log_prob[-1] >= inference_params.early_exit_thres
                # Printout the max prob
                print("!!!!!!!")
                print(inference_params.has_early_exited)
                print("!!!!!!!")
                if inference_params.has_early_exited:
                    print("-----------------")
                    print(f"max log prob: {max_log_prob[-1]}, early exit thres: {inference_params.early_exit_thres}")
                    print("-----------------")
                new_sample = sample(last_token_logits,
                                    top_k=top_k,
                                    top_p=top_p,
                                    temperature=temperature,
                                    vocab_size=tokenizer.vocab_size)
                if top_p > 0.0 and top_p_decay > 0.0:
                    top_p = top_p * top_p_decay
                    if top_p_bound > 0.0:
                        top_p = max(top_p, top_p_bound)

                # If a prompt length is smaller or equal th current context
                # length, it means we have started generating tokens
                started = lengths <= context_length
                # Update the tokens.
                tokens[started, context_length] = new_sample[started]
                # Pick the tokens that we need to get the log
                # probabilities for. Note that next input token is
                # the token which we selected in the current logits,
                # so shift by 1.
                indices = torch.unsqueeze(
                    tokens[
                        :,
                        (prev_context_length + 1):(context_length + 1)],
                    2)
                output_log_probs[:,
                                    prev_context_length:context_length] = \
                    torch.gather(log_probs, 2, indices).squeeze(2)
                send_token_and_probs_to_first_pipeline_stage(inference_params=inference_params,
                                                             token_tensor=tokens[:, context_length],
                                                             prob_tensor=output_log_probs[:, context_length - 1],
                                                             is_final=True)
            elif mpu.is_pipeline_first_stage():
                recv_token_and_probs(inference_params=inference_params, 
                                     token_tensor_buffer=tokens[:, context_length],
                                     prob_tensor_buffer=output_log_probs[:, context_length - 1])
            elif mpu.has_early_exit() and not(inference_params.has_early_exited or inference_params.prev_has_early_exited):
                send_token_and_probs_to_first_pipeline_stage(inference_params=inference_params)

            # Update the context length for the next token generation.
            prev_context_length = context_length
            inference_params.is_first_step = False

            # Check if all the sequences have hit the termination_id.
            # done = None
            # if mpu.is_pipeline_first_stage():
            #     # TODO(rprenger) These stopping methods are tokenizer dependent
            #     # instead tokenization should be in the inference loop so stop sequences can be used
            #     if stop_on_double_eol:
            #         hit_double_eol = (new_sample == 628).byte() & started.byte()
            #         hit_two_eols = (new_sample == 198).byte() & (tokens[:, context_length-1] == 198).byte() & started.byte()
            #         done_token = hit_double_eol | hit_two_eols
            #     elif stop_on_eol:
            #         hit_double_eol = (new_sample == 628).byte() & started.byte()
            #         hit_eol = (new_sample == 198).byte() & started.byte()
            #         done_token = hit_double_eol | hit_eol
            #     else: 
            #         done_token = (new_sample == termination_id).byte() & \
            #             started.byte()

            #     just_finished = (done_token & ~is_generation_done).bool()
            #     generated_sequence_lengths[just_finished.view(-1)] = \
            #         context_length + 1
            #     is_generation_done = is_generation_done | done_token
            #     done = torch.all(is_generation_done)
            # done = broadcast_from_first_pipeline_stage(1, torch.uint8,
            #                                           tensor=done)
            # if use_stop_tokens_for_early_termination and done:
            #     break

    # ===================================================
    # Update the length of based on max generated length.
    # ===================================================

    # tokens = tokens[:, :(context_length + 1)]
    # if mpu.is_pipeline_last_stage():
    #     if return_output_log_probs:
    #         output_log_probs = output_log_probs[:, :context_length]

    # ======================================
    # Broadcast to the first pipeline stage.
    # ======================================

    # if return_output_log_probs:
    #     output_log_probs_size = (batch_size, context_length)
    #     output_log_probs = broadcast_from_last_to_first_pipeline_stage(
    #         output_log_probs_size, torch.float32, output_log_probs)
    if not echo_prompts and mpu.is_pipeline_first_stage():
        generated_sequence_lengths -= lengths
        for i, (sequence, length) in enumerate(zip(tokens, lengths)):
            tokens[i] = sequence.roll(-length.item(), dims=0)
        if return_output_log_probs:
            for i, (prob, length) in enumerate(zip(output_log_probs, lengths)):
                output_log_probs[i] = prob.roll(-(length.item() - 1), dims=0)
    return tokens, generated_sequence_lengths, output_log_probs, None


def _build_attention_mask_and_position_ids(tokens):
    """Build the attention mask and postition ids for the input tokens."""

    # Since we are not interested in loss-mask and reset attention/position
    # is also False, eod_token is not used so it is safe to set it to None.
    attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        data=tokens,
        eod_token=None,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False)

    return attention_mask, position_ids
