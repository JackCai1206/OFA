# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

import string
import math
import json
from itertools import chain
import os
import numpy as np

import torch
import torch.distributed as dist
from fairseq import utils
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

from data import data_utils
from tasks.nlg_tasks.gigaword import fix_tokenization
import editdistance


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.bos, generator.eos}


def decode_fn(x, tgt_dict, bpe, generator, tokenizer=None):
    x = tgt_dict.string(x.int().cpu(), extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator))
    if bpe is not None:
        x = bpe.decode(x)
    if tokenizer is not None:
        x = tokenizer.decode(x)
    return x

def strip_pad(tensor, pad):
    return tensor[tensor.ne(pad)]

def _calculate_error_rate(hyps, refs):
    """each line is "<text> (None-<index>)" """
    assert (len(hyps) == len(refs))
    err_rates = [
        (editdistance.eval(hyp.split(), ref.split()), len(ref.split())) for hyp, ref in zip(hyps, refs)
    ]
    return err_rates


def eval_caption(task, generator, models, sample, **kwargs):
    transtab = str.maketrans({key: None for key in string.punctuation})
    hypos = task.inference_step(generator, models, sample)
    results = []
    for i, sample_id in enumerate(sample["id"].tolist()):
        detok_hypo_str = decode_fn(hypos[i][0]["tokens"], task.tgt_dict, task.bpe, generator)
        results.append({"image_id": str(sample_id), "caption": detok_hypo_str.translate(transtab).strip()})
    return results, None


def eval_caption_cn(task, generator, models, sample, **kwargs):
    hypos = task.inference_step(generator, models, sample)
    results = []
    for i, sample_id in enumerate(sample["id"].tolist()):
        detok_hypo_str = decode_fn(
            hypos[i][0]["tokens"], task.tgt_dict, task.bpe, generator
        )
        results.append(
            {
                "image_id": str(sample_id),
                "caption": detok_hypo_str.strip(),
            }
        )
    return results, None


def eval_ocr(task, generator, models, sample, **kwargs):
    gen_out = task.inference_step(generator, models, sample)
    hyps, refs, results = [], [], []
    for i, sample_id in enumerate(sample["id"].tolist()):
        decode_tokens = decode_fn(gen_out[i][0]["tokens"], task.tgt_dict, task.bpe, generator).strip()
        hyps.append(decode_tokens.strip().replace(" ", ""))
        if sample["target"]:
            refs.append(
                decode_fn(
                    utils.strip_pad(sample["target"][i], task.tgt_dict.pad()),
                    task.tgt_dict, task.bpe, generator
                )
                .strip()
                .replace(" ", "")
            )
        results.append(
            {
                "image_id": str(sample_id),
                "ocr": decode_tokens.strip().replace(" ", ""),
            }
        )
    if refs:
        acc = [1.0 if hyp == ref else 0.0 for hyp, ref in zip(hyps, refs)]
    else:
        acc = None

    return results, acc


def eval_vqa_gen(task, generator, models, sample, **kwargs):
    if kwargs['beam_search_vqa_eval']:
        hypos = task.inference_step(generator, models, sample, prefix_tokens=sample['prefix_tokens'])
        results = []
        for i, sample_id in enumerate(sample["id"].tolist()):
            prefix_len = sample['prefix_tokens'][i].ne(1).sum().item()
            detok_hypo_str = decode_fn(hypos[i][0]["tokens"][prefix_len:], task.tgt_dict, task.bpe, generator)
            results.append({"question_id": int(sample_id), "answer": detok_hypo_str.strip()})
        scores = [ref_dict.get(result['answer'], 0) for ref_dict, result in zip(sample['ref_dict'], results)]
        return results, scores

    encoder_out = models[0].encoder(
        sample["net_input"]["src_tokens"],
        src_lengths=sample["net_input"]["src_lengths"],
        patch_images=sample["net_input"]["patch_images"],
        patch_masks=sample["net_input"]["patch_masks"]
    )
    device = sample["net_input"]["src_tokens"].device
    eos_item = torch.tensor([task.src_dict.eos()])
    pad = task.src_dict.pad()
    valid_result = []
    for valid_answers, valid_constraint_masks in zip(task.valid_answers_list, task.valid_constraint_masks_list):
        valid_size = len(valid_answers)
        valid_tgt_items = [
            torch.cat([torch.tensor(decoder_prompt[1:]), valid_answer, eos_item])
            for decoder_prompt in sample["decoder_prompts"] for valid_answer in valid_answers
        ]
        valid_prev_items = [
            torch.cat([torch.tensor(decoder_prompt), valid_answer])
            for decoder_prompt in sample["decoder_prompts"] for valid_answer in valid_answers
        ]
        valid_constraint_mask_items = [
            torch.cat(
                [torch.zeros(len(decoder_prompt) - 1, valid_constraint_mask.size(1)).bool(), valid_constraint_mask],
                dim=0
            )
            for decoder_prompt in sample["decoder_prompts"] for valid_constraint_mask in valid_constraint_masks
        ]
        valid_tgt = data_utils.collate_tokens(valid_tgt_items, pad_idx=pad).to(device)
        valid_prev_output = data_utils.collate_tokens(valid_prev_items, pad_idx=pad).to(device)
        valid_constraint_masks = data_utils.collate_tokens(valid_constraint_mask_items, pad_idx=pad).to(device)

        new_encoder_out = {}
        new_encoder_out["encoder_out"] = [
            encoder_out["encoder_out"][0].repeat_interleave(valid_size, dim=1)
        ]
        new_encoder_out["encoder_padding_mask"] = [
            encoder_out["encoder_padding_mask"][0].repeat_interleave(valid_size, dim=0)
        ]
        new_encoder_out["position_embeddings"] = [
            encoder_out["position_embeddings"][0].repeat_interleave(valid_size, dim=0)
        ]

        decoder_out = models[0].decoder(valid_prev_output, encoder_out=new_encoder_out)
        decoder_out[0].masked_fill_(~valid_constraint_masks, -math.inf)
        lprobs = models[0].get_normalized_probs(decoder_out, log_probs=True)
        scores = lprobs.gather(dim=-1, index=valid_tgt.unsqueeze(-1)).squeeze(-1)
        scores = scores.masked_fill(valid_tgt.eq(task.tgt_dict.pad()), 0)
        scores = scores.masked_fill((~valid_constraint_masks).all(2), 0)
        scores = scores.sum(1)
        scores = scores.view(-1, valid_size)
        valid_result.append(scores)
    valid_result = torch.cat(valid_result, dim=-1)
    predicts = valid_result.argmax(1).tolist()
    hyps = [task.index2ans[predict_index] for predict_index in predicts]
    results = [{"question_id": int(id), "answer": hyp} for id, hyp in zip(sample["id"].tolist(), hyps)]
    scores = [ref_dict.get(hyp, 0) for ref_dict, hyp in zip(sample['ref_dict'], hyps)]
    return results, scores


def eval_refcoco(task, generator, models, sample, **kwargs):
    def _calculate_ap_score(hyps, refs, thresh=0.5):
        interacts = torch.cat(
            [torch.where(hyps[:, :2] < refs[:, :2], refs[:, :2], hyps[:, :2]),
             torch.where(hyps[:, 2:] < refs[:, 2:], hyps[:, 2:], refs[:, 2:])],
            dim=1
        )
        area_predictions = (hyps[:, 2] - hyps[:, 0]) * (hyps[:, 3] - hyps[:, 1])
        area_targets = (refs[:, 2] - refs[:, 0]) * (refs[:, 3] - refs[:, 1])
        interacts_w = interacts[:, 2] - interacts[:, 0]
        interacts_h = interacts[:, 3] - interacts[:, 1]
        area_interacts = interacts_w * interacts_h
        ious = area_interacts / (area_predictions + area_targets - area_interacts + 1e-6)
        return ((ious >= thresh) & (interacts_w > 0) & (interacts_h > 0)).float()

    gen_out = task.inference_step(generator, models, sample)
    hyps = []
    for i in range(len(gen_out)):
        hyps.append(gen_out[i][0]["tokens"][:-1] - len(task.src_dict) + task.cfg.num_bins)
    hyps = torch.stack(hyps, dim=0)
    hyps = hyps / (task.cfg.num_bins - 1) * task.cfg.max_image_size
    hyps[:, ::2] /= sample['w_resize_ratios'].unsqueeze(1)
    hyps[:, 1::2] /= sample['h_resize_ratios'].unsqueeze(1)

    results = [
        {"uniq_id": sample_id,
         "box": [hyps[i][0].item(), hyps[i][1].item(), hyps[i][2].item(), hyps[i][3].item()]}
        for i, sample_id in enumerate(sample["id"].tolist())
    ]
    scores = _calculate_ap_score(hyps, sample['region_coords'].float())
    return results, scores


def eval_snli_ve(task, generator, models, sample, **kwargs):
    encoder_out = models[0].encoder(
        sample["net_input"]["src_tokens"],
        src_lengths=sample["net_input"]["src_lengths"],
        patch_images=sample["net_input"]["patch_images"],
        patch_masks=sample["net_input"]["patch_masks"]
    )
    device = sample["net_input"]["src_tokens"].device
    eos_item = torch.tensor([task.src_dict.eos()])
    pad = task.src_dict.pad()
    valid_result = []
    for valid_answers, valid_constraint_masks in zip(task.valid_answers_list, task.valid_constraint_masks_list):
        valid_size = len(valid_answers)
        valid_tgt_items = [
            torch.cat([torch.tensor(decoder_prompt[1:]), valid_answer, eos_item])
            for decoder_prompt in sample["decoder_prompts"] for valid_answer in valid_answers
        ]
        valid_prev_items = [
            torch.cat([torch.tensor(decoder_prompt), valid_answer])
            for decoder_prompt in sample["decoder_prompts"] for valid_answer in valid_answers
        ]
        valid_constraint_mask_items = [
            torch.cat(
                [torch.zeros(len(decoder_prompt) - 1, valid_constraint_mask.size(1)).bool(), valid_constraint_mask],
                dim=0
            )
            for decoder_prompt in sample["decoder_prompts"] for valid_constraint_mask in valid_constraint_masks
        ]
        valid_tgt = data_utils.collate_tokens(valid_tgt_items, pad_idx=pad).to(device)
        valid_prev_output = data_utils.collate_tokens(valid_prev_items, pad_idx=pad).to(device)
        valid_constraint_masks = data_utils.collate_tokens(valid_constraint_mask_items, pad_idx=pad).to(device)

        new_encoder_out = {}
        new_encoder_out["encoder_out"] = [
            encoder_out["encoder_out"][0].repeat_interleave(valid_size, dim=1)
        ]
        new_encoder_out["encoder_padding_mask"] = [
            encoder_out["encoder_padding_mask"][0].repeat_interleave(valid_size, dim=0)
        ]
        new_encoder_out["position_embeddings"] = [
            encoder_out["position_embeddings"][0].repeat_interleave(valid_size, dim=0)
        ]

        decoder_out = models[0].decoder(valid_prev_output, encoder_out=new_encoder_out)
        decoder_out[0].masked_fill_(~valid_constraint_masks, -math.inf)
        lprobs = models[0].get_normalized_probs(decoder_out, log_probs=True)
        scores = lprobs.gather(dim=-1, index=valid_tgt.unsqueeze(-1)).squeeze(-1)
        scores = scores.masked_fill(valid_tgt.eq(task.tgt_dict.pad()), 0)
        scores = scores.masked_fill((~valid_constraint_masks).all(2), 0)
        scores = scores.sum(1)
        scores = scores.view(-1, valid_size)
        valid_result.append(scores)
    valid_result = torch.cat(valid_result, dim=-1)
    predicts = valid_result.argmax(1).tolist()
    hyps = [task.index2ans[predict_index] for predict_index in predicts]
    results = [{"uniq_id": id, "answer": hyp} for id, hyp in zip(sample["id"].tolist(), hyps)]
    scores = [ref_dict.get(hyp, 0) for ref_dict, hyp in zip(sample['ref_dict'], hyps)]
    return results, scores


def eval_image_gen(task, generator, models, sample, **kwargs):
    hypos, _ = task.inference_image(generator, sample, models)
    tokens = sample['net_input']['src_tokens'][0].view(-1).tolist()
    caption = task.bpe.decode(task.tgt_dict.string([token for token in tokens if token >= 4]))[
              38:].replace('/', '')

    text_similarity_score, indices = task.compute_text_similarity(hypos, caption,
                                                                  sample['net_input']['src_tokens'].device)
    results = []
    for i, indice in enumerate(indices):
        results.append({"sample_id": str(sample["id"][0]), "score": text_similarity_score[i], "image": hypos[indice]})
    scores = [max(text_similarity_score).item()]
    sorted_hyps = [hypos[indice] for indice in indices]
    # dump results
    if task.cfg.gen_images_path:
        caption_tokens = sample['net_input']['src_tokens'][0].view(-1).tolist()
        caption = task.bpe.decode(task.tgt_dict.string([token for token in caption_tokens if token >= 4]))[
                  38:].replace('/', '')
        task.dump_images(sorted_hyps, text=caption, path=os.path.join(task.cfg.gen_images_path, 'all_results'))
        task.dump_images(sorted_hyps, text=caption, path=os.path.join(task.cfg.gen_images_path, 'top1'), topk=1)

    return results, scores


def eval_glue(task, generator, models, sample, **kwargs):
    net_output = models[0](**sample["net_input"])
    net_output[0].masked_fill_(~sample["constraint_masks"], -math.inf)
    last_token_ids = sample["net_input"]["prev_output_tokens"].ne(task.src_dict.pad()).sum(1, keepdim=True) - 1
    logits = net_output[0].gather(1, last_token_ids.unsqueeze(2).expand(-1, -1, net_output[0].size(2)))
    logits = logits.squeeze(1)
    predicts = logits.argmax(1).tolist()
    hyps = [task.bpe.decode(task.src_dict[predict]).strip() for predict in predicts]
    results = [{"hyp": hyp, "ref": ref_dict.keys()[0]} for hyp, ref_dict in zip(hyps, sample['ref_dict'])]
    return results, None


def eval_gigaword(task, generator, models, sample, **kwargs):
    gen_out = task.inference_step(generator, models, sample)
    hyps, refs = [], []
    results = []
    for i in range(len(gen_out)):
        hyp = decode_fn(gen_out[i][0]["tokens"], task.tgt_dict, task.bpe, generator).lower().strip()
        hyp = fix_tokenization(hyp).replace('1', '#')
        ref = sample['target_strs'][i]
        hyps.append(hyp)
        refs.append(ref)
        results.append({"hyp": hyp, "ref": ref})
    return results, None


def eval_image_classify(task, generator, models, sample, **kwargs):
    batch_size = sample["net_input"]["src_tokens"].size(0)
    encoder_out = models[0].encoder(
        sample["net_input"]["src_tokens"],
        src_lengths=sample["net_input"]["src_lengths"],
        patch_images=sample["net_input"]["patch_images"],
        patch_masks=sample["net_input"]["patch_masks"]
    )
    device = sample["net_input"]["src_tokens"].device
    valid_result = []
    for valid_tgt, valid_prev_output, valid_constraint_masks in zip(task.valid_tgt_list,
                                                                    task.valid_prev_output_list,
                                                                    task.valid_constraint_masks_list):
        valid_tgt_size = valid_tgt.size(0)
        valid_tgt = valid_tgt.repeat(batch_size, 1).to(device)
        valid_prev_output = valid_prev_output.repeat(batch_size, 1).to(device)
        valid_constraint_masks = valid_constraint_masks.repeat(batch_size, 1, 1).to(device)
        new_encoder_out = {}
        new_encoder_out["encoder_out"] = [
            encoder_out["encoder_out"][0].repeat_interleave(valid_tgt_size, dim=1)
        ]
        new_encoder_out["encoder_padding_mask"] = [
            encoder_out["encoder_padding_mask"][0].repeat_interleave(valid_tgt_size, dim=0)
        ]
        new_encoder_out["position_embeddings"] = [
            encoder_out["position_embeddings"][0].repeat_interleave(valid_tgt_size, dim=0)
        ]

        decoder_out = models[0].decoder(valid_prev_output, encoder_out=new_encoder_out)
        decoder_out[0].masked_fill_(~valid_constraint_masks, -math.inf)
        lprobs = models[0].get_normalized_probs(decoder_out, log_probs=True)
        scores = lprobs.gather(dim=-1, index=valid_tgt.unsqueeze(-1)).squeeze(-1)
        scores = scores.masked_fill(valid_tgt.eq(task.tgt_dict.pad()), 0)
        scores = scores.sum(1)
        scores = scores.view(-1, valid_tgt_size)
        valid_result.append(scores)
    valid_result = torch.cat(valid_result, dim=-1)
    predicts = valid_result.argmax(1).tolist()
    hyps = [task.index2ans[predict_index] for predict_index in predicts]
    scores = [ref_dict.get(hyp, 0) for ref_dict, hyp in zip(sample['ref_dict'], hyps)]
    results = [{"uniq_id": id, "answer": hyp} for id, hyp in zip(sample["id"].tolist(), hyps)]
    return results, scores

def eval_asr(task, generator, models, sample, **kwargs):
    transtab = str.maketrans({key: None for key in string.punctuation})
    gen_out = task.inference_step(generator, models, sample)

    hyps, refs, results = [], [], []

    for i, sample_id in enumerate(sample["id"].tolist()):
        decode_tokens = decode_fn(gen_out[i][0]["tokens"], task.tgt_dict, task.bpe, generator)
        hyps.append(decode_tokens.translate(transtab).strip())
        results.append({"speech_id": str(sample_id), "transcript": decode_tokens.strip().translate(transtab)})

        decode_target = decode_fn(strip_pad(sample["target"][i], task.tgt_dict.pad()), task.tgt_dict, task.bpe, generator)
        refs.append(decode_target.translate(transtab).strip())

    scores = _calculate_error_rate(hyps, refs)
    return results, scores

def eval_sgcls(task, generator, models, sample, **kwargs):
    def decode(toks):
        s = task.tgt_dict.string(toks.int().cpu())
        return s

    models[0].eval()
    gen_out = task.inference_step(generator, models, sample)
    hyps, refs = [], []
    img_ids = []
    # transtab = str.maketrans({key: None for key in string.punctuation})
    for i in range(len(gen_out)):
        hyps.append(decode(gen_out[i][0]["tokens"]))
        refs.append(decode(utils.strip_pad(sample["target"][i], task.tgt_dict.pad())))
        img_ids.append(sample["id"][i])

    hyp_triplets = [toks2triplets(hyps[i].split(), task) for i in range(len(gen_out))]
    ref_triplets = [toks2triplets(refs[i].split(), task) for i in range(len(gen_out))]

    # print('IMG:', '\"/data/hulab/zcai75/visual_genome/VG_100K/' + img_ids[0] + '.jpg\"')
    # print('HYP:', hyp_triplets[0])
    # print('REF:', ref_triplets[0])
    obj_list = task.vg_json['object_count'].keys()
    pred_list = task.vg_json['predicate_to_idx'].keys()

    results = []
    # calculate triplet recall
    scores = []
    for img_id, hyp, ref in zip(img_ids, hyp_triplets, ref_triplets):
        res = {"img_id": img_id, "hyp": hyp, "ref": ref}
        pred_count = dict(zip(pred_list, [[0, 0] for _ in range(len(pred_list))]))
        match_count, rel_count = calculate_recall(hyp, ref, pred_count) # per image
        scores.append(match_count / rel_count)
        res['pred_count'] = pred_count
        res['match_count'] = match_count
        results.append(res)

    return results, scores

def get_boxes_SAM(image):
    sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(image)
    bbox = [mask['box'] for mask in masks]
    # convert to x1, y1, x2, y2
    bbox = [[box[0], box[1], box[0] + box[2], box[1] + box[3]] for box in bbox]
    return bbox

def eval_vrd(task, generator, models, sample, **kwargs):
    def decode(toks):
        s = task.tgt_dict.string(toks.int().cpu())
        return s

    hyps, refs = [], []
    hyp_triplets, ref_triplets = [], []
    obj_counts = []
    img_ids = []
    dataset = task.datasets['test']
    models[0].eval()
    examples = []
    for i in range(len(sample["id"])):
        image_id = sample["id"][i]
        img_ids.append(image_id)
        boxes = sample['all_boxes'][i]
        # boxes = get_boxes_SAM(sample['net_input']['patch_images'][i])
        last = obj_counts[-1][1] if len(obj_counts) > 0 else 0
        obj_counts.append((last, last + len(boxes)))
        prompt = dataset.encode_text(' describe the relations for this region: ')
        for j, sub_box in enumerate(boxes):
            # if (len(sub_box) == 5):
            #     assert decode(sub_box[4:]) == '<unk>'
            #     sub_box = sub_box[:4]
            # print(decode(box))
            obj_boxes = boxes[:j] + boxes[j+1:]
            sub_box_str = '<sub> ' + sub_box + ' ' + ' '.join(['<obj> ' + ob for ob in obj_boxes])
            # print(sub_box_str)
            sub_box_str = dataset.encode_text(sub_box_str, use_bpe=False)
            src_item = torch.cat([dataset.bos_item.cuda(), prompt.cuda(), sub_box_str.cuda(), dataset.eos_item.cuda()])
            example = {
                "id": image_id + '-' + str(j),
                "source": src_item,
                "patch_image": sample['net_input']['patch_images'][i],
                "patch_mask": sample['net_input']['patch_masks'][i:i+1],
                "target": torch.Tensor([]), # not used
                "prev_output_tokens": torch.Tensor([]) # not used
            }
            examples.append(example)
        
    examples_extra = None
    if len(examples) > 60: 
        examples_extra = examples[60:]
        examples = examples[:60]

    batched_examples = dataset.collater(examples)
    gen_out = task.inference_step(generator, models, batched_examples)
    if examples_extra is not None:
        batched_examples_extra = dataset.collater(examples_extra)
        gen_out_extra = task.inference_step(generator, models, batched_examples_extra)
        gen_out.extend(gen_out_extra)
    for i, (first, last) in enumerate(obj_counts):
        hyp_trip = []
        hyp = ""
        for out in gen_out[first:last]:
            sent = decode(out[0]["tokens"])
            hyp += sent + ' '
            trips = toks2triplets(sent.split(), task)
            hyp_trip.extend(trips)
        # print(hyp_trip)
        hyp_triplets.append(hyp_trip)
        hyps.append(hyp)
        refs.append(decode(utils.strip_pad(sample["target"][i], task.tgt_dict.pad())))
        ref_triplets.append(toks2triplets(refs[i].split(), task))

    obj_list = task.vg_json['object_count'].keys()
    pred_list = task.vg_json['predicate_to_idx'].keys()

    results = []
    # calculate triplet recall
    scores = []
    for img_id, hyp, ref in zip(img_ids, hyp_triplets, ref_triplets):
        res = {"img_id": img_id, "hyp": hyp, "ref": ref}
        pred_count = dict(zip(pred_list, [[0, 0] for _ in range(len(pred_list))]))
        match_count, rel_count = calculate_recall(hyp, ref, pred_count) # per image
        scores.append(match_count / rel_count)
        res['pred_count'] = pred_count
        res['match_count'] = match_count
        results.append(res)

    return results, scores

def eval_vrd2(task, generator, models, sample, **kwargs):
    def decode(toks):
        s = task.tgt_dict.string(toks.int().cpu())
        return s

    hyps, refs = [], []
    hyp_triplets, ref_triplets = [], []
    rel_counts = []
    img_ids = []
    dataset = task.datasets['test']
    models[0].eval()
    examples = []
    for i in range(len(sample["id"])):
        image_id = sample["id"][i]
        img_ids.append(image_id)
        boxes = sample['all_boxes'][i]
        boxes_raw = sample['all_boxes_raw'][i]
        # boxes = get_boxes_SAM(sample['net_input']['patch_images'][i])
        box_dist = get_box_dist(boxes_raw)
        last = rel_counts[-1][1] if len(rel_counts) > 0 else 0
        rel_count = 0
        prompt = dataset.encode_text(' describe the relations: ')
        for j, sub_box in enumerate(boxes):
            # if (len(sub_box) == 5):
            #     assert decode(sub_box[4:]) == '<unk>'
            #     sub_box = sub_box[:4]
            # print(decode(box))
            obj_boxes = boxes[:j] + boxes[j+1:]
            for k, obj_box in enumerate(obj_boxes):
                if box_dist[j][k] < 500:
                    sub_box_str = '<sub> ' + sub_box + ' <obj> ' + obj_box
                    # print(sub_box_str)
                    sub_box_str = dataset.encode_text(sub_box_str, use_bpe=False)
                    src_item = torch.cat([dataset.bos_item.cuda(), prompt.cuda(), sub_box_str.cuda(), dataset.eos_item.cuda()])
                    example = {
                        "id": image_id + '-' + str(j),
                        "source": src_item,
                        "patch_image": sample['net_input']['patch_images'][i],
                        "patch_mask": sample['net_input']['patch_masks'][i:i+1],
                        "target": torch.Tensor([]), # not used
                        "prev_output_tokens": torch.Tensor([]) # not used
                    }
                    examples.append(example)
                rel_count += 1
        rel_counts.append((last, last + rel_count))
    gen_out = []
    for i in range(0, len(examples), 60):
        batched_examples = dataset.collater(examples[i:i+60])
        gen_out.extend(task.inference_step(generator, models, batched_examples))
    for i, (first, last) in enumerate(rel_counts):
        hyp_trip = []
        hyp = ""
        for out in gen_out[first:last]:
            sent = decode(out[0]["tokens"])
            hyp += sent + ' '
            trips = toks2triplets(sent.split(), task, reverse_sub_obj=True)
            hyp_trip.extend(trips)
        # print(hyp_trip)
        hyp_triplets.append(hyp_trip)
        hyps.append(hyp)
        refs.append(decode(utils.strip_pad(sample["target"][i], task.tgt_dict.pad())))
        ref_triplets.append(toks2triplets(refs[i].split(), task))
    
    # print(hyp_triplets[0])
    # print(ref_triplets[0])

    obj_list = task.vg_json['object_count'].keys()
    pred_list = task.vg_json['predicate_to_idx'].keys()

    results = []
    # calculate triplet recall
    scores = []
    for img_id, hyp, ref in zip(img_ids, hyp_triplets, ref_triplets):
        res = {"img_id": img_id, "hyp": hyp, "ref": ref}
        pred_count = dict(zip(pred_list, [[0, 0] for _ in range(len(pred_list))]))
        match_count, rel_count = calculate_recall(hyp, ref, pred_count) # per image
        scores.append(match_count / rel_count)
        res['pred_count'] = pred_count
        res['match_count'] = match_count
        results.append(res)

    return results, scores

def get_box_dist(boxes):
    # use numpy to speed up
    boxes = np.array(boxes)
    box_dist = np.zeros((len(boxes), len(boxes)))
    # broadcast
    box_dist = np.sqrt(np.sum((boxes[:, None, :2] - boxes[None, :, :2]) ** 2, axis=-1))
    return box_dist

def calculate_recall(hyp, ref, pred_count):
    matches = []
    for r in ref:
        for h in hyp:
            if len(h) != 3 or len(r) != 3:
                print(h, r)
                continue
            if h[0] == r[0] and h[1] == r[1] and h[2] == r[2]:
                matches.append(h)
                break
    for match in matches:
        pred_count[match[1]][0] += 1 if match[1] in pred_count else 0
    for r in ref:
        if r[1] in pred_count:
            pred_count[r[1]][1] += 1
        if r[1] not in pred_count:
            print(r)
    # for p in pred_count:
    #     assert pred_count[p][0] <= pred_count[p][1], pred_count
    return len(matches), len(ref)

def toks2triplets(toks, task, reverse_pred_obj=False, reverse_sub_obj=False):
    triplets = []
    curr_sub = ''
    curr_obj = ''
    curr_pred = ''
    state = None
    toks.append('<sub>')
    for i, tok in enumerate(toks):
        # print(tok, '|'.join([curr_sub, curr_obj, curr_pred]), triplets)
        if tok == '<sub>':
            state = 'sub'
            curr_sub = ''
            if len(curr_obj) > 0 and len(triplets[-1]) == 2:
                triplets[-1].append(task.bpe.decode(curr_obj).strip())

        elif tok == '<pred>':
            state = 'pred'
            if len(curr_obj) > 0 and len(triplets[-1]) == 2: 
                triplets[-1].append(task.bpe.decode(curr_obj).strip())
            triplets.append([task.bpe.decode(curr_sub).strip()])
            curr_pred = ''
        elif tok == '<obj>':
            state = 'obj'
            triplets[-1].append(task.bpe.decode(curr_pred).strip())
            curr_obj = ''
            
        else:
            if state == 'sub':
                curr_sub += ' ' + tok
            elif state == 'obj':
                curr_obj += ' ' + tok
            elif state == 'pred':
                curr_pred += ' ' + tok

    if len(triplets[-1]) < 3:
        del triplets[-1]
    for trip in triplets:
        if len(trip) < 3:
            print(trip)
            continue
        if len(trip) > 3:
            print(trip)
            trip = trip[:3]
        if reverse_pred_obj:
            trip[1], trip[2] = trip[2], trip[1]
        if reverse_sub_obj:
            trip[0], trip[2] = trip[2], trip[0]
        if '<rare>' in trip[1]:
            trip[1] = trip[1].replace('<rare>', '').strip()
        if '<bin' in trip[0]:
            trip[0] = trip[0].split('<bin')[0].strip()
        if '<bin' in trip[2]:
            trip[2] = trip[2].split('<bin')[0].strip()

    return triplets

def eval_step(task, generator, models, sample, **kwargs):
    if task.cfg._name == 'caption':
        return eval_caption(task, generator, models, sample, **kwargs)
    elif task.cfg._name == "caption_cn":
        return eval_caption_cn(task, generator, models, sample, **kwargs)
    elif task.cfg._name == "ocr":
        return eval_ocr(task, generator, models, sample, **kwargs)
    elif task.cfg._name == 'vqa_gen':
        return eval_vqa_gen(task, generator, models, sample, **kwargs)
    elif task.cfg._name == 'refcoco':
        return eval_refcoco(task, generator, models, sample, **kwargs)
    elif task.cfg._name == 'snli_ve':
        return eval_snli_ve(task, generator, models, sample, **kwargs)
    elif task.cfg._name == 'image_gen':
        return eval_image_gen(task, generator, models, sample, **kwargs)
    elif task.cfg._name in {'cola', 'mnli', 'mrpc', 'qnli', 'qqp', 'rte', 'sst2'}:
        return eval_glue(task, generator, models, sample, **kwargs)
    elif task.cfg._name == 'gigaword':
        return eval_gigaword(task, generator, models, sample, **kwargs)
    elif task.cfg._name == 'image_classify':
        return eval_image_classify(task, generator, models, sample, **kwargs)
    elif task.cfg._name == 'unify_speech_text_task' or task.cfg._name == 'speech_unify_cn_big_fbank':
        return eval_asr(task, generator, models, sample, **kwargs)
    elif task.cfg._name == 'sgcls':
        return eval_sgcls(task, generator, models, sample, **kwargs)
    elif task.cfg._name == 'vrd':
        return eval_vrd(task, generator, models, sample, **kwargs)
    elif task.cfg._name == 'vrd2':
        return eval_vrd2(task, generator, models, sample, **kwargs)
    else:
        raise NotImplementedError


def merge_results(task, cfg, logger, score_cnt, score_sum, results):
    if task.cfg._name == 'image_gen':
        if cfg.distributed_training.distributed_world_size > 1:
            dist.all_reduce(score_sum.data)
            dist.all_reduce(score_cnt.data)
        if score_cnt.item() > 0:
            logger.info("score_sum: {}, score_cnt: {}, score: {}".format(
                score_sum, score_cnt, round(score_sum.item() / score_cnt.item(), 4)
            ))
    elif task.cfg._name == 'sgcls' or task.cfg._name == 'vrd' or task.cfg._name == 'vrd2':
        gather_results = None
        if cfg.distributed_training.distributed_world_size > 1:
            gather_results = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gather_results, results)
            dist.all_reduce(score_sum.data)
            dist.all_reduce(score_cnt.data)

        gather_results = list(chain(*gather_results)) if gather_results is not None else results
        match_count = 0
        pred_list = task.vg_json['predicate_to_idx'].keys()
        pred_count = dict(zip(pred_list, [[0, 0] for _ in range(len(pred_list))]))
        total_ref = 0
        total_hyp = 0
        for res in gather_results:
            match_count += res['match_count']
            for pred in res['pred_count']:
                pred_count[pred][0] += res['pred_count'][pred][0]
                pred_count[pred][1] += res['pred_count'][pred][1]
            total_ref += len(res['ref'])
            total_hyp += len(res['hyp'])
        
        # print(pred_count)
        mean_recall = sum([pred_count[pred][0] / pred_count[pred][1] for pred in pred_count if pred_count[pred][1] != 0]) / len(pred_count)
        if score_cnt.item() > 0:
            logger.info("recall_by_image: {} / {} = {}, recall: {} / {} = {}, mean recall: {}, mean hyp n_rel: {}, mean ref n_rel {}".format(
                round(score_sum.item(), 4), score_cnt.item(), round(score_sum.item() / score_cnt.item(), 4),
                match_count, total_ref, round(match_count / total_ref, 4),
                round(mean_recall, 4), round( total_hyp / score_cnt.item(), 4), round(total_ref / score_cnt.item(), 4)
            ))
        
        if cfg.distributed_training.distributed_world_size == 1 or dist.get_rank() == 0:
            os.makedirs(cfg.common_eval.results_path, exist_ok=True)
            output_path = os.path.join(cfg.common_eval.results_path, "{}_predict_{}.json".format(cfg.dataset.gen_subset, mean_recall))
            with open(output_path, 'w') as fw:
                # for res in gather_results:
                #     del res['pred_count']
                #     del res['match_count']
                json.dump(gather_results, fw)

    else:
        gather_results = None
        if cfg.distributed_training.distributed_world_size > 1:
            gather_results = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gather_results, results)
            dist.all_reduce(score_sum.data)
            dist.all_reduce(score_cnt.data)
        if score_cnt.item() > 0:
            logger.info("score_sum: {}, score_cnt: {}, score: {}".format(
                score_sum, score_cnt, round(score_sum.item() / score_cnt.item(), 4)
            ))

        if cfg.distributed_training.distributed_world_size == 1 or dist.get_rank() == 0:
            os.makedirs(cfg.common_eval.results_path, exist_ok=True)
            output_path = os.path.join(cfg.common_eval.results_path, "{}_predict.json".format(cfg.dataset.gen_subset))
            gather_results = list(chain(*gather_results)) if gather_results is not None else results
            with open(output_path, 'w') as fw:
                json.dump(gather_results, fw)
