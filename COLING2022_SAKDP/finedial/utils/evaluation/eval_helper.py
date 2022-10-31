# This code is forked/revised from Tensorflow-NMT

"""Utility for evaluating various tasks, e.g., translation & summarization."""
import argparse
import os
import subprocess
import sys
import math
import re
import time
from multiprocessing import Pool

import numpy as np

from finedial.utils.evaluation.score_helper import ScoreManager
from finedial.utils.evaluation.scripts import bleu
from finedial.utils.evaluation.scripts import rouge
from finedial.utils.evaluation.scripts import embed
from finedial.utils.evaluation.scripts import tokens2wordlevel
from finedial.utils.logging.logger_helper import logger

from rouge import Rouge

from collections import Counter, defaultdict

__all__ = ["evaluate"]


def evaluate(metric, ground_truth_reference, ground_truth_query, generations, embed_file=0,
             dim=200, vocab_size=None, word_option=None, beam_width=10):
    """
    :param metric:
    :param ground_truth_reference: Ground-truth reference tgt file
    :param ground_truth_query: Ground-truth reference src file
    :param generations: : The generated tgt file
    :param embed_file:
    :param dim:
    :param vocab_size:
    :param word_option:
    :param beam_width:
    :return:
    """
    if metric.lower() == "embed":
        evaluation_score = embed.eval(ground_truth_query, ground_truth_reference, generations, embed_file, dim,
                                      word_option)
    elif metric.lower() == "f1":
        evaluation_score = f1(generations, ground_truth_reference, word_option=word_option)
    elif metric.lower() == "unk":
        evaluation_score = unk(generations, -1, word_option=word_option)
    elif metric.lower() == 'unk_sentence':
        evaluation_score = unk_sentence(generations, -1, word_option=word_option)
    elif len(metric.lower()) > 4 and metric.lower()[0:4] == 'bleu':
        max_order = int(metric.lower()[5:])
        evaluation_score = _bleu(ground_truth_reference, generations, max_order=max_order,
                                 word_option=word_option)
        reverse_evaluation_score = _bleu(ground_truth_query, generations, max_order=max_order,
                                         word_option=word_option)
        for key in reverse_evaluation_score.keys():
            evaluation_score['SRC_' + key] = reverse_evaluation_score[key]

    elif metric.lower() == "rouge":
        evaluation_score = _rouge(ground_truth_reference, generations,
                                  word_option=word_option)
    elif metric.lower()[0:len('distinct_c')] == 'distinct_c':
        max_order = int(metric.lower()[len('distinct_c') + 1:])
        evaluation_score = _distinct_c(generations, max_order, word_option=word_option)
    elif metric.lower()[0:len('distinct')] == 'distinct':
        max_order = int(metric.lower()[len('distinct') + 1:])
        evaluation_score = _distinct(generations, max_order, word_option=word_option)
    elif metric.lower()[0:len('intra_distinct')] == 'intra_distinct':
        max_order = int(metric.lower()[len('intra_distinct') + 1:])
        evaluation_score = _intra_distinct(generations, max_order, beam_width, word_option=word_option)
    elif metric.lower()[0:len('len')] == 'len':
        evaluation_score = seq_len(generations, -1, word_option=word_option)
    elif metric.lower() == 'std_ent':
        evaluation_score = _std_ent_n(generations, word_option=word_option)
    else:
        raise ValueError("Unknown metric %s" % metric)

    return evaluation_score


def _clean(sentence, word_option):
    sentence = tokens2wordlevel.revert_from_sentence(sentence, word_option)
    return sentence


# 验证没问题OK https://github.com/mgalley/DSTC7-End-to-End-Conversation-Modeling/blob/master/evaluation/src/metrics.py
def _distinct(trans_file, max_order=1, word_option=None):
    """Compute Distinct Score"""
    translations = []
    with open(trans_file, 'r', encoding='utf-8') as fh:
        for line in fh.readlines():
            line = _clean(line, word_option=word_option)
            translations.append(line.split(" "))

    num_tokens = 0
    unique_tokens = set()
    scores = []
    for words in translations:
        local_unique_tokens = set()
        local_count = 0.0
        # print(items)
        for i in range(0, len(words) - max_order + 1):
            valid_words = []
            for x in words[i:i + max_order]:
                if x.find('<unk>') == -1:
                    valid_words.append(x)
            tmp = ' '.join(valid_words)
            unique_tokens.add(tmp)
            num_tokens += 1
            local_unique_tokens.add(tmp)
            local_count += 1
        if local_count == 0:
            scores.append(0)
        else:
            scores.append(100 * len(local_unique_tokens) / local_count)
    if num_tokens == 0:
        ratio = 0
    else:
        ratio = len(unique_tokens) / num_tokens

    result_dict = {
        'DIST_%d' % max_order: ratio * 100,
        'DIST_%d_Scores' % max_order: scores,
    }
    return result_dict


def _intra_distinct(trans_file, max_order, beam_width, word_option=None):
    """Compute Distinct Score"""
    if beam_width == 0:
        beam_width = 1
    translations = []
    with open(trans_file, 'r', encoding='utf-8') as fh:
        for line in fh.readlines():
            line = _clean(line, word_option=word_option)
            translations.append(line.split(" "))

    assert len(translations) % beam_width == 0

    all_ratios = []
    for i in range(0, len(translations), beam_width):
        num_tokens = 0
        unique_tokens = set()
        for items in translations[i:i + beam_width]:
            # print(items)
            for i in range(0, len(items) - max_order + 1):
                valid_words = []
                for x in items[i:i + max_order]:
                    if x != '<unk>':
                        valid_words.append(x)
                tmp = ' '.join(valid_words)
                unique_tokens.add(tmp)
                num_tokens += 1
        if num_tokens == 0:
            ratio = 0
        else:
            ratio = len(unique_tokens) / num_tokens
        all_ratios.append(ratio * 100)

    result_dict = {
        'DIST_Intra_%d' % max_order: sum(all_ratios) / len(all_ratios),
        'DIST_Intra_%d_Scores' % max_order: all_ratios,
    }
    return result_dict


def seq_len(trans_file, max_order=1, word_option=None):
    """Compute Length Score"""
    sum = 0.0
    total = 0
    scores = []
    with open(trans_file, 'r', encoding='utf-8') as fh:
        for line in fh.readlines():
            line = _clean(line, word_option=word_option)
            word_len = len(line.split())
            sum += word_len
            scores.append(word_len)
            total += 1
    result_dict = {
        'Len': sum / total,
        'Len_Scores': scores,
    }
    return result_dict


def f1(trans_file, golden_file, word_option=None):
    """
    calc_f1
    """
    responses = []
    with open(trans_file, 'r', encoding='utf-8') as fh:
        for line in fh.readlines():
            line = _clean(line, word_option=word_option)
            responses.append(line.split(" "))
    golden_responses = []
    with open(golden_file, 'r', encoding='utf-8') as fh:
        for line in fh.readlines():
            line = _clean(line, word_option=word_option)
            golden_responses.append(line.split(" "))


    golden_char_total = 0.0
    pred_char_total = 0.0
    hit_char_total = 0.0
    for response, golden_response in zip(responses, golden_responses):
        golden_response = "".join(golden_response)
        response = "".join(response)
        common = Counter(response) & Counter(golden_response)
        hit_char_total += sum(common.values())
        golden_char_total += len(golden_response)
        pred_char_total += len(response)
    p = hit_char_total / pred_char_total
    r = hit_char_total / golden_char_total
    f1 = 2 * p * r / (p + r)
    result_dict = {
        'CharF1': f1 * 100
    }
    return result_dict

def unk(trans_file, max_order=1, word_option=None):
    """Compute UNK Number"""
    unk = 0.0
    total = 0.0
    scores = []
    with open(trans_file, 'r', encoding='utf-8') as fh:
        for line in fh.readlines():
            line = _clean(line, word_option=word_option).split()
            total += len(line)
            my_unk = 0.0
            for word in line:
                if word.find('<unk>') > -1:
                    unk += 1
                    my_unk += 1
            if len(line) == 0:
                scores.append(1.0)
            else:
                scores.append((my_unk / len(line)))

    result_dict = {
        'UNKRate': unk / total,
        'UNKRate_Scores': scores,
    }
    return result_dict


def unk_sentence(trans_file, max_order=1, word_option=None):
    """Compute UNK Number"""
    unk = 0.0
    total = 0.0
    scores = []
    with open(trans_file, 'r', encoding='utf-8') as fh:
        for line in fh.readlines():
            line = _clean(line, word_option=word_option).split()
            total += 1
            my_unk = 0.0
            for word in line:
                if word == '<unk>':
                    unk += 1
                    my_unk += 1
                    break
            if len(line) == 0:
                scores.append(1.0)
            else:
                scores.append(my_unk)
    result_dict = {
        'UNKSentRate': unk / total,
        'UNKSentRate_Scores': scores,
    }
    return result_dict


def _distinct_c(trans_file, max_order=1, word_option=None):
    """Compute Distinct Score"""

    translations = []
    with open(trans_file, 'r', encoding='utf-8') as fh:
        for line in fh.readlines():
            line = _clean(line, word_option=word_option)
            translations.append(line.split(" "))

    unique_tokens = set()
    local_counts = []
    for items in translations:
        # print(items)
        local_unique_tokens = set()
        for i in range(0, len(items) - max_order + 1):
            valid_words = []
            for x in items[i:i + max_order]:
                if x != '<unk>':
                    valid_words.append(x)
            tmp = ' '.join(valid_words)
            unique_tokens.add(tmp)
            local_unique_tokens.add(tmp)
        local_counts.append(len(local_unique_tokens))
    ratio = len(unique_tokens)
    result_dict = {
        'DIST_Count_%d' % max_order: ratio,
        'DIST_Count_%d_Scores' % max_order: local_counts,
    }
    return result_dict


def _std_ent_n(trans_file, word_option=None):
    # based on Yizhe Zhang's code https://github.com/mgalley/DSTC7-End-to-End-Conversation-Modeling/blob/master/evaluation/src/metrics.py
    ngram_scores = [0, 0, 0, 0]
    counter = [defaultdict(int), defaultdict(int), defaultdict(int), defaultdict(int)]
    i = 0
    for line in open(trans_file, encoding='utf-8'):
        i += 1
        words = _clean(line, word_option=word_option).split()
        for n in range(4):
            for idx in range(len(words) - n):
                ngram = ' '.join(words[idx:idx + n + 1])
                counter[n][ngram] += 1

    for n in range(4):
        total = sum(counter[n].values())
        for v in counter[n].values():
            ngram_scores[n] += - v / total * (np.log(v) - np.log(total))

    result_dict = {
        'ent1': ngram_scores[0],
        'ent2': ngram_scores[1],
        'ent3': ngram_scores[2],
        'ent4': ngram_scores[3],
    }

    return result_dict


# Follow //transconsole/localization/machine_translation/metrics/bleu_calc.py
def _bleu(ref_file, trans_file, max_order=4, word_option=None):
    """Compute BLEU scores and handling BPE."""
    smooth = False
    ref_files = [ref_file]
    reference_text = []
    for reference_filename in ref_files:
        with open(reference_filename, 'r+', encoding='utf-8') as fh:
            reference_text.append(fh.readlines())

    per_segment_references = []
    for references in zip(*reference_text):
        reference_list = []
        for reference in references:
            reference = _clean(reference, word_option)
            reference_list.append(reference.split(" "))
        per_segment_references.append(reference_list)

    #     print(per_segment_references[0:3])

    translations = []
    with open(trans_file, 'r+', encoding='utf-8') as fh:
        for line in fh.readlines():
            line = _clean(line, word_option=word_option)
            translations.append(line.split(" "))
    #     print(translations[0:3])

    # bleu_score, precisions, bp, ratio, translation_length, reference_length
    bleu_score, _, _, _, _, _ = bleu.compute_bleu(
        per_segment_references, translations, max_order, smooth)

    blue_scores = []
    for ref, trans in zip(per_segment_references, translations):
        tmp_bleu_score, _, _, _, _, _ = bleu.compute_bleu(
            [ref], [trans], max_order, smooth)
        blue_scores.append(tmp_bleu_score * 100)

    result_dict = {
        'BLEU-%d' % max_order: 100 * bleu_score,
        'BLEU-%d_Scores' % max_order: blue_scores,
    }
    return result_dict


def _rouge(ref_file, summarization_file, word_option=None):
    """Compute ROUGE scores and handling BPE."""

    references = []
    with open(ref_file, 'r', encoding='utf-8') as fh:
        for line in fh.readlines():
            references.append(_clean(line, word_option))

    hypotheses = []
    with open(summarization_file, 'r', encoding='utf-8') as fh:
        for line in fh.readlines():
            hypotheses.append(_clean(line, word_option=word_option))

    rouge_score_map = rouge.rouge(hypotheses, references)

    rouge_l_scores = []
    rouge_1_scores = []
    rouge_2_scores = []
    for hypo, ref in zip(hypotheses, references):
        tmp_rouge_score_map = rouge.rouge([hypo], [ref])
        rouge_l_scores.append(100 * tmp_rouge_score_map["rouge_l/f_score"])
        rouge_1_scores.append(100 * tmp_rouge_score_map["rouge_1/f_score"])
        rouge_2_scores.append(100 * tmp_rouge_score_map["rouge_2/f_score"])

    result_dict = {
        'ROUGE_L': 100 * rouge_score_map["rouge_l/f_score"],
        'ROUGE_L_Scores': rouge_l_scores,
        'ROUGE_1': 100 * rouge_score_map["rouge_1/f_score"],
        'ROUGE_1_Scores': rouge_1_scores,
        'ROUGE_2': 100 * rouge_score_map["rouge_2/f_score"],
        'ROUGE_2_Scores': rouge_2_scores,
    }
    return result_dict


def _moses_bleu(multi_bleu_script, tgt_test, trans_file, word_option=None):
    """Compute BLEU scores using Moses multi-bleu.perl script."""

    # TODO(thangluong): perform rewrite using python
    # BPE
    if word_option == "bpe":
        debpe_tgt_test = tgt_test + ".debpe"
        if not os.path.exists(debpe_tgt_test):
            # TODO(thangluong): not use shell=True, can be a security hazard
            subprocess.call("cp %s %s" % (tgt_test, debpe_tgt_test), shell=True)
            subprocess.call("sed s/@@ //g %s" % (debpe_tgt_test),
                            shell=True)
        tgt_test = debpe_tgt_test
    elif word_option == "spm":
        despm_tgt_test = tgt_test + ".despm"
        if not os.path.exists(despm_tgt_test):
            subprocess.call("cp %s %s" % (tgt_test, despm_tgt_test))
            subprocess.call("sed s/ //g %s" % (despm_tgt_test))
            subprocess.call(u"sed s/^\u2581/g %s" % (despm_tgt_test))
            subprocess.call(u"sed s/\u2581/ /g %s" % (despm_tgt_test))
        tgt_test = despm_tgt_test
    cmd = "%s %s < %s" % (multi_bleu_script, tgt_test, trans_file)

    # subprocess
    # TODO(thangluong): not use shell=True, can be a security hazard
    bleu_output = subprocess.check_output(cmd, shell=True)

    # extract BLEU score
    m = re.search("BLEU = (.+?),", bleu_output)
    bleu_score = float(m.group(1))

    return bleu_score


def std_eval(ref_src_file, ref_tgt_file, top1_out_file_path, topk_out_file_path,
             pre_embed_file, pre_embed_dim, subword, vocab_size, beam_width):

    metrics = 'unk,unk_sentence,rouge,bleu-1,bleu-2,bleu-3,bleu-4,' \
              'intra_distinct-1,intra_distinct-2,distinct-1,distinct-2,distinct_c-1,distinct_c-2,' \
              'len,std_ent,f1'.split(',')
    thread_pool = Pool(8)
    jobs = []
    for metric in metrics:
        if metric[:len('intra_distinct')] == 'intra_distinct':
            job = thread_pool.apply_async(evaluate, (
                metric, ref_tgt_file, ref_src_file, topk_out_file_path,
                pre_embed_file,  pre_embed_dim, None, subword, beam_width))
        else:
            job = thread_pool.apply_async(evaluate, (
                metric, ref_tgt_file, ref_src_file, top1_out_file_path,
                pre_embed_file, pre_embed_dim, None, subword, beam_width))
        jobs.append(job)

    thread_pool.close()
    thread_pool.join()

    scores_suffix = '_Scores'
    scores_suffix_len = len(scores_suffix)
    scores_len = -1

    result_dict = {}
    logger.info('[Pretrained-Embed]: %s' % pre_embed_file)
    if pre_embed_file is not None:
        embed_scores = evaluate('embed', ref_tgt_file, ref_src_file, top1_out_file_path,
                                             pre_embed_file, dim=pre_embed_dim, word_option=subword)

        for key in embed_scores:
            assert key not in result_dict
            result_dict[key] = embed_scores[key]
            if key[-scores_suffix_len:] != scores_suffix:
                print('%s->%s' % (key, embed_scores[key]))
            else:
                if scores_len == -1:
                    scores_len = len(embed_scores[key])
                else:
                    assert scores_len == len(embed_scores[key]), 'embed/'+key

    for job, metric in zip(jobs, metrics):
        scores = job.get()
        if scores is None:
            logger.info("[Metric] Ignore %s" % metric)
            continue
        for key in scores:
            assert key not in result_dict
            score_res = scores[key]
            if score_res is None:
                continue
            result_dict[key] = score_res
            if key[-scores_suffix_len:] != scores_suffix:
                print('%s->%s' % (key, score_res))
            else:
                if scores_len == -1:
                    scores_len = len(score_res)
                else:
                    assert scores_len == len(score_res), metric + '/' + key

    bleu_metrics = ['BLEU-1', 'BLEU-2', 'BLEU-3', 'BLEU-4']
    bleu_scores = [result_dict[x] for x in bleu_metrics]
    src_bleu_scores = [result_dict['SRC_' + x] for x in bleu_metrics]
    geomean_blue = bleu_scores[0] * bleu_scores[1] * bleu_scores[2] * bleu_scores[3]
    geomean_self_blue = src_bleu_scores[0] * src_bleu_scores[1] * src_bleu_scores[2] * src_bleu_scores[3]
    geomean_blue = math.pow(geomean_blue, 1/4)
    geomean_self_blue = math.pow(geomean_self_blue, 1/4)
    geomean_rouge = result_dict['ROUGE_L']
    geomean_overlap = geomean_rouge * geomean_blue
    geomean_overlap = math.pow(geomean_overlap, 1/2)

    geomean_diversity = result_dict['DIST_1'] * result_dict['DIST_2'] * result_dict['ent4']
    geomean_diversity = math.pow(geomean_diversity, 1/3)

    geomean_non_embed_score = geomean_overlap * geomean_diversity
    geomean_non_embed_score = math.pow(geomean_non_embed_score, 1/2)
    result_dict['GM_Relevance_Overlap'] = geomean_overlap
    result_dict['GM_Diversity'] = geomean_diversity
    result_dict['GM_Total_NoEmbed'] = geomean_non_embed_score
    result_dict['GM_Novelty_BLEU'] = 100 - geomean_self_blue


    if pre_embed_file is not None:
        geomean_embed = result_dict['EmbedA'] * result_dict['EmbedG'] * result_dict['EmbedX']
        geomean_embed = math.pow(geomean_embed, 1/3)
        geomean_embed_novelty = result_dict['SRC_EmbedA'] * result_dict['SRC_EmbedG'] * result_dict['SRC_EmbedX']
        geomean_embed_novelty = 1.0 - math.pow(geomean_embed_novelty, 1 / 3)
        geomean_relevance = geomean_overlap * geomean_embed
        geomean_relevance = math.pow(geomean_relevance, 1/2)
        geomean_score = geomean_relevance * geomean_diversity
        geomean_score = math.pow(geomean_score, 1/2)
        geomean_novelty = math.pow(geomean_self_blue * geomean_embed_novelty , 1/2)
        result_dict['GM_Relevance'] = geomean_relevance
        result_dict['GM_Relevance_Embed'] = geomean_embed
        result_dict['GM_Novelty_Embed'] = geomean_embed_novelty
        result_dict['GM_Novelty'] = geomean_novelty
        result_dict['GM_Total'] = geomean_score
        result_dict['GM_Total_Novel'] = math.pow(geomean_relevance * geomean_diversity * geomean_novelty, 1/3)


    return result_dict




def main(args):
    score_manager = ScoreManager(args.res_path, '')

    ref_src_file = args.ref_src_file
    ref_tgt_file = args.ref_tgt_file
    train_src_file = args.train_src_file
    train_tgt_file = args.train_tgt_file
    top1_out_file_path = args.test_file
    topk_out_file_path = args.test_file_topk
    pre_embed_file = args.pre_embed_file
    pre_embed_dim = args.pre_embed_dim
    vocab_size = args.vocab_size
    subword = args.subword

    res_dict = std_eval(ref_src_file, ref_tgt_file,
             top1_out_file_path, topk_out_file_path,
             pre_embed_file, pre_embed_dim, subword, vocab_size, args.beam_width)
    group_name = args.group_name
    score_manager.update_group(group_name, res_dict)


if __name__ == '__main__':
    start_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--res_path", type=str, default="",
                        help="enable binary selector")
    parser.add_argument("--pre_embed_file", type=str, default="/home/sixing/dataset/embed/tencent.txt",
                        help="enable binary selector")
    parser.add_argument("--test_file", type=str, default="", help="enable binary selector")
    parser.add_argument("--test_file_topk", type=str, default="", help="enable binary selector")
    parser.add_argument("--ref_src_file", type=str,
                        default="",
                        help="training src file")
    parser.add_argument("--ref_tgt_file", type=str,
                        default="",
                        help="enable binary selector")
    parser.add_argument("--train_src_file", type=str,
                        default="",
                        help="enable binary selector")
    parser.add_argument("--train_tgt_file", type=str,
                        default="",
                        help="enable binary selector")
    parser.add_argument("--fact_vocab", type=str, default="", help="enable binary selector")
    parser.add_argument("--test_fact_path", type=str, default="", help="enable binary selector")
    parser.add_argument("--pre_embed_dim", type=int, default=200, help="enable binary selector")
    parser.add_argument("--vocab_size", type=int, default=50000, help="enable binary selector")
    parser.add_argument("--thread", type=int, default=16, help="thread")
    parser.add_argument("--beam_width", type=int, default=0, help="beam_width")
    parser.add_argument("--group_name", type=str, default=None)
    parser.add_argument("--subword", type=str, default=None)
    args = parser.parse_args()

    main(args)
    print('Evaluation Time Consuming : %d' % (time.time() - start_time))
