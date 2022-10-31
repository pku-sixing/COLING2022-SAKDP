import os

from finedial.utils.data_utils import input_helper
from finedial.utils.data_utils.data_agent import dialogue_data_agent
from finedial.utils.evaluation import eval_helper
from finedial.utils.logging.logger_helper import logger


def write_generation_results(params, generations, meta_generations, vocab_size, mode_name, beam_width,
                             ref_file_suffix=""):
    result_path = os.path.join(params['model_path'], params['experiment_name'], "decoded")
    try:
        os.makedirs(result_path)
    except:
        pass
    top1_out_file_path = os.path.join(result_path, "decoded_top1_" + mode_name + ".txt")
    topk_out_file_path = os.path.join(result_path, "decoded_topk_" + mode_name + ".txt")
    meta_path = os.path.join(result_path, "decoded_meta_" + mode_name + ".txt")
    with open(top1_out_file_path, 'w+', encoding='utf-8') as fout:
        count = 0
        for idx in range(0, len(generations), beam_width):
            fout.write(generations[idx] + '\n')
            count += 1
        logger.info('[Generations] %d generations have been wrote to %s' % (count, top1_out_file_path))

    with open(topk_out_file_path, 'w+', encoding='utf-8') as fout:
        count = 0
        for idx in range(0, len(generations), 1):
            fout.write(generations[idx] + '\n')
            count += 1
        logger.info('[Generations] %d generations have been wrote to %s' % (count, topk_out_file_path))

    with open(meta_path, 'w+', encoding='utf-8') as fout:
        count = 0
        for idx in range(0, len(meta_generations), 1):
            fout.write('#%d\n' % idx)
            for key in meta_generations[idx].keys():
                fout.write('%s\t%s\n' % (key, meta_generations[idx][key]))
            count += 1
        logger.info('[Generations] %d generations have been wrote to %s' % (count, meta_path))



    logger.info('[Evaluating generations using external metrics]')

    ref_src_file = params['dataset']['test_data_path_prefix'] + "." + dialogue_data_agent.SRC_SUFFIX + ref_file_suffix
    ref_tgt_file = params['dataset']['test_data_path_prefix'] + "." + dialogue_data_agent.TGT_SUFFIX + ref_file_suffix
    pre_embed_file = params['embedding'].get('pre_embed_file', None)
    pre_embed_dim = params['embedding'].get('pre_embed_dim', 200)
    subword = None

    res_dict = eval_helper.std_eval(ref_src_file, ref_tgt_file,
                                    top1_out_file_path, topk_out_file_path,
                                    pre_embed_file, pre_embed_dim, subword, vocab_size, beam_width)

    return res_dict