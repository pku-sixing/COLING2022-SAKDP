# Set the target file: 'test' or 'dev'/'train'
# we use the data-level parallel, so there are more than one bulk for each file
# this is the default number
import argparse
import os

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--modes', type=str, default='test', help='train, dev, test')
parser.add_argument('--model_path', type=str, default='model/prior_ranking/PriorRanking', help='model path')
parser.add_argument('--dataset_path', type=str, default='datasets/weibo/', help='dataaset path')
args = parser.parse_args()
bulk_nums = {
    'test': 6,
    'dev': 6,
    'train': 102,
}
# Finally, the re-ranked result ``test/dev/train.naf_fact_all2`` can be found in the dataset path

for file_in in args.modes.split(','):
    precisions = [0] * 100
    recalls = [0] * 1000
    f1s = [0] * 1000
    total_count = 0
    mrr = 0.0
    bulk_num = bulk_nums[file_in]

    file_path = os.path.join(args.dataset_path, "%s.naf_fact" % file_in)
    out_file_path = os.path.join(args.dataset_path, "bert_ccm_ranked/%s.naf_fact_all2" % file_in)
    ground_truth = os.path.join(args.dataset_path, "%s.naf_golden_facts_position" % file_in)

    with open(ground_truth) as fin:
        ground_truth = [[int(y) for y in x.strip('\r\n').split()] for x in fin.readlines()]

    print(file_path)
    with open(file_path, 'r+', encoding='utf-8') as fin:
        my_fact_ids = [x.strip('\r\n').split() for x in fin.readlines()]

    with open(out_file_path, 'w+', encoding='utf-8') as fout:

        for bulk_id in range(bulk_num):
            file = os.path.join(args.dataset_path, 'ccm_contrastive/infer/%s_rank.rank_labels_%d' % (
                file_in, bulk_id))
            score_file = os.path.join(args.model_path, 'rank_scores/%s_%d.txt' % (file_in, bulk_id))
            with open(score_file) as fin:
                scores = [float(x.strip('\r\n')) for x in fin.readlines()]

            with open(file) as fin:
                labels = fin.readlines()
                labels = [x.strip('\r\n').split() for x in labels]

            assert len(scores) == len(labels)

            last_lable = -1
            golden_pos = []
            pairs = []
            labels.append(['-2', 'False'])
            for idx in range(len(labels)):
                lable = labels[idx]
                if lable[0] != last_lable or idx == len(labels) - 1:
                    if last_lable == -1:
                        pass
                    else:
                        new_pairs = []
                        for pair in pairs:
                            if pair[0] == 0:
                                continue
                            new_pairs.append(pair)
                        pairs = new_pairs
                        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)

                        original_ids = my_fact_ids[total_count]
                        sorted_ids = [original_ids[x[0]] for x in sorted_pairs]
                        fout.write(' '.join(sorted_ids) + '\n')

                        #                 print(lable[0], golden_pos)
                        #                 print(sorted_pairs)
                        total_count += 1
                        golden_pos = set(golden_pos)
                        orders = [x[0] for x in sorted_pairs]
                        has_mrr = False
                        for k in range(100):
                            parts = set(orders[0:k + 1])
                            pre = (len(parts & golden_pos) / len(parts))
                            recall = (len(parts & golden_pos) / len(golden_pos))
                            if not has_mrr and (pre + recall > 0):
                                has_mrr = True
                                mrr += 1 / (k + 1)
                            precisions[k] += pre
                            recalls[k] += recall
                            if pre + recall != 0:
                                f1s[k] += pre * recall * 2 / (pre + recall)
                        golden_pos = []
                        pairs = []
                if idx == len(labels) - 1:
                    break
                last_lable = lable[0]
                if lable[1] == 'True':
                    golden_pos.append(len(pairs))
                pairs.append((len(pairs), scores[idx]))

    print(out_file_path)
    assert total_count == len(my_fact_ids)
    print(total_count, 'MRR', mrr / total_count)
    for k in [1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 100]:
        pre = precisions[k - 1] / total_count * 100
        recall = recalls[k - 1] / total_count * 100
        macro = f1s[k - 1] / total_count * 100
        micro = pre * recall * 2 / (pre + recall)
        print(k, 'p=%.2f%%,r=%.2f%%,macro=%.2f%%,micro=%.2f%%' % (pre, recall, macro, micro))
