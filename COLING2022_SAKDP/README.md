The codes of our COLING22 paper `Section-Aware Commonsense Knowledge-Grounded Dialogue Generation with
               Pre-trained Language Model`

# Release Timeline 
We will release all codes soon, the timeline is:

- Now, we have provided the codes of our PriorRanking Network.
- We will soon release the reaming codes.

# PriorRanking Network

## Dataset 

Please download it from [Google Drive](https://drive.google.com/file/d/1Sipkjih_vII9MmarOVto-G4bc0PaDoE-/view?usp=sharing). Note that this dataset is only prepared for the prior ranking network, you need to download the new dataset after the full codes are released.


## Training
Before training the model, you can configure your model in ``configs/PriorRanking.json``. 

Please remember your ``model_path/experiment_name`` in the config file.

### Dataset Format (If you need to run this code on other dataset)
In the training stage, you should prepare the following files for each ``train/dev/test``:

Taking ``test`` as an example, the positive samples should be placed to ``test.src``
```commandline
Format:
Question : $YOUR_DIALOGUE_QUERY [SEP] Knowledge candidate : <CSK> $HEAD_ENTITY $RELATION $TAIL_ENTITY

Example:
问 题 : 感 谢 我 拍 照 的 技 术 吧 [SEP] 候 选 知 识 : <CSK> 感 谢 RelatedTo 谢 谢
```
each line corresponding to a context-knowledge pair, and this line must be tokenized by the according tokenizer.

Similarly, the negative samples should be placed to ``test.src_related`` using the same format.

Then, configure this part in your own config file (batch_size, max_src_len, max_tgt_len, vocab_path, sp_token_vocab_path, ...prefix):
```json
  "dataset": {
    "batch_size": 96,
    "mid_representation": false,
    "full_pretrained_vocab": true,
    "pair_wise_input_mode": true,
    "share_src_tgt_vocab": true,
    "copy_order": [],
    "cuda": true,
    "max_line": -1,
    "max_src_len": 105,
    "max_tgt_len": 105,
    "response_suffix" :  "src_related",
    "sp_token_vocab_path": "datasets/weibo/bert_ccm_gdfirst/sp_tokens.txt",
    "tgt_vocab_path": "datasets/weibo/vocab30K.txt",
    "src_vocab_path": "datasets/weibo/vocab30K.txt",
    "val_data_path_prefix": "datasets/weibo/ccm_contrastive/default_dev",
    "test_data_path_prefix": "datasets/weibo/ccm_contrastive/default_test",
    "training_data_path_prefix": "datasets/weibo/ccm_contrastive/default_train"
  },
```
In fact, ``src/tgt_vocab_path`` will not be used if using the default setting (BERT Encoder), so just simply give the path of your BERT vocab.
But you should pay attention to ``sp_token_vocab_path``, if you have changed the vocab of the default BERT.

We recommend to use the default BERT if your dataset is Chinese.

### Script
```commandline
python finenlp.py --config=configs/PriorRanking.json --mode=train
```

It costs about 5 hours on a RTX2080Ti, and it will achieve the best result at the ~7th epoch.
## Inference

Before running the inference, you should first generate a folder in your ``model_path/experiment_name``:
```commandline
mkdir YOUR_MODEL_PATH/YOUR_EXPERIMENT_NAME/rank_scores
```
The inference needs two files (in your `configs/PriorRanking_Infer_TestSet.json`):
```json
    "query_suffix" :  "rank_candidates",
    "response_suffix" :  "rank_tgt",
    "result_suffix": "/test",
    "bulks": [0,6],
    "test_data_path_prefix": "datasets/weibo/ccm_contrastive/infer/test_rank",
    "sp_token_vocab_path": "datasets/weibo/bert_ccm_gdfirst/sp_tokens.txt",
    "tgt_vocab_path": "datasets/weibo/vocab30K.txt",
    "src_vocab_path": "datasets/weibo/vocab30K.txt",
    "val_data_path_prefix": "datasets/weibo/ccm_contrastive/infer/demo_rank",
    "training_data_path_prefix": "datasets/weibo/ccm_contrastive/infer/demo_rank"
```
Here, `test.rank_candidates` has the same data format as the above, where each line is a context-knowledge pair. 

Excpet for this file, the reaming `test.rank_tgt` should be a placeholder file, and you should also assign a placeholder to the `val/train` datasets. Please check our provided data.



Subsequently, using the following commands. Not that this ``test``inference has been divided into 6 sub-infernece, each costs about 0.5h on a RTX2080.
```commandline
python finenlp.py --config=configs/PriorRanking_Infer_TestSet.json --mode infer_bulks
mkdir YOUR_DATA_PATH/bert_ccm_ranked
```
We also provide a ``configs/PriorRanking_Infer_TrainSet.json `` for the training set. The training set has been divided into 107  sub-inferneces.

If you have multiple gpus, you can run multiple instances at the same time. For each instance, you can config the range of data:

```json
    "bulks": [0,120],
    "test_data_path_prefix": "datasets/weibo/ccm_contrastive/infer/train_rank",
```
For example, if you need 4 instances, you can set the `"bulks"` to  [0,30], [30,60], [60,90],[90,120], respectively.
```commandline
export CUDA_VISIBLE_DEVICES=0
python finenlp.py --config=configs/PriorRanking_Infer_TrainSet_0.json --mode infer_bulks &

export CUDA_VISIBLE_DEVICES=1
python finenlp.py --config=configs/PriorRanking_Infer_TrainSet_1.json --mode infer_bulks &

export CUDA_VISIBLE_DEVICES=2
python finenlp.py --config=configs/PriorRanking_Infer_TrainSet_2.json --mode infer_bulks &

export CUDA_VISIBLE_DEVICES=3
python finenlp.py --config=configs/PriorRanking_Infer_TrainSet_3.json --mode infer_bulks &
```

## Evaluation and Outputs
Subsequently, you can use the provided script to evaluate the generated candidates, and then outputs the ranked candidates to `bert_ccm_ranked/[test/train/dev].naf_fact_all2`.

```commandline
python reranking_knowledge_candidates.py --mode=test
```

# Citation

```commandline

@inproceedings{DBLP:conf/coling/Wu000W22,
  author    = {Sixing Wu and
               Ying Li and
               Ping Xue and
               Dawei Zhang and
               Zhonghai Wu},
  editor    = {Nicoletta Calzolari and
               Chu{-}Ren Huang and
               Hansaem Kim and
               James Pustejovsky and
               Leo Wanner and
               Key{-}Sun Choi and
               Pum{-}Mo Ryu and
               Hsin{-}Hsi Chen and
               Lucia Donatelli and
               Heng Ji and
               Sadao Kurohashi and
               Patrizia Paggio and
               Nianwen Xue and
               Seokhwan Kim and
               Younggyun Hahm and
               Zhong He and
               Tony Kyungil Lee and
               Enrico Santus and
               Francis Bond and
               Seung{-}Hoon Na},
  title     = {Section-Aware Commonsense Knowledge-Grounded Dialogue Generation with
               Pre-trained Language Model},
  booktitle = {Proceedings of the 29th International Conference on Computational
               Linguistics, {COLING} 2022, Gyeongju, Republic of Korea, October 12-17,
               2022},
  pages     = {521--531},
  publisher = {International Committee on Computational Linguistics},
  year      = {2022},
  url       = {https://aclanthology.org/2022.coling-1.43},
  timestamp = {Thu, 20 Oct 2022 07:16:49 +0200},
  biburl    = {https://dblp.org/rec/conf/coling/Wu000W22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}

```


