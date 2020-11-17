from torch.nn.functional import softmax
from transformers import BertForNextSentencePrediction, BertTokenizer
import json
import argparse
import os
import csv
import torch


def read_original_data(fname):
    """
    Read original data. Only read O1 and O2
    as keys.
    """
    data = []
    with open(fname, 'r') as fr:
        lines = fr.readlines()
        for i, line in enumerate(lines):
            line = line.strip()
            line = json.loads(line)

            data.append(' | '.join([line['obs1'], line['obs2']]))

    print('Original data size: ', len(data))

    return data


def read_hyps_from_json(file):
    o1o2_hyps = []
    with open(file, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            line = json.loads(line)
            O1 = line["O1"].split("<|endoftext|>")[1]
            O2 = line["O2"]
            hyps = set(line["H_Candidates"])
            hyps = sorted(hyps)
            o1o2_hyps.append((O1, O2, hyps))
    return o1o2_hyps


def write_text(text, output_fn):
    with open(output_fn, "w") as fout:
        for t in text:
            fout.write(t + '\n')
    print ('printing {}'.format(output_fn))


def score_by_bert(A, hyps, B, model, tokenizer, device='cuda'):
    """
    Use BERT next-sentence-prediction to compute the scores of
    (A-hyps, B) and (A, hyps-B)

    Args:
        A: O1
        hyps: hypothesis
        B: O2
    """
    def _score(a, b):
        encoded = tokenizer.encode_plus(a, text_pair=b, return_tensors='pt')
        for k in encoded:
            encoded[k] = encoded[k].to(device)
        seq_relationship_logits = model(**encoded)[0]
        return (seq_relationship_logits[0, 0].tolist())

    res_A_hB = []
    res_Ah_B = []
    for hyp in hyps:
        if hyp == 'DEPRECATED':
            res_A_hB.append(-1)
            res_Ah_B.append(-1)
            continue
        hB = ' '.join([hyp, B])
        Ah = ' '.join([A, hyp])
        res_A_hB.append(_score(A, hB))
        res_Ah_B.append(_score(Ah, B))

    return res_A_hB, res_Ah_B


def rank_and_align(o1o2_list, hypotheses, scores, original_o1o2_list):
    """
    Rank hypotheses by the scores, and align with original_o1o2_list (i.e.,
    ensure each instance repeats the same number of times as in original_o1o2_list)
    """

    def _get_o1o2_cnt():
        o1o2_cnt = {}
        for o1o2 in original_o1o2_list:
            o1o2_cnt[o1o2] = o1o2_cnt.get(o1o2, 0) + 1
        return o1o2_cnt

    o1o2_cnt = _get_o1o2_cnt()

    ranked_hypotheses = []

    cur_h_score_list = []
    cur_o1o2 = ""
    for o1o2, h, score in zip(o1o2_list, hypotheses, scores):
        if len(cur_h_score_list) > 0 and o1o2 != cur_o1o2:
            # a new instance
            sorted_h_score_list = sorted(cur_h_score_list, key=lambda x: x[1], reverse=True)
            best_h = sorted_h_score_list[0][0]
            ranked_hypotheses.extend([best_h] * o1o2_cnt[cur_o1o2])

            cur_h_score_list = []

        cur_h_score_list.append((h, score))
        cur_o1o2 = o1o2

    # The last instance
    if len(cur_h_score_list) > 0:
        sorted_h_score_list = sorted(cur_h_score_list, key=lambda x: x[1], reverse=True)
        best_h = sorted_h_score_list[0][0]
        ranked_hypotheses.extend([best_h] * o1o2_cnt[cur_o1o2])

    return ranked_hypotheses


def _find_nth_overlapping(haystack, needle, n):
    """
    Copy from https://stackoverflow.com/questions/1883980/find-the-nth-occurrence-of-substring-in-a-string
    """
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+1)
        n -= 1
    return start


def _truncate_text(text):
    """
    Truncate text to remove trailing text and get complete
    sentences (e.g., ending with `.?!`)
    """
    if '<|endoftext|>' in text:
        text = text[:text.find('<|endoftext|>')]
    ## Truncate out text based on length
    if '.' not in text and '?' not in text and '!' not in text:
        return text
    else:
        first_period_index = text.find('.')
        if first_period_index >= 10:
            return text[:first_period_index+1]
        first_exclamation_index = text.find('!')
        if first_exclamation_index >= 10:
            return text[:first_exclamation_index+1]
        first_q_index = text.find('?')
        if first_q_index >= 10:
            return text[:first_q_index+1]

        period_index = _find_nth_overlapping(text, '.', 2)
        if period_index < 0:
            return text[:first_period_index+1]
        return text[:period_index+1]


def has_repeat_substring(s, MINLEN=5, MINCNT=4):
    d = {}
    has_repeat = False
    for sublen in range(int(len(s)/MINCNT)-1, MINLEN-1, -1):
        for i in range(0,len(s)-sublen):
            sub = s[i:i+sublen]
            if len(sub.strip()) < sublen:
                continue
            cnt = s.count(sub)
            if cnt >= MINCNT and sub not in d:
                 d[sub] = cnt
                 has_repeat = True
                 break
        if has_repeat:
            break
    return has_repeat


def process_hyps(hyps):
    """
    Do pre-processing such as truncation and
    discarding repeat texts.
    """
    hs = []
    for h in hyps:
        h = h.strip()
        h = _truncate_text(h)
        if has_repeat_substring(h):
            h = "DEPRECATED"
        hs.append(h)
    return hs


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--hyps_file", type=str, default="none",
                        help="Hypothesis josn file to be ranked.")
    parser.add_argument("--n_lines", type=int, default=-1,
                        help="Number of lines to process.")
    parser.add_argument("--output_dir", type=str, default='./outputs', help="Output dir")
    parser.add_argument("--original_data_file", type=str, default='../data/abductive/small_data.json',
                        help="Filename of the original input data for abductive reasoning.")
    parser.add_argument("--no_cuda", action="store_true", help="no cuda")

    args = parser.parse_args()

    # Set the device
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    def _maybe_create_dir(dirname):
        """Creates directory if doesn't exist
        """
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
            return True
        return False

    output_dir = args.output_dir
    _maybe_create_dir(output_dir)

    o1o2_hyps = read_hyps_from_json(args.hyps_file)  # list [o1,o2,hyps]

    original_o1o2_list = None
    if args.original_data_file != '':
        original_o1o2_list = read_original_data(args.original_data_file)

    # Load pretrained model
    model = BertForNextSentencePrediction.from_pretrained('bert-base-cased')
    model.to(device)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    # Rank
    o1o2 = []
    o1ho2 = []
    all_scores = []
    hypotheses = []
    processed = set()
    i = 0
    for O1, O2, hyps in o1o2_hyps:
        hyps = process_hyps(hyps)

        # The original dataset can include repeated instances.
        # We keep track and skip instances that are already processed
        key_ = O1 + ' | ' + O2
        if key_ not in processed:
            processed.add(key_)
        else:
            continue

        res_o1_ho2, res_o1h_o2 = score_by_bert(O1, hyps, O2, model, tokenizer)

        for ro1h_o2, ro1_ho2, h in zip(res_o1h_o2, res_o1_ho2, hyps):
            o1o2.append(O1 + ' | ' + O2)
            o1ho2.append(' '.join([O1, h, O2]))

            all_scores.append(ro1_ho2 + ro1h_o2)
            hypotheses.append(h.replace('\"', '\''))

        if i == args.n_lines:
            break
        i += 1

    ranked_hypotheses = rank_and_align(o1o2, hypotheses, all_scores, original_o1o2_list)

    write_text(o1o2, output_dir + '/o1o2.txt')
    write_text(o1ho2, output_dir + '/o1ho2.txt')
    write_text(hypotheses, output_dir + '/hypotheses.txt')
    write_text(ranked_hypotheses, output_dir + '/ranked_hypotheses.txt')
    write_text([str(x) for x in all_scores], output_dir + '/bert_score.txt')


if __name__ == "__main__":
    main()

