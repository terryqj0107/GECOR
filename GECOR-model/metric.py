import csv
from collections import Counter
from nltk.util import ngrams
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import math, re, argparse
import json
import functools
from config import global_config as cfg
import pickle
from reader import clean_replace
import operator

en_sws = set(stopwords.words())
wn = WordNetLemmatizer()

order_to_number = {
    'first': 1, 'one': 1, 'seco': 2, 'two': 2, 'third': 3, 'three': 3, 'four': 4, 'forth': 4, 'five': 5, 'fifth': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nin': 9, 'ten': 10, 'eleven': 11, 'twelve': 12
}

def similar(a,b):
    return a == b or a in b or b in a or a.split()[0] == b.split()[0] or a.split()[-1] == b.split()[-1]
    #return a == b or b.endswith(a) or a.endswith(b)    

def setsub(a,b):
    junks_a = []
    useless_constraint = ['temperature','week','est ','quick','reminder','near']
    for i in a:
        flg = False
        for j in b:
            if similar(i,j):
                flg = True
        if not flg:
            junks_a.append(i)
    for junk in junks_a:
        flg = False
        for item in useless_constraint:
            if item in junk:
                flg = True
        if not flg:
            return False
    return True

def setsim(a,b):
    a,b = set(a),set(b)
    return setsub(a,b) and setsub(b,a)

class BLEUScorer(object):
    ## BLEU score calculator via GentScorer interface
    ## it calculates the BLEU-4 by taking the entire corpus in
    ## Calulate based multiple candidates against multiple references
    def __init__(self):
        pass

    def score(self, parallel_corpus):

        # containers
        count = [0, 0, 0, 0]
        clip_count = [0, 0, 0, 0]
        r = 0
        c = 0
        weights = [0.25, 0.25, 0.25, 0.25]

        # accumulate ngram statistics
        for hyps, refs in parallel_corpus:
            hyps = [hyp.split() for hyp in hyps]
            refs = [ref.split() for ref in refs]
            for hyp in hyps:

                for i in range(4):
                    # accumulate ngram counts
                    hypcnts = Counter(ngrams(hyp, i + 1))
                    cnt = sum(hypcnts.values())
                    count[i] += cnt

                    # compute clipped counts
                    max_counts = {}
                    for ref in refs:
                        refcnts = Counter(ngrams(ref, i + 1))
                        for ng in hypcnts:
                            max_counts[ng] = max(max_counts.get(ng, 0), refcnts[ng])
                    clipcnt = dict((ng, min(count, max_counts[ng])) \
                                   for ng, count in hypcnts.items())
                    clip_count[i] += sum(clipcnt.values())

                # accumulate r & c
                bestmatch = [1000, 1000]
                for ref in refs:
                    if bestmatch[0] == 0: break
                    diff = abs(len(ref) - len(hyp))
                    if diff < bestmatch[0]:
                        bestmatch[0] = diff
                        bestmatch[1] = len(ref)
                r += bestmatch[1]
                c += len(hyp)

        # computing bleu score
        p0 = 1e-7
        bp = 1 if c > r else math.exp(1 - float(r) / float(c))
        p_ns = [float(clip_count[i]) / float(count[i] + p0) + p0 \
                for i in range(4)]
        s = math.fsum(w * math.log(p_n) \
                      for w, p_n in zip(weights, p_ns) if p_n)
        bleu = bp * math.exp(s)
        return bleu


def report(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        args[0].metric_dict[func.__name__ + ' '+str(args[2])] = res
        return res
    return wrapper


class GenericEvaluator:
    def __init__(self, result_path):
        self.file = open(result_path, 'r')
        self.meta = []
        self.metric_dict = {}
        self.entity_dict = {}
        filename = result_path.split('/')[-1]
        dump_dir = './sheets/' + filename.replace('.csv','.report.txt')
        self.dump_file = open(dump_dir, 'w')

    def _print_dict(self, dic):
        for k, v in sorted(dic.items(),key=lambda x:x[0]):
            print(k+'\t'+str(v))


    @report
    def user_utterance_bleu_metric(self, data, type='BLEU'):
        gen, truth = [], []
        for row in data:
            gen.append(row['generated_user'])
            truth.append(row['user_complete'])
        wrap_generated = [[_] for _ in gen]
        wrap_truth = [[_] for _ in truth]
        sc = BLEUScorer().score(zip(wrap_generated, wrap_truth))
        return sc

    def run_metrics(self):
        raise ValueError('Please specify the evaluator first, bro')

    def read_result_data(self):
        while True:
            line = self.file.readline()
            if 'START_CSV_SECTION' in line:
                break
            self.meta.append(line)
        reader = csv.DictReader(self.file)
        data = [_ for _ in reader]
        return data

    def _extract_constraint(self, z):
        z = z.split()
        if 'EOS_Z1' not in z:
            return set(z).difference(['name', 'address', 'postcode', 'phone', 'area', 'pricerange', 'restaurant',
                                           'restaurants', 'style', 'price', 'food', 'EOS_M'])
        else:
            idx = z.index('EOS_Z1')
            return set(z[:idx]).difference(['name', 'address', 'postcode', 'phone', 'area', 'pricerange', 'restaurant',
                                           'restaurants', 'style', 'price', 'food', 'EOS_M'])

    def _extract_request(self, z):
        z = z.split()
        if 'EOS_Z1' not in z or z[-1] == 'EOS_Z1':
            return set()
        else:
            idx = z.index('EOS_Z1')
            return set(z[idx+1:])

    def pack_dial(self,data):
        dials = {}
        for turn in data:
            dial_id = int(turn['dial_id'])
            if dial_id not in dials:
                dials[dial_id] = []
            dials[dial_id].append(turn)
        return dials

    def dump(self):
        self.dump_file.writelines(self.meta)
        self.dump_file.write('START_REPORT_SECTION\n')
        for k,v in self.metric_dict.items():
            self.dump_file.write('{}\t{}\n'.format(k,v))


    def clean(self,s):
        s = s.replace('<go> ', '').replace(' SLOT', '_SLOT')
        s = '<GO> ' + s + ' </s>'
        for item in self.entity_dict:
            # s = s.replace(item, 'VALUE_{}'.format(self.entity_dict[item]))
            s = clean_replace(s, item, '{}_SLOT'.format(self.entity_dict[item]))
        return s


class CamRestEvaluator(GenericEvaluator):
    def __init__(self, result_path):
        super().__init__(result_path)
        self.entities = []
        self.entity_dict = {}

    def run_metrics(self):
        raw_json = open(cfg.data)
        raw_entities = open(cfg.entity)
        raw_data = json.loads(raw_json.read().lower())
        raw_entities = json.loads(raw_entities.read().lower())
        self.get_entities(raw_entities)
        data = self.read_result_data()
        resolution_bleu_score = self.user_utterance_bleu_metric(data, 'BLEU')
        user_f1, user_precision, user_recall = self.user_utterance_f1_metric(data, 'F1')
        resolution_f1, resolution_precision, resolution_recall = self.resolution_f1_metric(data, 'Resolution_F1')
        user_accuracy = self.user_utterance_acc_metric(data, 'Accuracy')
        self._print_dict(self.metric_dict)
        return resolution_bleu_score, user_f1, user_accuracy


    @report
    def user_utterance_acc_metric(self, data, sub='Accuracy'):
        dials = self.pack_dial(data)
        complete_count, incomplete_count = 0, 0
        complete_correct, incomplete_correct = 0, 0
        unchange_false, change_false, incomplete_false = 0, 0, 0
        correct, total = 0, 0
        for dial_id in dials:
            dial = dials[dial_id]
            for turn_num, turn in enumerate(dial):
                total += 1
                gen_user_token = turn['generated_user'].split()
                user_complete_token = turn['user_complete'].split()
                user_token = turn['user'].split()
                if operator.eq(user_token, user_complete_token):
                    complete_count += 1
                    if operator.eq(gen_user_token, user_complete_token):
                        complete_correct += 1
                    else:
                        pass
                else:
                    incomplete_count += 1
                    if operator.eq(gen_user_token, user_complete_token):
                        incomplete_correct += 1
                    else:
                        incomplete_false += 1
                        if operator.eq(gen_user_token, user_token):
                            unchange_false += 1
                        else:
                            change_false += 1

                if operator.eq(gen_user_token, user_complete_token):
                    correct += 1
        # print('correct:{} , total:{} '.format(correct, total))
        accuracy = correct / (total + 1e-8)
        complete_accuracy = complete_correct / (complete_count + 1e-8)
        incomplete_accuracy = incomplete_correct / (incomplete_count + 1e-8)
        print('complete_accuracy:{}  =  {}/{}'.format(complete_accuracy, complete_correct, complete_count))
        print('incomplete_accuracy:{}  =  {}/{}'.format(incomplete_accuracy, incomplete_correct, incomplete_count))
        print('accuracy:{}  =  {}/{}'.format(accuracy, correct, total))
        unchange_false_rate = unchange_false / (incomplete_false + 1e-8)
        change_false_rate = change_false / (incomplete_false + 1e-8)
        print('unchange_false_rate:{}   change_false_rate:{}'.format(unchange_false_rate, change_false_rate))
        return accuracy


    @report
    def resolution_f1_metric(self, data, sub='resolution_F1'):
        dials = self.pack_dial(data)
        tp, fp, fn = 0, 0, 0
        for dial_id in dials:
            complete_user, gen_user, user = set(), set(), set()
            dial = dials[dial_id]
            for turn_num, turn in enumerate(dial):
                gen_user_token = turn['generated_user'].split()
                user_complete_token = turn['user_complete'].split()
                user_token = turn['user'].split()
                for idx, w in enumerate(user_token):
                    user.add(w)
                for idx, w in enumerate(gen_user_token):
                    gen_user.add(w)
                for idx, w in enumerate(user_complete_token):
                    complete_user.add(w)

            complete_user = complete_user - user
            gen_user = gen_user - user

            for token in gen_user:
                if token in complete_user:
                    tp += 1
                else:
                    fp += 1
            for token in complete_user:
                if token not in gen_user:
                    fn += 1
        precision, recall = tp / (tp + fp + 1e-8), tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return f1, precision, recall



    @report
    def user_utterance_f1_metric(self, data, sub='F1'):
        dials = self.pack_dial(data)
        tp, fp, fn = 0, 0, 0
        for dial_id in dials:
            truth_user, gen_user = set(), set()
            dial = dials[dial_id]
            for turn_num, turn in enumerate(dial):
                gen_user_token = turn['generated_user'].split()
                user_complete_token = turn['user_complete'].split()
                for idx, w in enumerate(gen_user_token):
                    gen_user.add(w)
                for idx, w in enumerate(user_complete_token):
                    truth_user.add(w)

            for token in gen_user:
                if token in truth_user:
                    tp += 1
                else:
                    fp += 1
            for token in truth_user:
                if token not in gen_user:
                    fn += 1
        precision, recall = tp / (tp + fp + 1e-8), tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return f1, precision, recall


    def get_entities(self, entity_data):
        for k in entity_data['informable']:
            self.entities.extend(entity_data['informable'][k])
            for item in entity_data['informable'][k]:
                self.entity_dict[item] = k

    def _extract_constraint(self, z):
        z = z.split()
        if 'EOS_Z1' not in z:
            s = set(z)
        else:
            idx = z.index('EOS_Z1')
            s = set(z[:idx])
        if 'moderately' in s:
            s.discard('moderately')
            s.add('moderate')
        #print(self.entities) 
        #return s
        return s.intersection(self.entities)
        #return set(z).difference(['name', 'address', 'postcode', 'phone', 'area', 'pricerange'])

    def _extract_request(self, z):
        z = z.split()
        return set(z).intersection(['address', 'postcode', 'phone', 'area', 'pricerange','food'])


def metric_handler():
    parser = argparse.ArgumentParser()
    parser.add_argument('-file')
    parser.add_argument('-type')
    args = parser.parse_args()
    ev_class = None
    if args.type == 'camrest':
        ev_class = CamRestEvaluator
    ev = ev_class(args.file)
    ev.run_metrics()
    ev.dump()


if __name__ == '__main__':
    metric_handler()
