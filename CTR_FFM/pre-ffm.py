# -*- coding=utf-8 -*-

import argparse, csv, sys

from common import *

if len(sys.argv) == 1:
    sys.argv.append('-h')

from common import *

parser = argparse.ArgumentParser()
parser.add_argument('csv_path', type=str)
parser.add_argument('out_path', type=str)
args = vars(parser.parse_args())


def gen_fm_feats(feats):
    feats = ['{0}:{1}:{2}'.format(field, feat, value) for (field, feat, value) in feats]
    return feats


with open(args['out_path'], 'w') as f:
    for row in csv.DictReader(open(args['csv_path'])):
        feats = []
        for feat in gen_feats(row):
            f1, f2, f3 = feat.split('-')
            if f1[0] == "C":
                field = int(f1[1:])
                feats.append((field, f2, 1))

            else:
                feats.append((f1, f2, f3))

        feats = gen_fm_feats(feats)
        f.write(row['click'] + ' ' + ' '.join(feats) + '\n')
