from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

def data_splits():
    file = 'data_splits.json'
    with open(file, 'r') as f:
        data = json.load(f)
    splits = ['train', 'val', 'test']
    for split in splits:
        print("%s set has %s examples." % (split, len(data[split])))
        with open(split + '.txt', 'w') as f:
            for id in data[split]:
                f.write("%s\n" % id)


if __name__ == '__main__':
    data_splits()
