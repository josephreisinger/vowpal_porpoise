import gzip
import vw_py
import random
from itertools import ifilter, imap, izip, repeat


def stream_instances(positive_bag, negative_bag):
    instances = list(izip(repeat(1), positive_bag)) + list(izip(repeat(0), negative_bag))
    random.shuffle(instances)

    for l in instances:
        yield l


def clean(l):
    return l.strip().replace(':', ' ')


def make_instance(label, features):
    return '%d 1.0 | %s' % (label, features)


def train(positive_bag, negative_bag, passes, params):
    a = vw_py.VW(params)

    test_sample = []
    train_correct = 0
    train_total = 0
    for p in range(passes):
        for label, features in stream_instances(positive_bag, negative_bag):
            print make_instance(label, features)
            if random.random() < 0.01:
                test_sample.append((label, features))
            else:
                pred = int(a.learn(make_instance(label, features)) > 0.5)

            if pred == label:
                train_correct += 1
            train_total += 1

    a.finish()
    assert False

    # test
    b = vw_py.VW("--cache_file=alice.cache -i alice.model")

    test_correct = 0
    test_total = 0
    for label, instance in test_sample:
        pred = int(b.learn(make_instance(label, instance)) > 0.5)
        if label == pred:
            test_correct += 1
        test_total += 1

        # print pred, label, instance

    print 'TRAIN: %d / %d (%.3f%%) TEST: %d / %d (%.3f%%)' % (train_correct, train_total, train_correct / float(train_total) * 100, \
            test_correct, test_total, test_correct / float(test_total) * 100)
    b.finish()


if __name__ == '__main__':
    positive_bag = list(ifilter(lambda x: x, imap(clean, gzip.open('training_data/aiw.txt.gz'))))
    negative_bag = list(ifilter(lambda x: x, imap(clean, gzip.open('training_data/ttlg.txt.gz'))))

    train(positive_bag, negative_bag, 10, "--cache_file=alice.cache -f alice.model")

    for i in range(100):
        # train(positive_bag, negative_bag, 1, "--cache_file=alice.cache -i alice.model -f alice.model --bfgs --passes=10")
        train(positive_bag, negative_bag, 1, "--cache_file=alice.cache -i alice.model -f alice.model")
        # train(positive_bag, negative_bag, 10, "-i alice.model -f alice.model --save_per_pass --passes 10")
