class Instance(object):
    def __init__(self, label, weight, raw_features):
        self.label = label
        self.weight = weight
        self.raw_features = raw_features

    def featurize(self):
        raise Exception('Not yet implemented: Instance.featurize')

    def __repr__(self):
        return u'%s %f |%s' % (str(self.label), self.weight, \
                u' |'.join([u'%s %s' % (namespace, feats) for namespace, feats in self.featurize().iteritems()]))


class SimpleInstance(Instance):
    def featurize(self):
        return {'a': self.raw_features}
