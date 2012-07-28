from vowpal_porpoise import VW, SimpleInstance
import itertools


class SimpleModel(object):
    def __init__(self, moniker):
        self.moniker = moniker
        self.model = VW(moniker=moniker, \
                        **{'passes': 10,
                           'learning_rate': 15,
                           'power_t': 1.0, })

    def train(self, instance_stream):
        """
        Trains the model on the given data stream.
        """
        print '%s: training' % (self.moniker)
        with self.model.training():
            seen = 0
            for instance in instance_stream:
                self.model.push_instance(instance)
                seen += 1
                if seen % 10000 == 0:
                    print 'streamed %d instances...' % seen
            print 'done streaming.'
        print '%s: trained on %d data points' % (self.moniker, seen)
        return self

    def predict_library(self, instance_stream):
        print '%s: predicting' % self.moniker
        with self.model.predicting_library():
            seen = 0
            for instance in instance_stream:
                yield instance, self.model.push_instance(instance)
                seen += 1

        print '%s: predicted for %d data points' % (self.moniker, seen)

    def predict(self, instance_stream):
        print '%s: predicting' % self.moniker
        instances = []
        with self.model.predicting():
            seen = 0
            for instance in instance_stream:
                self.model.push_instance(instance)
                instances.append(instance)
                seen += 1

        print '%s: predicted for %d data points' % (self.moniker, seen)
        predictions = list(self.model.read_predictions_())
        if seen != len(predictions):
            raise Exception("Number of labels and predictions do not match!  (%d vs %d)" % \
                (seen, len(predictions)))
        return itertools.izip(instances, predictions)


if __name__ == '__main__':
    instances = [
            SimpleInstance(1.0, 1.0, "the quick brown fox"),
            SimpleInstance(1.0, 1.0, "the smart brown fox"),
            SimpleInstance(1.0, 1.0, "the smart brown wolf"),
            SimpleInstance(0.0, 1.0, "the slow brown sheep"),
            SimpleInstance(0.0, 1.0, "the stupid brown sheep"),
            SimpleInstance(0.0, 1.0, "the stupid brown lamp"),
            ]

    for (instance, prediction) in SimpleModel('example1').train(instances).predict(instances):
        print prediction, instance
