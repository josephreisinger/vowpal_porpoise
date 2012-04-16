from vowpal_porpoise import VW, SimpleInstance, VPLogger
import itertools


class SimpleModel(object):
    def __init__(self, moniker):
        self.moniker = moniker
        self.log = VPLogger()
        self.model = VW(moniker=moniker, \
                        logger=self.log, \
                        **{'passes': 10,
                           'learning_rate': 15,
                           'power_t': 1.0, })

    def train(self, instance_stream):
        """
        Trains the model on the given data stream.
        """
        self.log.info('%s: training' % (self.moniker))
        with self.model.training():
            seen = 0
            for instance in instance_stream:
                self.model.push_instance(instance)
                seen += 1
                if seen % 10000 == 0:
                    self.log.debug('streamed %d instances...' % seen)
            self.log.debug('done streaming.')
        self.log.info('%s: trained on %d data points' % (self.moniker, seen))
        return self

    def predict(self, instance_stream):
        self.log.info('%s: predicting' % self.moniker)
        instances = []
        with self.model.predicting():
            seen = 0
            for instance in instance_stream:
                self.model.push_instance(instance)
                instances.append(instance)
                seen += 1

        self.log.info('%s: predicted for %d data points' % (self.moniker, seen))
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
