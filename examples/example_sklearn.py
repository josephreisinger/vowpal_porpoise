from vowpal_porpoise.sklearn import VW_Classifier, VW_Regressor


# train + test data
X_train = [
    {'big': 1,   'red': 1, 'square': 1},
    {'small':1, 'blue': 1, 'circle': 1}
]
Y_train = [1, -1]

X_test  = [
    { 'large': 1, 'burnt': 1,  'red': 1, 'rhombus': 1},
    {'little': 1,  'blue': 1, 'oval': 1}
]

def run(cls, loss):
  # construct Vowpal_XXX
  vw = cls(
      moniker='test',    # a name for the model
      passes=10,         # vw arg: passes
      loss='hinge',      # vw arg: loss
      learning_rate=10,  # vw arg: learning_rate
      silent=True,
  )

  # train and predict
  Y_test = vw.fit(X_train, Y_train).predict(X_test)

  print 'model: %s' % cls.__name__
  for (x, y) in zip(X_test, Y_test):
    if isinstance(y, int):
      print "%+2d | %s" % (y, x)
    else:
      print "%+5f | %s" % (y, x)



if __name__ == '__main__':
  run(VW_Classifier,  'hinge')
  run(VW_Regressor,   'squared')
