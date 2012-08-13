from contextlib import contextmanager
import os
import sys
import subprocess
import shlex
import tempfile
from vp_utils import safe_remove, VPLogger

class VW:
    def __init__(self,
                 logger=None,
                 vw='vw',
                 moniker=None,
                 name=None,
                 bits=None,
                 loss=None,
                 passes=None,
                 log_stderr_to_file=False,
                 silent=False,
                 l1=None,
                 l2=None,
                 learning_rate=None,
                 quadratic=None,
                 audit=None,
                 power_t=None,
                 adaptive=False,
                 working_dir=None,
                 decay_learning_rate=None,
                 initial_t=None,
                 lda=None,
                 lda_D=None,
                 lda_rho=None,
                 lda_alpha=None,
                 minibatch=None,
                 total=None,
                 node=None,
                 unique_id=None,
                 span_server=None,
                 bfgs=None,
                 oaa=None,
                 old_model=None,
                 incremental=False,
                 mem=None,
                 **kwargs):
        assert moniker and passes

        if logger is None:
            self.log = VPLogger()
        else:
            self.log = logger

        self.node = node
        self.total = total
        self.unique_id = unique_id
        self.span_server = span_server
        if self.node is not None:
            assert self.total is not None
            assert self.unique_id is not None
            assert self.span_server is not None

        if name is None:
            self.handle = '%s' % moniker
        else:
            self.handle = '%s.%s' % (moniker, name)

        if self.node is not None:
            self.handle = "%s.%d" % (self.handle, self.node)

        if old_model is None:
            self.filename = '%s.model' % self.handle
            self.incremental = False
        else:
            self.filename = old_model
            self.incremental = True

        self.incremental = incremental
        self.filename = '%s.model' % self.handle

        self.name = name
        self.bits = bits
        self.loss = loss
        self.vw = vw
        self.l1 = l1
        self.l2 = l2
        self.learning_rate = learning_rate
        self.log_stderr_to_file = log_stderr_to_file
        self.silent = silent
        self.passes = passes
        self.quadratic = quadratic
        self.power_t = power_t
        self.adaptive = adaptive
        self.decay_learning_rate = decay_learning_rate
        self.audit = audit
        self.initial_t = initial_t
        self.lda = lda
        self.lda_D = lda_D
        self.lda_rho = lda_rho
        self.lda_alpha = lda_alpha
        self.minibatch = minibatch
        self.oaa = oaa
        self.bfgs = bfgs
        self.mem = mem

        # Do some sanity checking for compatability between models
        if self.lda:
            assert not self.l1
            assert not self.l1
            assert not self.l2
            assert not self.loss
            assert not self.adaptive
            assert not self.oaa
            assert not self.bfgs
        else:
            assert not self.lda_D
            assert not self.lda_rho
            assert not self.lda_alpha
            assert not self.minibatch

        self.working_directory = working_dir or os.getcwd()

    def vw_base_command(self, base):
        l = base
        if self.bits                is not None: l.append('-b %d' % self.bits)
        if self.learning_rate       is not None: l.append('--learning_rate=%f' % self.learning_rate)
        if self.l1                  is not None: l.append('--l1=%f' % self.l1)
        if self.l2                  is not None: l.append('--l2=%f' % self.l2)
        if self.initial_t           is not None: l.append('--initial_t=%f' % self.initial_t)
        if self.quadratic           is not None: l.append('-q %s' % self.quadratic)
        if self.power_t             is not None: l.append('--power_t=%f' % self.power_t)
        if self.loss                is not None: l.append('--loss_function=%s' % self.loss)
        if self.decay_learning_rate is not None: l.append('--decay_learning_rate=%f' % self.decay_learning_rate)
        if self.lda                 is not None: l.append('--lda=%d' % self.lda)
        if self.lda_D               is not None: l.append('--lda_D=%d' % self.lda_D)
        if self.lda_rho             is not None: l.append('--lda_rho=%f' % self.lda_rho)
        if self.lda_alpha           is not None: l.append('--lda_alpha=%f' % self.lda_alpha)
        if self.minibatch           is not None: l.append('--minibatch=%d' % self.minibatch)
        if self.oaa                 is not None: l.append('--oaa=%d' % self.oaa)
        if self.unique_id           is not None: l.append('--unique_id=%d' % self.unique_id)
        if self.total               is not None: l.append('--total=%d' % self.total)
        if self.node                is not None: l.append('--node=%d' % self.node)
        if self.span_server         is not None: l.append('--span_server=%s' % self.span_server)
        if self.mem                 is not None: l.append('--mem=%d' % self.mem)
        if self.audit:                           l.append('--audit')
        if self.bfgs:                            l.append('--bfgs')
        if self.adaptive:                        l.append('--adaptive')
        return ' '.join(l)

    def vw_train_command(self, cache_file, model_file):
        if os.path.exists(model_file) and self.incremental:
            return self.vw_base_command([self.vw]) + ' --passes %d --cache_file %s -i %s -f %s' \
                    % (self.passes, cache_file, model_file, model_file)
        else:
            self.log.debug('No existing model file or not options.incremental')
            return self.vw_base_command([self.vw]) + ' --passes %d --cache_file %s -f %s' \
                    % (self.passes, cache_file, model_file)

    def vw_test_command(self, model_file, prediction_file):
        return self.vw_base_command([self.vw]) + ' -t -i %s -p %s' % (model_file, prediction_file)

    def vw_test_command_library(self, model_file):
        return self.vw_base_command([]) + ' -t -i %s' % (model_file)

    @contextmanager
    def training(self):
        self.start_training()
        yield
        self.close_process()

    @contextmanager
    def predicting(self):
        self.start_predicting()
        yield
        self.close_process()

    @contextmanager
    def predicting_library(self):
        self.start_predicting_library()
        yield
        self.end_predicting_library()

    def start_training(self):
        cache_file = self.get_cache_file()
        model_file = self.get_model_file()

        # Remove the old cache and model files
        if not self.incremental:
            safe_remove(cache_file)
            safe_remove(model_file)

        # Run the actual training
        self.vw_process = self.make_subprocess(self.vw_train_command(cache_file, model_file))

        # set the instance pusher
        self.push_instance = self.push_instance_stdin

    def close_process(self):
        # Close the process
        assert self.vw_process
        self.vw_process.stdin.flush()
        self.vw_process.stdin.close()
        if self.vw_process.wait() != 0:
            raise Exception("vw_process %d (%s) exited abnormally with return code %d" % \
                (self.vw_process.pid, self.vw_process.command, self.vw_process.returncode))

    def push_instance_stdin(self, instance):
        self.vw_process.stdin.write(('%s\n' % instance).encode('utf8'))

    def start_predicting(self):
        model_file = self.get_model_file()
        # Be sure that the prediction file has a unique filename, since many processes may try to
        # make predictions using the same model at the same time
        _, prediction_file = tempfile.mkstemp(dir='.', prefix=self.get_prediction_file())
        os.close(_)

        self.vw_process = self.make_subprocess(self.vw_test_command(model_file, prediction_file))
        self.prediction_file = prediction_file
        self.push_instance = self.push_instance_stdin

    def start_predicting_library(self):
        import vw_py
        model_file = self.get_model_file()
        self.vw_process = vw_py.VW(self.vw_test_command_library(model_file))

        # Set the library instance pusher
        self.push_instance = self.predict_push_instance

    def end_predicting_library(self):
        # Close the process
        assert self.vw_process
        self.vw_process.finish()

    def parse_prediction(self, p):
        if self.lda:
            return map(float, p.split())
        else:
            return float(p)

    def read_predictions_(self):
        for x in open(self.prediction_file):
            yield self.parse_prediction(x)
        # clean up the prediction file
        os.remove(self.prediction_file)

    def predict_push_instance(self, instance):
        return self.parse_prediction(self.vw_process.learn(('%s\n' % instance).encode('utf8')))

    def make_subprocess(self, command):
        if not self.log_stderr_to_file:
            stdout = open('/dev/null', 'w')
            stderr = open('/dev/null', 'w') if self.silent else sys.stderr
            self.current_stdout = None
            self.current_stderr = None
        else:
            # Save the output of vw to file for debugging purposes
            log_file_base = tempfile.mktemp(dir=self.working_directory, prefix="vw-")
            self.current_stdout = log_file_base + '.out'
            self.current_stderr = log_file_base + '.err'
            stdout = open(self.current_stdout, 'w')
            stderr = open(self.current_stderr, 'w')
            stdout.write(command + '\n')
            stderr.write(command + '\n')
        self.log.debug('Running command: "%s"' % str(command))
        result = subprocess.Popen(shlex.split(str(command)), stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, close_fds=True, universal_newlines=True)
        result.command = command
        return result

    def get_current_stdout(self):
        return open(self.current_stdout)

    def get_current_stderr(self):
        return open(self.current_stderr)

    def get_model_file(self):
        return os.path.join(self.working_directory, self.filename)

    def get_cache_file(self):
        return os.path.join(self.working_directory, '%s.cache' % (self.handle))

    def get_prediction_file(self):
        return os.path.join(self.working_directory, '%s.prediction' % (self.handle))
