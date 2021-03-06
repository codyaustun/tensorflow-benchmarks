# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Benchmark script for TensorFlow.

See the README for more information.
"""

from __future__ import print_function

import argparse
import math
import os
import subprocess
import sys
import threading
import time

import numpy as np

import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.python.client import timeline
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.platform import gfile
from tensorflow.python.util import nest
import benchmark_storage
import cnn_util
import convnet_builder
import datasets
import model_config
import variable_mgr
from cnn_util import log_fn

tf.flags.DEFINE_string('model', 'trivial', 'name of the model to run')

# The code will first check if it's running under benchmarking mode
# or evaluation mode, depending on FLAGS.eval:
# Under the evaluation mode, this script will read a saved model,
#   and compute the accuracy of the model against a validation dataset.
#   Additional ops for accuracy and top_k predictors are only used under this
#   mode.
# Under the benchmarking mode, user can specify whether nor not to use
#   the forward-only option, which will only compute the loss function.
#   forward-only cannot be enabled with eval at the same time.
tf.flags.DEFINE_boolean('eval', False, 'whether use eval or benchmarking')
tf.flags.DEFINE_integer('eval_interval_secs', 0,
                        """How often to run eval on saved checkpoints. Usually
                        the same as save_model_secs from the corresponding
                        training run. Pass 0 to eval only once.""")
tf.flags.DEFINE_boolean('forward_only', False, """whether use forward-only or
                         training for benchmarking""")
tf.flags.DEFINE_boolean('print_training_accuracy', False, """whether to
                        calculate and print training accuracy during
                        training""")
tf.flags.DEFINE_boolean('resume_training', False, """Resume training from last checkpoint
                         in passed-in checkpoint directory""")
tf.flags.DEFINE_integer('batch_size', 0, 'batch size per compute device')
tf.flags.DEFINE_integer('batch_group_size', 1,
                        'number of groups of batches processed in the image '
                        'producer.')
tf.flags.DEFINE_integer('num_batches', 100,
                        'number of batches to run, excluding warmup')
tf.flags.DEFINE_string('subset', None, 'Subset of data to operate on')
tf.flags.DEFINE_integer('num_warmup_batches', None,
                        'number of batches to run before timing')
tf.flags.DEFINE_integer('autotune_threshold', None,
                        'The autotune threshold for the models')
tf.flags.DEFINE_integer('num_gpus', 1, 'the number of GPUs to run on')
tf.flags.DEFINE_integer('display_every', 10,
                        """Number of local steps after which progress is printed
                        out""")
tf.flags.DEFINE_string('data_dir', None, """Path to dataset in TFRecord format
                       (aka Example protobufs). If not specified,
                       synthetic data will be used.""")
tf.flags.DEFINE_string('data_name', None,
                       """Name of dataset: imagenet or cifar10.
                       If not specified, it is automatically guessed
                       based on --data_dir.""")
tf.flags.DEFINE_string('resize_method', 'bilinear',
                       """Method for resizing input images:
                       crop, nearest, bilinear, bicubic, area, or round_robin.
                       The 'crop' mode requires source images to be at least
                       as large as the network input size.
                       The 'round_robin' mode applies different resize methods
                       based on position in a batch in a round-robin fashion.
                       Other modes support any sizes and apply random bbox
                       distortions before resizing (even with
                       --nodistortions).""")
tf.flags.DEFINE_boolean('distortions', True,
                        """Enable/disable distortions during image
                        preprocessing. These include bbox and color
                        distortions.""")
tf.flags.DEFINE_boolean('use_data_sets', False,
                        """Enable use of data sets for input pipeline""")
tf.flags.DEFINE_string('local_parameter_device', 'gpu',
                       """Device to use as parameter server: cpu or gpu.
                          For distributed training, it can affect where caching
                          of variables happens.""")
tf.flags.DEFINE_string('device', 'gpu',
                       """Device to use for computation: cpu or gpu""")
tf.flags.DEFINE_string('data_format', 'NCHW',
                       """Data layout to use: NHWC (TF native)
                       or NCHW (cuDNN native, requires GPU).""")
tf.flags.DEFINE_integer('num_intra_threads', 1,
                        """Number of threads to use for intra-op
                       parallelism. If set to 0, the system will pick
                       an appropriate number.""")
tf.flags.DEFINE_integer('num_inter_threads', 0,
                        """Number of threads to use for inter-op
                       parallelism. If set to 0, the system will pick
                       an appropriate number.""")
tf.flags.DEFINE_string('trace_file', None,
                       """Enable TensorFlow tracing and write trace to
                       this file.""")
tf.flags.DEFINE_string('graph_file', None,
                       """Write the model's graph definition to this
                       file. Defaults to binary format unless filename ends
                       in 'txt'.""")
tf.flags.DEFINE_string('optimizer', 'sgd',
                       'Optimizer to use: momentum or sgd or rmsprop or adam')
tf.flags.DEFINE_float('learning_rate', None,
                      """Initial learning rate for training.""")
tf.flags.DEFINE_float('num_epochs_per_decay', 0,
                      """Steps after which learning rate decays. If 0,
                      the learning rate does not decay.""")
tf.flags.DEFINE_float('learning_rate_decay_factor', 0,
                      """Learning rate decay factor. Decay by this factor every
                      `num_epochs_per_decay` epochs. If 0, learning rate does
                      not decay.""")
tf.flags.DEFINE_float('minimum_learning_rate', 0,
                      """The minimum learning rate. The learning rate will
                      never decay past this value. Requires  `learning_rate`,
                      `num_epochs_per_decay` and `learning_rate_decay_factor` to
                      be set.""")
tf.flags.DEFINE_float('momentum', 0.9, """Momentum for training.""")
tf.flags.DEFINE_float('rmsprop_decay', 0.9, """Decay term for RMSProp.""")
tf.flags.DEFINE_float('rmsprop_momentum', 0.9, """Momentum in RMSProp.""")
tf.flags.DEFINE_float('rmsprop_epsilon', 1.0, """Epsilon term for RMSProp.""")
tf.flags.DEFINE_float('adam_beta1', 0.9, """Beta1 term for Adam.""")
tf.flags.DEFINE_float('adam_beta2', 0.999, """Beta2 term for Adam.""")
tf.flags.DEFINE_float('adam_epsilon', 1e-8, """Epsilon term for Adam.""")
tf.flags.DEFINE_float('gradient_clip', None, """Gradient clipping magnitude.
                       Disabled by default.""")
tf.flags.DEFINE_float('weight_decay', 0.00004,
                      """Weight decay factor for training.""")
tf.flags.DEFINE_float('gpu_memory_frac_for_testing', 0, """If non-zero, the
                      fraction of GPU memory that will be used. Useful for
                      testing the benchmark script, as this allows distributed
                      mode to be run on a single machine. For example, if there
                      are two tasks, each can be allocated ~40% of the
                      memory on a single machine""")
tf.flags.DEFINE_boolean('use_tf_layers', None, """If True, use tf.layers for
                        neural network layers. If None, use tf.layers if
                        possible. This should not affect performance or accuracy
                        in any way.""")

# Performance tuning flags.
tf.flags.DEFINE_boolean('winograd_nonfused', True,
                        """Enable/disable using the Winograd non-fused
                        algorithms.""")
tf.flags.DEFINE_boolean('sync_on_finish', False,
                        """Enable/disable whether the devices are synced after
                        each step.""")
tf.flags.DEFINE_boolean('staged_vars', False,
                        """whether the variables are staged from the main
                        computation""")
tf.flags.DEFINE_boolean('force_gpu_compatible', True,
                        """whether to enable force_gpu_compatible in
                        GPU_Options""")
tf.flags.DEFINE_boolean('xla', False, """whether to enable XLA""")

# Performance tuning specific to MKL.
tf.flags.DEFINE_boolean('mkl', False,
                        """If true, set MKL environment variables.""")
tf.flags.DEFINE_integer('kmp_blocktime', 30,
                        """The time, in milliseconds, that a thread should
                        wait, after completing the execution of a parallel
                        region, before sleeping""")
tf.flags.DEFINE_string('kmp_affinity', 'granularity=fine,verbose,compact,1,0',
                       """Restricts execution of certain threads (virtual
                        execution units) to a subset of the physical processing
                        units in a multiprocessor computer.""")
tf.flags.DEFINE_integer('kmp_settings', 1,
                        """If set to 1, MKL settings will be printed.""")

# fp16 flags. If --use_fp16=False, no other fp16 flags apply.
tf.flags.DEFINE_boolean('use_fp16', False,
                        """Use 16-bit floats for certain tensors instead of
                        32-bit floats. This is currently experimental.""")
# TODO(reedwm): The default loss scale of 128 causes most models to diverge on
# the second step with synthetic data. Changing the tf.set_random_seed call to
# tf.set_random_seed(1235) or most other seed values causes the issue not to
# occur.
tf.flags.DEFINE_float('fp16_loss_scale', None, """If fp16 is enabled, the loss
                      is multiplied by this amount right before gradients are
                      computed, then each gradient is divided by this amount.
                      Mathematically, this has no effect, but it helps avoid
                      fp16 underflow. Set to 1 to effectively disable.""")
tf.flags.DEFINE_boolean('fp16_vars', False,
                        """If fp16 is enabled, also use fp16 for variables. If
                        False, the variables are stored in fp32 and casted to
                        fp16 when retrieved. Recommended to leave as False.
                        """)

# The method for managing variables:
#   parameter_server: variables are stored on a parameter server that holds
#       the master copy of the variable.  In local execution, a local device
#       acts as the parameter server for each variable; in distributed
#       execution, the parameter servers are separate processes in the cluster.
#       For each step, each tower gets a copy of the variables from the
#       parameter server, and sends its gradients to the param server.
#   replicated: each GPU has its own copy of the variables. To apply gradients,
#       nccl all-reduce or regular cross-device aggregation is used to replicate
#       the combined gradients to all towers (depending on --use_nccl option).
#   independent: each GPU has its own copy of the variables, and gradients are
#       not shared between towers. This can be used to check performance when no
#       data is moved between GPUs.
#   distributed_replicated: Distributed training only. Each GPU has a copy of
#       the variables, and updates its copy after the parameter servers are all
#       updated with the gradients from all servers. Only works with
#       cross_replica_sync=true. Unlike 'replicated', currently never uses
#       nccl all-reduce for replicating within a server.
tf.flags.DEFINE_string(
    'variable_update', 'parameter_server',
    ('The method for managing variables: '
     'parameter_server, replicated, distributed_replicated, independent'))
tf.flags.DEFINE_boolean(
    'use_nccl', True,
    'Whether to use nccl all-reduce primitives where possible')

# Distributed training flags.
tf.flags.DEFINE_string('job_name', '',
                       'One of "ps", "worker", "".  Empty for local training')
tf.flags.DEFINE_string('ps_hosts', '', 'Comma-separated list of target hosts')
tf.flags.DEFINE_string('worker_hosts', '',
                       'Comma-separated list of target hosts')
tf.flags.DEFINE_integer('task_index', 0, 'Index of task within the job')
tf.flags.DEFINE_string('server_protocol', 'grpc', 'protocol for servers')
tf.flags.DEFINE_boolean('cross_replica_sync', True, '')

# Summary and Save & load checkpoints.
tf.flags.DEFINE_integer('summary_verbosity', 0,
                        """Verbosity level for summary ops. Pass 0 to disable
                        both summaries and checkpoints.""")
tf.flags.DEFINE_integer('save_summaries_steps', 0,
                        """How often to save summaries for trained models.
                        Pass 0 to disable summaries.""")
tf.flags.DEFINE_string('train_dir', None,
                       """Path to session checkpoints. Pass None to disable
                       saving checkpoint at the end.""")
tf.flags.DEFINE_string('checkpoint_dir', None,
                       """Path to checkpoints.""")
tf.flags.DEFINE_string('eval_dir', '/tmp/tf_cnn_benchmarks/eval',
                       """Directory where to write eval event logs.""")
tf.flags.DEFINE_string('result_storage', None,
                       """Specifies storage option for benchmark results.
                       None means results won't be stored.
                       'cbuild_benchmark_datastore' means results will be stored
                       in cbuild datastore (note: this option requires special
                       pemissions and meant to be used from cbuilds).""")
FLAGS = tf.flags.FLAGS


class GlobalStepWatcher(threading.Thread):
  """A helper class for globe_step.

  Polls for changes in the global_step of the model, and finishes when the
  number of steps for the global run are done.
  """

  def __init__(self, sess, global_step_op,
               start_at_global_step, end_at_global_step):
    threading.Thread.__init__(self)
    self.sess = sess
    self.global_step_op = global_step_op
    self.start_at_global_step = start_at_global_step
    self.end_at_global_step = end_at_global_step

    self.start_time = 0
    self.start_step = 0
    self.finish_time = 0
    self.finish_step = 0

  def run(self):
    while self.finish_time == 0:
      time.sleep(.25)
      global_step_val, = self.sess.run([self.global_step_op])
      if self.start_time == 0 and global_step_val >= self.start_at_global_step:
        # Use tf.logging.info instead of log_fn, since print (which is log_fn)
        # is not thread safe and may interleave the outputs from two parallel
        # calls to print, which can break tests.
        tf.logging.info('Starting real work at step %s at time %s' % (
            global_step_val, time.ctime()))
        self.start_time = time.time()
        self.start_step = global_step_val
      if self.finish_time == 0 and global_step_val >= self.end_at_global_step:
        tf.logging.info('Finishing real work at step %s at time %s' % (
            global_step_val, time.ctime()))
        self.finish_time = time.time()
        self.finish_step = global_step_val

  def done(self):
    return self.finish_time > 0

  def steps_per_second(self):
    return ((self.finish_step - self.start_step) /
            (self.finish_time - self.start_time))


class CheckpointNotFoundException(Exception):
  pass


def get_data_type():
  return tf.float16 if FLAGS.use_fp16 else tf.float32


def loss_function(logits, labels, aux_logits):
  """Loss function."""
  with tf.name_scope('xentropy'):
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(
        logits=logits, labels=labels)
    loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  if aux_logits is not None:
    with tf.name_scope('aux_xentropy'):
      aux_cross_entropy = tf.losses.sparse_softmax_cross_entropy(
          logits=aux_logits, labels=labels)
      aux_loss = 0.4 * tf.reduce_mean(aux_cross_entropy, name='aux_loss')
      loss = tf.add_n([loss, aux_loss])
  return loss


def create_config_proto():
  """Returns session config proto."""
  config = tf.ConfigProto()
  config.allow_soft_placement = True
  config.intra_op_parallelism_threads = FLAGS.num_intra_threads
  config.inter_op_parallelism_threads = FLAGS.num_inter_threads
  config.gpu_options.force_gpu_compatible = FLAGS.force_gpu_compatible
  if FLAGS.gpu_memory_frac_for_testing > 0:
    config.gpu_options.per_process_gpu_memory_fraction = (
        FLAGS.gpu_memory_frac_for_testing)
  if FLAGS.xla:
    config.graph_options.optimizer_options.global_jit_level = (
        tf.OptimizerOptions.ON_1)
  return config


def get_mode_from_flags():
  """Determine which mode this script is running."""
  if FLAGS.forward_only and FLAGS.eval:
    raise ValueError('Only one of forward_only and eval flags is true')

  if FLAGS.eval:
    return 'evaluation'
  if FLAGS.forward_only:
    return 'forward-only'
  return 'training'


def benchmark_one_step(sess, fetches, step, batch_size, step_train_times,
                       trace_filename, image_producer, summary_op=None):
  """Advance one step of benchmarking."""
  if trace_filename is not None and step == -1:
    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    run_metadata = tf.RunMetadata()
  else:
    run_options = None
    run_metadata = None
  summary_str = None
  start_time = time.time()
  if summary_op is None:
    results = sess.run(fetches, options=run_options, run_metadata=run_metadata)
  else:
    (results, summary_str) = sess.run(
        [fetches, summary_op], options=run_options, run_metadata=run_metadata)

  if not FLAGS.forward_only:
    lossval = results['total_loss']
  else:
    lossval = 0.
  image_producer.notify_image_consumption()
  train_time = time.time() - start_time
  step_train_times.append(train_time)
  if step >= 0 and (step == 0 or (step + 1) % FLAGS.display_every == 0):
    log_str = '%i\t%s\t%.3f' % (
        step + 1, get_perf_timing_str(batch_size, step_train_times), lossval)
    if 'top_1_accuracy' in results:
      log_str += '\t%.3f\t%.3f' % (results['top_1_accuracy'],
                                   results['top_5_accuracy'])
    log_fn(log_str)
    sys.stdout.flush()
  if trace_filename is not None and step == -1:
    log_fn('Dumping trace to', trace_filename)
    trace = timeline.Timeline(step_stats=run_metadata.step_stats)
    with open(trace_filename, 'w') as trace_file:
      trace_file.write(trace.generate_chrome_trace_format(show_memory=True))
  return summary_str


def get_perf_timing_str(batch_size, step_train_times, scale=1):
  times = np.array(step_train_times)
  speeds = batch_size / times
  speed_mean = scale * batch_size / np.mean(times)
  if scale == 1:
    speed_uncertainty = np.std(speeds) / np.sqrt(float(len(speeds)))
    speed_madstd = 1.4826 * np.median(np.abs(speeds - np.median(speeds)))
    speed_jitter = speed_madstd
    return 'images/sec: %.1f +/- %.1f (jitter = %.1f)' % (
        speed_mean, speed_uncertainty, speed_jitter)
  else:
    return 'images/sec: %.1f' % speed_mean


def load_checkpoint(saver, sess, ckpt_dir):
  ckpt = tf.train.get_checkpoint_state(ckpt_dir)
  if ckpt and ckpt.model_checkpoint_path:
    if os.path.isabs(ckpt.model_checkpoint_path):
      # Restores from checkpoint with absolute path.
      model_checkpoint_path = ckpt.model_checkpoint_path
    else:
      # Restores from checkpoint with relative path.
      model_checkpoint_path = os.path.join(ckpt_dir, ckpt.model_checkpoint_path)
    # Assuming model_checkpoint_path looks something like:
    #   /my-favorite-path/imagenet_train/model.ckpt-0,
    # extract global_step from it.
    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    if not global_step.isdigit():
      global_step = 0
    else:
      global_step = int(global_step)
    saver.restore(sess, model_checkpoint_path)
    log_fn('Successfully loaded model from %s.' % ckpt.model_checkpoint_path)
    return global_step
  else:
    raise CheckpointNotFoundException('No checkpoint file found.')


class BenchmarkCNN(object):
  """Class for benchmarking a cnn network."""

  def __init__(self):
    self.dataset = datasets.create_dataset(FLAGS.data_dir, FLAGS.data_name)
    self.model = model_config.get_model_config(FLAGS.model, self.dataset)
    self.trace_filename = FLAGS.trace_file
    self.data_format = FLAGS.data_format
    self.num_batches = FLAGS.num_batches
    autotune_threshold = FLAGS.autotune_threshold if (
        FLAGS.autotune_threshold) else 1
    min_autotune_warmup = 5 * autotune_threshold * autotune_threshold
    self.num_warmup_batches = FLAGS.num_warmup_batches if (
        FLAGS.num_warmup_batches is not None) else max(10, min_autotune_warmup)
    self.graph_file = FLAGS.graph_file
    self.resize_method = FLAGS.resize_method
    self.sync_queue_counter = 0
    self.num_gpus = FLAGS.num_gpus
    self.use_synthetic_gpu_images = self.dataset.use_synthetic_gpu_images()

    if FLAGS.device == 'cpu' and FLAGS.data_format == 'NCHW':
      raise ValueError('--device=cpu requires that --data_format=NHWC')

    if ((FLAGS.num_epochs_per_decay or FLAGS.learning_rate_decay_factor) and
        not (FLAGS.learning_rate and FLAGS.num_epochs_per_decay and
             FLAGS.learning_rate_decay_factor)):
      raise ValueError('If one of --num_epochs_per_decay or '
                       '--learning_rate_decay_factor is set, both must be set'
                       'and --learning_rate must be set')
    if (FLAGS.minimum_learning_rate and
        not (FLAGS.learning_rate and FLAGS.num_epochs_per_decay and
             FLAGS.learning_rate_decay_factor)):
      raise ValueError('--minimum_learning_rate requires --learning_rate,'
                       '--num_epochs_per_decay, and '
                       '--learning_rate_decay_factor to be set')

    if (FLAGS.use_fp16 and FLAGS.fp16_vars and
        'replicated' in FLAGS.variable_update and FLAGS.use_nccl):
      raise ValueError('fp16 variables are not supported with NCCL')

    if FLAGS.use_tf_layers and FLAGS.use_fp16:
      raise ValueError('FLAGS.use_tf_layers and FLAGS.use_fp16 cannot both be '
                       'True, because tf.layers does not fully support fp16 '
                       'yet.')

    if FLAGS.use_tf_layers is None:
      self.use_tf_layers = not FLAGS.use_fp16
    else:
      self.use_tf_layers = FLAGS.use_tf_layers

    # Use the batch size from the command line if specified, otherwise use the
    # model's default batch size.  Scale the benchmark's batch size by the
    # number of GPUs.
    if FLAGS.batch_size > 0:
      self.model.set_batch_size(FLAGS.batch_size)
    self.batch_size = self.model.get_batch_size() * self.num_gpus
    self.batch_group_size = FLAGS.batch_group_size

    if FLAGS.use_fp16:
      self.loss_scale = (FLAGS.fp16_loss_scale or
                         self.model.get_fp16_loss_scale())
    else:
      self.loss_scale = 1.

    self.job_name = FLAGS.job_name  # "" for local training
    self.ps_hosts = FLAGS.ps_hosts.split(',')
    self.worker_hosts = FLAGS.worker_hosts.split(',')

    self.local_parameter_device_flag = FLAGS.local_parameter_device
    if self.job_name:
      self.task_index = FLAGS.task_index
      self.cluster = tf.train.ClusterSpec({'ps': self.ps_hosts,
                                           'worker': self.worker_hosts})
      self.server = None

      if not self.server:
        self.server = tf.train.Server(self.cluster, job_name=self.job_name,
                                      task_index=self.task_index,
                                      config=create_config_proto(),
                                      protocol=FLAGS.server_protocol)
      worker_prefix = '/job:worker/task:%s' % self.task_index
      self.param_server_device = tf.train.replica_device_setter(
          worker_device=worker_prefix + '/cpu:0', cluster=self.cluster)
      # This device on which the queues for managing synchronization between
      # servers should be stored.
      num_ps = len(self.ps_hosts)
      self.sync_queue_devices = ['/job:ps/task:%s/cpu:0' % i
                                 for i in range(num_ps)]
    else:
      self.task_index = 0
      self.cluster = None
      self.server = None
      worker_prefix = ''
      self.param_server_device = '/%s:0' % FLAGS.local_parameter_device
      self.sync_queue_devices = [self.param_server_device]

    # Device to use for ops that need to always run on the local worker's CPU.
    self.cpu_device = '%s/cpu:0' % worker_prefix

    # Device to use for ops that need to always run on the local worker's
    # compute device, and never on a parameter server device.
    self.raw_devices = ['%s/%s:%i' % (worker_prefix, FLAGS.device, i)
                        for i in xrange(self.num_gpus)]

    if FLAGS.staged_vars and FLAGS.variable_update != 'parameter_server':
      raise ValueError('staged_vars for now is only supported with '
                       '--variable_update=parameter_server')

    if FLAGS.variable_update == 'parameter_server':
      if self.job_name:
        if not FLAGS.staged_vars:
          self.variable_mgr = variable_mgr.VariableMgrDistributedFetchFromPS(
              self)
        else:
          self.variable_mgr = (
              variable_mgr.VariableMgrDistributedFetchFromStagedPS(self))
      else:
        if not FLAGS.staged_vars:
          self.variable_mgr = variable_mgr.VariableMgrLocalFetchFromPS(self)
        else:
          self.variable_mgr = variable_mgr.VariableMgrLocalFetchFromStagedPS(
              self)
    elif FLAGS.variable_update == 'replicated':
      if self.job_name:
        raise ValueError('Invalid --variable_update in distributed mode: %s' %
                         FLAGS.variable_update)
      self.variable_mgr = variable_mgr.VariableMgrLocalReplicated(
          self, FLAGS.use_nccl)
    elif FLAGS.variable_update == 'distributed_replicated':
      if not self.job_name:
        raise ValueError('Invalid --variable_update in local mode: %s' %
                         FLAGS.variable_update)
      self.variable_mgr = variable_mgr.VariableMgrDistributedReplicated(self)
    elif FLAGS.variable_update == 'independent':
      if self.job_name:
        raise ValueError('Invalid --variable_update in distributed mode: %s' %
                         FLAGS.variable_update)
      self.variable_mgr = variable_mgr.VariableMgrIndependent(self)
    else:
      raise ValueError('Invalid --variable_update: %s' % FLAGS.variable_update)

    # Device to use for running on the local worker's compute device, but
    # with variables assigned to parameter server devices.
    self.devices = self.variable_mgr.get_devices()
    if self.job_name:
      self.global_step_device = self.param_server_device
    else:
      self.global_step_device = self.cpu_device

    self.image_preprocessor = self.get_image_preprocessor()
    self.init_global_step = 0

  def print_info(self):
    """Print basic information."""
    log_fn('Model:       %s' % self.model.get_model())
    log_fn('Mode:        %s' % get_mode_from_flags())
    log_fn('Batch size:  %s global' % self.batch_size)
    log_fn('             %s per device' % (self.batch_size / len(self.devices)))
    if self.batch_group_size > 1:
      log_fn('             %d batches per prepocessing group' %
             self.batch_group_size)
    log_fn('Devices:     %s' % self.raw_devices)
    log_fn('Data format: %s' % self.data_format)
    log_fn('Optimizer:   %s' % FLAGS.optimizer)
    log_fn('Variables:   %s' % FLAGS.variable_update)
    log_fn('Data type:   %s' % get_data_type().name)
    if FLAGS.variable_update == 'replicated':
      log_fn('Use NCCL:    %s' % FLAGS.use_nccl)
    if self.job_name:
      log_fn('Sync:        %s' % FLAGS.cross_replica_sync)
    if FLAGS.staged_vars:
      log_fn('Staged vars: %s' % FLAGS.staged_vars)
    log_fn('==========')

  def run(self):
    if FLAGS.job_name == 'ps':
      log_fn('Running parameter server %s' % self.task_index)
      self.server.join()
      return

    with tf.Graph().as_default():
      if FLAGS.eval:
        self._eval_cnn()
      else:
        self._benchmark_cnn()

  def _eval_cnn(self):
    """Evaluate a model every FLAGS.eval_interval_secs."""
    (image_producer_ops, enqueue_ops, fetches) = self._build_model()
    saver = tf.train.Saver(self.variable_mgr.savable_variables())
    summary_writer = tf.summary.FileWriter(FLAGS.eval_dir,
                                           tf.get_default_graph())
    target = ''
    local_var_init_op = tf.local_variables_initializer()
    variable_mgr_init_ops = [local_var_init_op]
    with tf.control_dependencies([local_var_init_op]):
      variable_mgr_init_ops.extend(self.variable_mgr.get_post_init_ops())
    local_var_init_op_group = tf.group(*variable_mgr_init_ops)
    summary_op = tf.summary.merge_all()
    self._eval_once(
        saver, summary_writer, target, local_var_init_op_group,
        image_producer_ops, enqueue_ops, fetches, summary_op)

  def _eval_once(self, saver, summary_writer, target,
                 local_var_init_op_group, image_producer_ops, enqueue_ops,
                 fetches, summary_op):
    """Evaluate the model from a checkpoint using validation dataset."""
    with tf.Session(target=target, config=create_config_proto()) as sess:
      if FLAGS.checkpoint_dir is None:
        raise ValueError('Trained model directory not specified')
      try:
        global_step = load_checkpoint(saver, sess, FLAGS.checkpoint_dir)
      except CheckpointNotFoundException:
        log_fn('Checkpoint not found in %s' % FLAGS.checkpoint_dir)
        return
      sess.run(local_var_init_op_group)
      if self.dataset.queue_runner_required():
        tf.train.start_queue_runners(sess=sess)
      image_producer = cnn_util.ImageProducer(sess, image_producer_ops,
                                              self.batch_group_size)
      image_producer.start()
      for i in xrange(len(enqueue_ops)):
        sess.run(enqueue_ops[:(i+1)])
        image_producer.notify_image_consumption()

      start_time = time.time()
      top_1_accuracy_sum = 0.0
      top_5_accuracy_sum = 0.0
      total_eval_count = self.num_batches * self.batch_size
      for step in xrange(self.num_batches):
        if (FLAGS.save_summaries_steps > 0 and
            (step + 1) % FLAGS.save_summaries_steps == 0):
          results, summary_str = sess.run([fetches, summary_op])
          summary_writer.add_summary(summary_str)
        else:
          results = sess.run(fetches)
        top_1_accuracy_sum += results['top_1_accuracy']
        top_5_accuracy_sum += results['top_5_accuracy']
        if (step + 1) % FLAGS.display_every == 0:
          duration = time.time() - start_time
          examples_per_sec = self.batch_size * FLAGS.display_every / duration
          log_fn('%i\t%.1f examples/sec' % (step + 1, examples_per_sec))
          start_time = time.time()
        image_producer.notify_image_consumption()
      image_producer.done()
      precision_at_1 = top_1_accuracy_sum / self.num_batches
      recall_at_5 = top_5_accuracy_sum / self.num_batches
      summary = tf.Summary()
      summary.value.add(tag='eval/Accuracy@1', simple_value=precision_at_1)
      summary.value.add(tag='eval/Recall@5', simple_value=recall_at_5)
      summary_writer.add_summary(summary, global_step)
      log_fn('Precision @ 1 = %.4f, Recall @ 5 = %.4f, Global step = %d [%d examples]' %
             (precision_at_1, recall_at_5, global_step, total_eval_count))

  def _benchmark_cnn(self):
    """Run cnn in benchmark mode. When forward_only on, it forwards CNN."""
    (image_producer_ops, enqueue_ops, fetches) = self._build_model()
    fetches_list = nest.flatten(list(fetches.values()))
    main_fetch_group = tf.group(*fetches_list)
    execution_barrier = None
    if self.job_name and not FLAGS.cross_replica_sync:
      execution_barrier = self.add_sync_queues_and_barrier(
          'execution_barrier_', [])

    global_step = tf.contrib.framework.get_global_step()
    with tf.device(self.global_step_device):
      with tf.control_dependencies([main_fetch_group]):
        fetches['inc_global_step'] = global_step.assign_add(1)

    if self.job_name and FLAGS.cross_replica_sync:
      # Block all replicas until all replicas are ready for next step.
      fetches['sync_queues'] = self.add_sync_queues_and_barrier(
          'sync_queues_step_end_', [main_fetch_group])

    local_var_init_op = tf.local_variables_initializer()
    variable_mgr_init_ops = [local_var_init_op]
    with tf.control_dependencies([local_var_init_op]):
      variable_mgr_init_ops.extend(self.variable_mgr.get_post_init_ops())
    local_var_init_op_group = tf.group(*variable_mgr_init_ops)

    summary_op = tf.summary.merge_all()
    is_chief = (not self.job_name or self.task_index == 0)
    summary_writer = None
    if (is_chief and FLAGS.summary_verbosity and
        FLAGS.train_dir and
        FLAGS.save_summaries_steps > 0):
      summary_writer = tf.summary.FileWriter(FLAGS.train_dir,
                                             tf.get_default_graph())

    # We want to start the benchmark timer right after a image_producer barrier
    # and avoids undesired wating times on barriers.
    if ((self.num_warmup_batches + len(enqueue_ops) - 1) %
        self.batch_group_size) != 0:
      self.num_warmup_batches = int(
          math.ceil((self.num_warmup_batches + len(enqueue_ops) - 1.0) /
                    self.batch_group_size) * self.batch_group_size
          - len(enqueue_ops) + 1)
      log_fn('Round up warm up steps to %d to match batch_group_size' %
             self.num_warmup_batches)
      assert ((self.num_warmup_batches + len(enqueue_ops) - 1) %
              self.batch_group_size) == 0
    # We run the summaries in the same thread as the training operations by
    # passing in None for summary_op to avoid a summary_thread being started.
    # Running summaries and training operations in parallel could run out of
    # GPU memory.
    saver = tf.train.Saver(self.variable_mgr.savable_variables(), max_to_keep=10000)
    ready_for_local_init_op = None
    if self.job_name:
      # In distributed mode, we don't want to run local_var_init_op_group until
      # the global variables are initialized, because local_var_init_op_group
      # may use global variables (such as in distributed replicated mode). We
      # don't set this in non-distributed mode, because in non-distributed mode,
      # local_var_init_op_group may itself initialize global variables (such as
      # in replicated mode).
      ready_for_local_init_op = tf.report_uninitialized_variables(
          tf.global_variables())
    sv = tf.train.Supervisor(
        is_chief=is_chief,
        logdir=FLAGS.train_dir,
        ready_for_local_init_op=ready_for_local_init_op,
        local_init_op=local_var_init_op_group,
        saver=saver,
        global_step=global_step,
        summary_op=None,
        save_model_secs=None,
        summary_writer=summary_writer)

    step_train_times = []
    start_standard_services = (FLAGS.summary_verbosity > 0 or
                               self.dataset.queue_runner_required())
    with sv.managed_session(
        master=self.server.target if self.server else '',
        config=create_config_proto(),
        start_standard_services=start_standard_services) as sess:
      image_producer = cnn_util.ImageProducer(sess, image_producer_ops,
                                              self.batch_group_size)
      image_producer.start()
      for i in xrange(len(enqueue_ops)):
        sess.run(enqueue_ops[:(i+1)])
        image_producer.notify_image_consumption()
      self.init_global_step, = sess.run([global_step])
      global_step_watcher = GlobalStepWatcher(
          sess, global_step,
          len(self.worker_hosts) * self.num_warmup_batches +
          self.init_global_step,
          len(self.worker_hosts) * (
              self.num_warmup_batches + self.num_batches) - 1)
      global_step_watcher.start()

      if self.graph_file is not None:
        path, filename = os.path.split(self.graph_file)
        as_text = filename.endswith('txt')
        log_fn('Writing GraphDef as %s to %s' % (
            'text' if as_text else 'binary', self.graph_file))
        tf.train.write_graph(sess.graph_def, path, filename, as_text)

      log_fn('Running warm up')
      local_step = -1 * self.num_warmup_batches
      global_step_increment = 0
      if FLAGS.checkpoint_dir is not None:
        if not FLAGS.resume_training:
          subprocess.call("rm -rf %s; mkdir -p %s" % (FLAGS.checkpoint_dir,
                                                      FLAGS.checkpoint_dir), shell=True)
          f = open(os.path.join(FLAGS.checkpoint_dir, "times.log"), 'w')
        else:
          f = open(os.path.join(FLAGS.checkpoint_dir, "times.log"), 'a')
          checkpoints = [int(checkpoint) for checkpoint in os.listdir(FLAGS.checkpoint_dir)
                         if os.path.isdir(os.path.join(FLAGS.checkpoint_dir, checkpoint))]
          latest_checkpoint = ("%5d" % max(checkpoints)).replace(' ', '0')
          local_step = load_checkpoint(sv.saver, sess, os.path.join(FLAGS.checkpoint_dir, latest_checkpoint))
          global_step_increment = local_step  # TODO: Fix this for distributed settings.
      start_time = time.time()

      if FLAGS.cross_replica_sync and FLAGS.job_name:
        # In cross-replica sync mode, all workers must run the same number of
        # local steps, or else the workers running the extra step will block.
        done_fn = lambda: local_step == end_step
      else:
        done_fn = global_step_watcher.done
      subset = 'train'
      if FLAGS.subset is not None:
        subset = FLAGS.subset

      while not done_fn():
        num_minibatches_per_epoch = int(self.dataset.num_examples_per_epoch(subset) / self.batch_size)
        epoch = local_step / num_minibatches_per_epoch
        if local_step == 0:
          log_fn('Done warm up')
          if execution_barrier:
            log_fn('Waiting for other replicas to finish warm up')
            assert global_step_watcher.start_time == 0
            sess.run([execution_barrier])

          header_str = 'Step\tImg/sec\tloss'
          if FLAGS.print_training_accuracy or FLAGS.forward_only:
            header_str += '\ttop_1_accuracy\ttop_5_accuracy'
          log_fn(header_str)
          assert len(step_train_times) == self.num_warmup_batches
          step_train_times = []  # reset to ignore warm up batches
        if (summary_writer and
            (local_step + 1) % FLAGS.save_summaries_steps == 0):
          fetch_summary = summary_op
        else:
          fetch_summary = None
        summary_str = benchmark_one_step(
            sess, fetches, local_step, self.batch_size, step_train_times,
            self.trace_filename, image_producer, fetch_summary)
        if summary_str is not None and is_chief:
          sv.summary_computed(sess, summary_str)

        if (FLAGS.checkpoint_dir is not None and is_chief and
            ((local_step % num_minibatches_per_epoch) == 0)):
          end_time = time.time()
          directory = os.path.join(FLAGS.checkpoint_dir, ("%5d" % epoch).replace(' ', '0'))
          subprocess.call("mkdir -p %s" % directory, shell=True)
          checkpoint_path = os.path.join(directory, 'model.ckpt')
          f.write("Step: %d\tTime: %s\n" % (local_step + 1, end_time - start_time))
          f.close()
          f = open(os.path.join(FLAGS.checkpoint_dir, "times.log"), 'a')
          sv.saver.save(sess, checkpoint_path, global_step=local_step)
          log_fn("Saved checkpoint after %d epoch(s) to %s..." % (epoch, directory))
          sys.stdout.flush()
          start_time = time.time()
        local_step += 1

      # Waits for the global step to be done, regardless of done_fn.
      while not global_step_watcher.done():
        time.sleep(.25)
      images_per_sec = global_step_watcher.steps_per_second() * self.batch_size
      log_fn('-' * 64)
      log_fn('total images/sec: %.2f' % images_per_sec)
      log_fn('-' * 64)
      image_producer.done()
      if is_chief:
        store_benchmarks({'total_images_per_sec': images_per_sec})

      # Save the model checkpoint.
      if FLAGS.checkpoint_dir is not None and is_chief:
        end_time = time.time()
        directory = os.path.join(FLAGS.checkpoint_dir, ("%5d" % epoch).replace(' ', '0'))
        subprocess.call("mkdir -p %s" % directory, shell=True)
        checkpoint_path = os.path.join(directory, 'model.ckpt')
        f.write("Step: %d\tTime: %s\n" % (local_step, end_time - start_time))
        f.close()
        f = open(os.path.join(FLAGS.checkpoint_dir, "times.log"), 'a')
        sv.saver.save(sess, checkpoint_path, global_step=local_step)
        log_fn("Saved checkpoint after %d epoch(s) to %s..." % (epoch, directory))
        sys.stdout.flush()
        start_time = time.time()

      if execution_barrier:
        # Wait for other workers to reach the end, so this worker doesn't
        # go away underneath them.
        sess.run([execution_barrier])
    f.close()
    sv.stop()

  def _build_model(self):
    """Build the TensorFlow graph."""
    tf.set_random_seed(1234)
    np.random.seed(4321)
    phase_train = not (FLAGS.eval or FLAGS.forward_only)

    log_fn('Generating model')
    losses = []
    device_grads = []
    all_logits = []
    all_top_1_ops = []
    all_top_5_ops = []
    enqueue_ops = []
    gpu_compute_stage_ops = []
    gpu_grad_stage_ops = []

    with tf.device(self.global_step_device):
      global_step = tf.contrib.framework.get_or_create_global_step()

    # Build the processing and model for the worker.
    with tf.device(self.cpu_device):
      if FLAGS.eval:
        subset = 'validation'
      else:
        subset = 'train'
      image_producer_ops = []
      image_producer_stages = []
      images_splits, labels_splits = self.image_preprocessor.minibatch(
          self.dataset, subset=subset, use_data_sets=FLAGS.use_data_sets)
      images_shape = images_splits[0].get_shape()
      labels_shape = labels_splits[0].get_shape()
      for device_num in range(len(self.devices)):
        image_producer_stages.append(data_flow_ops.StagingArea(
            [tf.float32, tf.int32],
            shapes=[images_shape, labels_shape]))
        for group_index in xrange(self.batch_group_size):
          if not self.use_synthetic_gpu_images:
            batch_index = group_index + device_num * self.batch_group_size
            put_op = image_producer_stages[device_num].put(
                [images_splits[batch_index], labels_splits[batch_index]])
            image_producer_ops.append(put_op)

    image_producer_ops = tf.group(*image_producer_ops)
    update_ops = None
    staging_delta_ops = []

    for device_num in range(len(self.devices)):
      with self.variable_mgr.create_outer_variable_scope(
          device_num), tf.name_scope('tower_%i' % device_num) as name_scope:
        results = self.add_forward_pass_and_gradients(
            phase_train, device_num, image_producer_stages[device_num],
            gpu_compute_stage_ops, gpu_grad_stage_ops)
        if phase_train:
          losses.append(results['loss'])
          device_grads.append(results['gradvars'])
        else:
          all_logits.append(results['logits'])
        if not phase_train or FLAGS.print_training_accuracy:
          all_top_1_ops.append(results['top_1_op'])
          all_top_5_ops.append(results['top_5_op'])

        if device_num == 0:
          # Retain the Batch Normalization updates operations only from the
          # first tower. These operations update the moving mean and moving
          # variance variables, which are updated (but not used) during
          # training, and used during evaluation. The moving mean and variance
          # approximate the true mean and variance across all images in the
          # dataset. Therefore, in replicated mode, these moving averages would
          # be almost identical for each tower, and so we only update and save
          # the moving averages for one tower. In parameter server mode, all
          # towers share a copy of the variables so we also only need to update
          # and save the moving averages once.
          update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, name_scope)
          staging_delta_ops = list(self.variable_mgr.staging_delta_ops)

    if self.variable_mgr.supports_staged_vars():
      for staging_ops in self.variable_mgr.staging_vars_on_devices:
        gpu_compute_stage_ops.extend(
            [put_op for _, (put_op, _) in six.iteritems(staging_ops)])
    enqueue_ops.append(tf.group(*gpu_compute_stage_ops))
    if gpu_grad_stage_ops:
      staging_delta_ops += gpu_grad_stage_ops
    if staging_delta_ops:
      enqueue_ops.append(tf.group(*(staging_delta_ops)))

    fetches = {'enqueue_ops': enqueue_ops}  # The return value

    if all_top_1_ops:
      fetches['top_1_accuracy'] = tf.reduce_sum(all_top_1_ops) / self.batch_size
      if self.task_index == 0 and FLAGS.summary_verbosity > 0:
        tf.summary.scalar('top_1_accuracy', fetches['top_1_accuracy'])
    if all_top_5_ops:
      fetches['top_5_accuracy'] = tf.reduce_sum(all_top_5_ops) / self.batch_size
      if self.task_index == 0 and FLAGS.summary_verbosity > 0:
        tf.summary.scalar('top_5_accuracy', fetches['top_5_accuracy'])

    if not phase_train:
      if FLAGS.forward_only:
        fetches['all_logits'] = tf.concat(all_logits, 0)
      return (image_producer_ops, enqueue_ops, fetches)
    extra_nccl_ops = []
    apply_gradient_devices, gradient_state = (
        self.variable_mgr.preprocess_device_grads(device_grads))

    training_ops = []
    for d, device in enumerate(apply_gradient_devices):
      with tf.device(device):
        total_loss = tf.reduce_mean(losses)
        avg_grads = self.variable_mgr.get_gradients_to_apply(d, gradient_state)

        gradient_clip = FLAGS.gradient_clip
        learning_rate = FLAGS.learning_rate or self.model.get_learning_rate(
            global_step, self.batch_size)
        if ((not self.use_synthetic_gpu_images) and
            FLAGS.learning_rate and
            FLAGS.num_epochs_per_decay > 0 and
            FLAGS.learning_rate_decay_factor > 0):
          num_batches_per_epoch = (
              float(self.dataset.num_examples_per_epoch()) / self.batch_size)
          decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)

          # Decay the learning rate exponentially based on the number of steps.
          learning_rate = tf.train.exponential_decay(
              FLAGS.learning_rate, global_step,
              decay_steps, FLAGS.learning_rate_decay_factor, staircase=True)

          if FLAGS.minimum_learning_rate != 0.:
            learning_rate = tf.maximum(learning_rate,
                                       FLAGS.minimum_learning_rate)

        if gradient_clip is not None:
          clipped_grads = [
              (tf.clip_by_value(grad, -gradient_clip, +gradient_clip), var)
              for grad, var in avg_grads
          ]
        else:
          clipped_grads = avg_grads

        learning_rate = tf.identity(learning_rate, name='learning_rate')
        if FLAGS.optimizer == 'momentum':
          opt = tf.train.MomentumOptimizer(
              learning_rate, FLAGS.momentum, use_nesterov=True)
        elif FLAGS.optimizer == 'sgd':
          opt = tf.train.GradientDescentOptimizer(learning_rate)
        elif FLAGS.optimizer == 'rmsprop':
          opt = tf.train.RMSPropOptimizer(learning_rate, FLAGS.rmsprop_decay,
                                          momentum=FLAGS.rmsprop_momentum,
                                          epsilon=FLAGS.rmsprop_epsilon)
        elif FLAGS.optimizer == 'adam':
          opt = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=FLAGS.adam_beta1,
                                       beta2=FLAGS.adam_beta2,
                                       epsilon=FLAGS.adam_epsilon)
        else:
          raise ValueError('Optimizer "%s" was not recognized', FLAGS.optimizer)

        self.variable_mgr.append_apply_gradients_ops(
            gradient_state, opt, clipped_grads, training_ops)
    train_op = tf.group(*(training_ops + update_ops + extra_nccl_ops))

    with tf.device(self.cpu_device):
      if self.task_index == 0 and FLAGS.summary_verbosity > 0:
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('total_loss', total_loss)
        if FLAGS.summary_verbosity >= 2:
          for grad, var in avg_grads:
            if grad is not None:
              tf.summary.histogram(var.op.name + '/gradients', grad)
          for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
    fetches['train_op'] = train_op
    fetches['total_loss'] = total_loss
    return (image_producer_ops, enqueue_ops, fetches)

  def add_forward_pass_and_gradients(
      self, phase_train, device_num, image_producer_stage,
      gpu_compute_stage_ops, gpu_grad_stage_ops):
    """Add ops for forward-pass and gradient computations."""
    nclass = self.dataset.num_classes + 1
    input_data_type = get_data_type()
    data_type = get_data_type()
    if not self.use_synthetic_gpu_images:
      with tf.device(self.cpu_device):
        host_images, host_labels = image_producer_stage.get()
        images_shape = host_images.get_shape()
        labels_shape = host_labels.get_shape()
    with tf.device(self.raw_devices[device_num]):
      if not self.use_synthetic_gpu_images:
        gpu_compute_stage = data_flow_ops.StagingArea(
            [host_images.dtype, host_labels.dtype],
            shapes=[images_shape, labels_shape]
        )
        # The CPU-to-GPU copy is triggered here.
        gpu_compute_stage_op = gpu_compute_stage.put(
            [host_images, host_labels])
        images, labels = gpu_compute_stage.get()
        images = tf.reshape(images, shape=images_shape)
        gpu_compute_stage_ops.append(gpu_compute_stage_op)
      else:
        # Minor hack to avoid H2D copy when using synthetic data
        image_size = self.model.get_image_size()
        image_shape = [self.batch_size // self.num_gpus, image_size, image_size,
                       self.dataset.depth]
        labels_shape = [self.batch_size // self.num_gpus]
        images = tf.truncated_normal(
            image_shape, dtype=input_data_type,
            stddev=1e-1, name='synthetic_images')
        images = tf.contrib.framework.local_variable(
            images, name='gpu_cached_images')
        labels = tf.random_uniform(
            labels_shape, minval=0, maxval=nclass-1,
            dtype=tf.int32, name='synthetic_labels')

    with tf.device(self.devices[device_num]):
      # Rescale from [0, 255] to [0, 2]
      images = tf.multiply(images, 1./127.5)
      # Rescale to [-1, 1]
      images = tf.subtract(images, 1.0)

      if self.data_format == 'NCHW':
        images = tf.transpose(images, [0, 3, 1, 2])
      if input_data_type != data_type:
        images = tf.cast(images, data_type)
      var_type = tf.float32
      if data_type == tf.float16 and FLAGS.fp16_vars:
        var_type = tf.float16
      network = convnet_builder.ConvNetBuilder(images, self.dataset.depth,
                                               phase_train, self.use_tf_layers,
                                               self.data_format, data_type,
                                               var_type)
      self.model.add_inference(network)
      # Add the final fully-connected class layer
      logits = network.affine(nclass, activation='linear')
      aux_logits = None
      if network.aux_top_layer is not None:
        with network.switch_to_aux_top_layer():
          aux_logits = network.affine(nclass, activation='linear', stddev=0.001)
      if data_type == tf.float16:
        # TODO(reedwm): Determine if we should do this cast here.
        logits = tf.cast(logits, tf.float32)
        if aux_logits is not None:
          aux_logits = tf.cast(aux_logits, tf.float32)

      results = {}  # The return value
      if not phase_train or FLAGS.print_training_accuracy:
        top_1_op = tf.reduce_sum(
            tf.cast(tf.nn.in_top_k(logits, labels, 1), data_type))
        top_5_op = tf.reduce_sum(
            tf.cast(tf.nn.in_top_k(logits, labels, 5), data_type))
        results['top_1_op'] = top_1_op
        results['top_5_op'] = top_5_op

      if not phase_train:
        results['logits'] = logits
        return results
      loss = loss_function(logits, labels, aux_logits=aux_logits)
      params = self.variable_mgr.trainable_variables_on_device(device_num)
      if data_type == tf.float16 and FLAGS.fp16_vars:
        # fp16 reductions are very slow on GPUs, so cast to fp32 before calling
        # tf.nn.l2_loss and tf.add_n.
        # TODO(b/36217816): Once the bug is fixed, investigate if we should do
        # this reduction in fp16.
        fp32_params = (tf.cast(p, tf.float32) for p in params)
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in fp32_params])
      else:
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in params])
      weight_decay = FLAGS.weight_decay
      if weight_decay is not None and weight_decay != 0.:
        loss += weight_decay * l2_loss

      aggmeth = tf.AggregationMethod.DEFAULT
      scaled_loss = loss if self.loss_scale == 1. else loss * self.loss_scale
      grads = tf.gradients(scaled_loss, params, aggregation_method=aggmeth)
      if self.loss_scale != 1.:
        # TODO(reedwm): We could avoid these multiplications by directly
        # modifying the learning rate instead. If this is done, care must be
        # taken to ensure that this scaling method is correct, as some
        # optimizers square gradients and do other operations which might not be
        # compatible with modifying both the gradients and the learning rate.
        grads = [grad * (1. / self.loss_scale) for grad in grads]

      if FLAGS.staged_vars:
        grad_dtypes = [grad.dtype for grad in grads]
        grad_shapes = [grad.shape for grad in grads]
        grad_stage = data_flow_ops.StagingArea(grad_dtypes, grad_shapes)
        grad_stage_op = grad_stage.put(grads)
        # In general, this decouples the computation of the gradients and
        # the updates of the weights.
        # During the pipeline warm up, this runs enough training to produce
        # the first set of gradients.
        gpu_grad_stage_ops.append(grad_stage_op)
        grads = grad_stage.get()

      param_refs = self.variable_mgr.trainable_variables_on_device(
          device_num, writable=True)
      gradvars = list(zip(grads, param_refs))
      results['loss'] = loss
      results['gradvars'] = gradvars
      return results

  def get_image_preprocessor(self):
    """Returns the image preprocessor to used, based on the model.

    Returns:
      The image preprocessor, or None if synthetic data should be used.
    """
    image_size = self.model.get_image_size()
    input_data_type = get_data_type()

    shift_ratio = 0
    if self.job_name:
      # shift_ratio prevents multiple workers from processing the same batch
      # during a step
      assert self.worker_hosts
      shift_ratio = float(self.task_index) / len(self.worker_hosts)

    processor_class = self.dataset.get_image_preprocessor()
    assert processor_class
    return processor_class(
        image_size, image_size, self.batch_size * self.batch_group_size,
        len(self.devices) * self.batch_group_size,
        dtype=input_data_type,
        train=(not FLAGS.eval), distortions=FLAGS.distortions,
        resize_method=self.resize_method, shift_ratio=shift_ratio)

  def add_sync_queues_and_barrier(self, name_prefix,
                                  enqueue_after_list):
    """Adds ops to enqueue on all worker queues.

    Args:
      name_prefix: prefixed for the shared_name of ops.
      enqueue_after_list: control dependency from ops.

    Returns:
      an op that should be used as control dependency before starting next step.
    """
    self.sync_queue_counter += 1
    num_workers = self.cluster.num_tasks('worker')
    with tf.device(self.sync_queue_devices[
        self.sync_queue_counter % len(self.sync_queue_devices)]):
      sync_queues = [
          tf.FIFOQueue(num_workers, [tf.bool], shapes=[[]],
                       shared_name='%s%s' % (name_prefix, i))
          for i in range(num_workers)]
      queue_ops = []
      # For each other worker, add an entry in a queue, signaling that it can
      # finish this step.
      token = tf.constant(False)
      with tf.control_dependencies(enqueue_after_list):
        for i, q in enumerate(sync_queues):
          if i == self.task_index:
            queue_ops.append(tf.no_op())
          else:
            queue_ops.append(q.enqueue(token))

      # Drain tokens off queue for this worker, one for each other worker.
      queue_ops.append(
          sync_queues[self.task_index].dequeue_many(len(sync_queues) - 1))

      return tf.group(*queue_ops)


def store_benchmarks(names_to_values):
  if FLAGS.result_storage:
    benchmark_storage.store_benchmark(names_to_values, FLAGS.result_storage)


def setup():
  """Sets up the environment that BenchmarkCNN should run in."""
  if FLAGS.winograd_nonfused:
    os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
  else:
    os.environ.pop('TF_ENABLE_WINOGRAD_NONFUSED', None)
  if FLAGS.autotune_threshold:
    os.environ['TF_AUTOTUNE_THRESHOLD'] = str(FLAGS.autotune_threshold)
  os.environ['TF_SYNC_ON_FINISH'] = str(int(FLAGS.sync_on_finish))
  argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # Sets environment variables for MKL
  if FLAGS.mkl:
    os.environ['KMP_BLOCKTIME'] = str(FLAGS.kmp_blocktime)
    os.environ['KMP_SETTINGS'] = str(FLAGS.kmp_settings)
    os.environ['KMP_AFFINITY'] = FLAGS.kmp_affinity
    if FLAGS.num_intra_threads > 0:
      os.environ['OMP_NUM_THREADS'] = str(FLAGS.num_intra_threads)



def main(_):
  setup()
  bench = BenchmarkCNN()

  tfversion = cnn_util.tensorflow_version_tuple()
  log_fn('TensorFlow:  %i.%i' % (tfversion[0], tfversion[1]))

  bench.print_info()
  bench.run()


if __name__ == '__main__':
  tf.app.run()
