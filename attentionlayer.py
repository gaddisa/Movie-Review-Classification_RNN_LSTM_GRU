
"""
####################################################################################
    The following codes implement BahdanauAttention
    for our encoder decoder architecture
# ==============================================================================
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import math

import numpy as np

from tensorflow.contrib.framework.python.framework import tensor_util
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.layers import base as layers_base
from tensorflow.python.layers import core as layers_core
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest


__all__ = [
    "AttentionMechanism",
    "AttentionWrapper",
    "AttentionWrapperState",
    "LuongAttention",
    "BahdanauAttention",
    "hardmax",
    "safe_cumprod",
    "monotonic_attention",
    "BahdanauMonotonicAttention",
    "LuongMonotonicAttention",
]


_zero_state_tensors = rnn_cell_impl._zero_state_tensors  # pylint: disable=protected-access


class AttentionMechanism(object):

  @property
  def alignments_size(self):
    raise NotImplementedError

  @property
  def state_size(self):
    raise NotImplementedError


def _prepare_memory(memory, memory_sequence_length, check_inner_dims_defined):
  
  memory = nest.map_structure(
      lambda m: ops.convert_to_tensor(m, name="memory"), memory)
  if memory_sequence_length is not None:
    memory_sequence_length = ops.convert_to_tensor(
        memory_sequence_length, name="memory_sequence_length")
  if check_inner_dims_defined:
    def _check_dims(m):
      if not m.get_shape()[2:].is_fully_defined():
        raise ValueError("Expected memory %s to have fully defined inner dims, "
                         "but saw shape: %s" % (m.name, m.get_shape()))
    nest.map_structure(_check_dims, memory)
  if memory_sequence_length is None:
    seq_len_mask = None
  else:
    seq_len_mask = array_ops.sequence_mask(
        memory_sequence_length,
        maxlen=array_ops.shape(nest.flatten(memory)[0])[1],
        dtype=nest.flatten(memory)[0].dtype)
    seq_len_batch_size = (
        tensor_shape.dimension_value(memory_sequence_length.shape[0])
        or array_ops.shape(memory_sequence_length)[0])
  def _maybe_mask(m, seq_len_mask):
    rank = m.get_shape().ndims
    rank = rank if rank is not None else array_ops.rank(m)
    extra_ones = array_ops.ones(rank - 2, dtype=dtypes.int32)
    m_batch_size = tensor_shape.dimension_value(
        m.shape[0]) or array_ops.shape(m)[0]
    if memory_sequence_length is not None:
      message = ("memory_sequence_length and memory tensor batch sizes do not "
                 "match.")
      with ops.control_dependencies([
          check_ops.assert_equal(
              seq_len_batch_size, m_batch_size, message=message)]):
        seq_len_mask = array_ops.reshape(
            seq_len_mask,
            array_ops.concat((array_ops.shape(seq_len_mask), extra_ones), 0))
        return m * seq_len_mask
    else:
      return m
  return nest.map_structure(lambda m: _maybe_mask(m, seq_len_mask), memory)


def _maybe_mask_score(score, memory_sequence_length, score_mask_value):
  if memory_sequence_length is None:
    return score
  message = ("All values in memory_sequence_length must greater than zero.")
  with ops.control_dependencies(
      [check_ops.assert_positive(memory_sequence_length, message=message)]):
    score_mask = array_ops.sequence_mask(
        memory_sequence_length, maxlen=array_ops.shape(score)[1])
    score_mask_values = score_mask_value * array_ops.ones_like(score)
    return array_ops.where(score_mask, score, score_mask_values)


class _BaseAttentionMechanism(AttentionMechanism):
  """A base AttentionMechanism class providing common functionality.
  Common functionality includes:
    1. Storing the query and memory layers.
    2. Preprocessing and storing the memory.
  """

  def __init__(self,
               query_layer,
               memory,
               probability_fn,
               memory_sequence_length=None,
               memory_layer=None,
               check_inner_dims_defined=True,
               score_mask_value=None,
               name=None):
     
    if (query_layer is not None
        and not isinstance(query_layer, layers_base.Layer)):
      raise TypeError(
          "query_layer is not a Layer: %s" % type(query_layer).__name__)
    if (memory_layer is not None
        and not isinstance(memory_layer, layers_base.Layer)):
      raise TypeError(
          "memory_layer is not a Layer: %s" % type(memory_layer).__name__)
    self._query_layer = query_layer
    self._memory_layer = memory_layer
    self.dtype = memory_layer.dtype
    if not callable(probability_fn):
      raise TypeError("probability_fn must be callable, saw type: %s" %
                      type(probability_fn).__name__)
    if score_mask_value is None:
      score_mask_value = dtypes.as_dtype(
          self._memory_layer.dtype).as_numpy_dtype(-np.inf)
    self._probability_fn = lambda score, prev: (  # pylint:disable=g-long-lambda
        probability_fn(
            _maybe_mask_score(score, memory_sequence_length, score_mask_value),
            prev))
    with ops.name_scope(
        name, "BaseAttentionMechanismInit", nest.flatten(memory)):
      self._values = _prepare_memory(
          memory, memory_sequence_length,
          check_inner_dims_defined=check_inner_dims_defined)
      self._keys = (
          self.memory_layer(self._values) if self.memory_layer  # pylint: disable=not-callable
          else self._values)
      self._batch_size = (
          tensor_shape.dimension_value(self._keys.shape[0]) or
          array_ops.shape(self._keys)[0])
      self._alignments_size = (tensor_shape.dimension_value(self._keys.shape[1])
                               or array_ops.shape(self._keys)[1])

  @property
  def memory_layer(self):
    return self._memory_layer

  @property
  def query_layer(self):
    return self._query_layer

  @property
  def values(self):
    return self._values

  @property
  def keys(self):
    return self._keys

  @property
  def batch_size(self):
    return self._batch_size

  @property
  def alignments_size(self):
    return self._alignments_size

  @property
  def state_size(self):
    return self._alignments_size

  def initial_alignments(self, batch_size, dtype):
    
    max_time = self._alignments_size
    return _zero_state_tensors(max_time, batch_size, dtype)

  def initial_state(self, batch_size, dtype):
    
    return self.initial_alignments(batch_size, dtype)


def _luong_score(query, keys, scale):
  
  depth = query.get_shape()[-1]
  key_units = keys.get_shape()[-1]
  if depth != key_units:
    raise ValueError(
        "Incompatible or unknown inner dimensions between query and keys.  "
        "Query (%s) has units: %s.  Keys (%s) have units: %s.  "
        "Perhaps you need to set num_units to the keys' dimension (%s)?"
        % (query, depth, keys, key_units, key_units))
  dtype = query.dtype

  # Reshape from [batch_size, depth] to [batch_size, 1, depth]
  # for matmul.
  query = array_ops.expand_dims(query, 1)

  
  score = math_ops.matmul(query, keys, transpose_b=True)
  score = array_ops.squeeze(score, [1])

  if scale:
    # Scalar used in weight scaling
    g = variable_scope.get_variable(
        "attention_g", dtype=dtype,
        initializer=init_ops.ones_initializer, shape=())
    score = g * score
  return score


class LuongAttention(_BaseAttentionMechanism):
  

  def __init__(self,
               num_units,
               memory,
               memory_sequence_length=None,
               scale=False,
               probability_fn=None,
               score_mask_value=None,
               dtype=None,
               name="LuongAttention"):
    
    # For LuongAttention, we only transform the memory layer; thus
    # num_units **must** match expected the query depth.
    if probability_fn is None:
      probability_fn = nn_ops.softmax
    if dtype is None:
      dtype = dtypes.float32
    wrapped_probability_fn = lambda score, _: probability_fn(score)
    super(LuongAttention, self).__init__(
        query_layer=None,
        memory_layer=layers_core.Dense(
            num_units, name="memory_layer", use_bias=False, dtype=dtype),
        memory=memory,
        probability_fn=wrapped_probability_fn,
        memory_sequence_length=memory_sequence_length,
        score_mask_value=score_mask_value,
        name=name)
    self._num_units = num_units
    self._scale = scale
    self._name = name

  def __call__(self, query, state):
    
    with variable_scope.variable_scope(None, "luong_attention", [query]):
      score = _luong_score(query, self._keys, self._scale)
    alignments = self._probability_fn(score, state)
    next_state = alignments
    return alignments, next_state


def _bahdanau_score(processed_query, keys, normalize):
  
  dtype = processed_query.dtype
  # Get the number of hidden units from the trailing dimension of keys
  num_units = tensor_shape.dimension_value(
      keys.shape[2]) or array_ops.shape(keys)[2]
  # Reshape from [batch_size, ...] to [batch_size, 1, ...] for broadcasting.
  processed_query = array_ops.expand_dims(processed_query, 1)
  v = variable_scope.get_variable(
      "attention_v", [num_units], dtype=dtype)
  if normalize:
    # Scalar used in weight normalization
    g = variable_scope.get_variable(
        "attention_g", dtype=dtype,
        initializer=init_ops.constant_initializer(math.sqrt((1. / num_units))),
        shape=())
    # Bias added prior to the nonlinearity
    b = variable_scope.get_variable(
        "attention_b", [num_units], dtype=dtype,
        initializer=init_ops.zeros_initializer())
    # normed_v = g * v / ||v||
    normed_v = g * v * math_ops.rsqrt(
        math_ops.reduce_sum(math_ops.square(v)))
    return math_ops.reduce_sum(
        normed_v * math_ops.tanh(keys + processed_query + b), [2])
  else:
    return math_ops.reduce_sum(v * math_ops.tanh(keys + processed_query), [2])


class BahdanauAttention(_BaseAttentionMechanism):


  def __init__(self,
               num_units,
               memory,
               memory_sequence_length=None,
               normalize=False,
               probability_fn=None,
               score_mask_value=None,
               dtype=None,
               name="BahdanauAttention"):

    if probability_fn is None:
      probability_fn = nn_ops.softmax
    if dtype is None:
      dtype = dtypes.float32
    wrapped_probability_fn = lambda score, _: probability_fn(score)
    super(BahdanauAttention, self).__init__(
        query_layer=layers_core.Dense(
            num_units, name="query_layer", use_bias=False, dtype=dtype),
        memory_layer=layers_core.Dense(
            num_units, name="memory_layer", use_bias=False, dtype=dtype),
        memory=memory,
        probability_fn=wrapped_probability_fn,
        memory_sequence_length=memory_sequence_length,
        score_mask_value=score_mask_value,
        name=name)
    self._num_units = num_units
    self._normalize = normalize
    self._name = name

  def __call__(self, query, state):
   
    with variable_scope.variable_scope(None, "bahdanau_attention", [query]):
      processed_query = self.query_layer(query) if self.query_layer else query
      score = _bahdanau_score(processed_query, self._keys, self._normalize)
    alignments = self._probability_fn(score, state)
    next_state = alignments
    return alignments, next_state


def safe_cumprod(x, *args, **kwargs):
  
  with ops.name_scope(None, "SafeCumprod", [x]):
    x = ops.convert_to_tensor(x, name="x")
    tiny = np.finfo(x.dtype.as_numpy_dtype).tiny
    return math_ops.exp(math_ops.cumsum(
        math_ops.log(clip_ops.clip_by_value(x, tiny, 1)), *args, **kwargs))


def monotonic_attention(p_choose_i, previous_attention, mode):
  
  # Force things to be tensors
  p_choose_i = ops.convert_to_tensor(p_choose_i, name="p_choose_i")
  previous_attention = ops.convert_to_tensor(
      previous_attention, name="previous_attention")
  if mode == "recursive":
    # Use .shape[0] when it's not None, or fall back on symbolic shape
    batch_size = tensor_shape.dimension_value(
        p_choose_i.shape[0]) or array_ops.shape(p_choose_i)[0]
    # Compute [1, 1 - p_choose_i[0], 1 - p_choose_i[1], ..., 1 - p_choose_i[-2]]
    shifted_1mp_choose_i = array_ops.concat(
        [array_ops.ones((batch_size, 1)), 1 - p_choose_i[:, :-1]], 1)
    # Compute attention distribution recursively as
    # q[i] = (1 - p_choose_i[i - 1])*q[i - 1] + previous_attention[i]
    # attention[i] = p_choose_i[i]*q[i]
    attention = p_choose_i*array_ops.transpose(functional_ops.scan(
        # Need to use reshape to remind TF of the shape between loop iterations
        lambda x, yz: array_ops.reshape(yz[0]*x + yz[1], (batch_size,)),
        # Loop variables yz[0] and yz[1]
        [array_ops.transpose(shifted_1mp_choose_i),
         array_ops.transpose(previous_attention)],
        # Initial value of x is just zeros
        array_ops.zeros((batch_size,))))
  elif mode == "parallel":
    # safe_cumprod computes cumprod in logspace with numeric checks
    cumprod_1mp_choose_i = safe_cumprod(1 - p_choose_i, axis=1, exclusive=True)
    # Compute recurrence relation solution
    attention = p_choose_i*cumprod_1mp_choose_i*math_ops.cumsum(
        previous_attention /
        # Clip cumprod_1mp to avoid divide-by-zero
        clip_ops.clip_by_value(cumprod_1mp_choose_i, 1e-10, 1.), axis=1)
  elif mode == "hard":
    # Remove any probabilities before the index chosen last time step
    p_choose_i *= math_ops.cumsum(previous_attention, axis=1)
    # Now, use exclusive cumprod to remove probabilities after the first
    # chosen index, like so:
    # p_choose_i = [0, 0, 0, 1, 1, 0, 1, 1]
    # cumprod(1 - p_choose_i, exclusive=True) = [1, 1, 1, 1, 0, 0, 0, 0]
    # Product of above: [0, 0, 0, 1, 0, 0, 0, 0]
    attention = p_choose_i*math_ops.cumprod(
        1 - p_choose_i, axis=1, exclusive=True)
  else:
    raise ValueError("mode must be 'recursive', 'parallel', or 'hard'.")
  return attention


def _monotonic_probability_fn(score, previous_alignments, sigmoid_noise, mode,
                              seed=None):
  
  # Optionally add pre-sigmoid noise to the scores
  if sigmoid_noise > 0:
    noise = random_ops.random_normal(array_ops.shape(score), dtype=score.dtype,
                                     seed=seed)
    score += sigmoid_noise*noise
  # Compute "choosing" probabilities from the attention scores
  if mode == "hard":
    # When mode is hard, use a hard sigmoid
    p_choose_i = math_ops.cast(score > 0, score.dtype)
  else:
    p_choose_i = math_ops.sigmoid(score)
  # Convert from choosing probabilities to attention distribution
  return monotonic_attention(p_choose_i, previous_alignments, mode)


class _BaseMonotonicAttentionMechanism(_BaseAttentionMechanism):
 

  def initial_alignments(self, batch_size, dtype):
    
    max_time = self._alignments_size
    return array_ops.one_hot(
        array_ops.zeros((batch_size,), dtype=dtypes.int32), max_time,
        dtype=dtype)


class BahdanauMonotonicAttention(_BaseMonotonicAttentionMechanism):
 

  def __init__(self,
               num_units,
               memory,
               memory_sequence_length=None,
               normalize=False,
               score_mask_value=None,
               sigmoid_noise=0.,
               sigmoid_noise_seed=None,
               score_bias_init=0.,
               mode="parallel",
               dtype=None,
               name="BahdanauMonotonicAttention"):
     
    # Set up the monotonic probability fn with supplied parameters
    if dtype is None:
      dtype = dtypes.float32
    wrapped_probability_fn = functools.partial(
        _monotonic_probability_fn, sigmoid_noise=sigmoid_noise, mode=mode,
        seed=sigmoid_noise_seed)
    super(BahdanauMonotonicAttention, self).__init__(
        query_layer=layers_core.Dense(
            num_units, name="query_layer", use_bias=False, dtype=dtype),
        memory_layer=layers_core.Dense(
            num_units, name="memory_layer", use_bias=False, dtype=dtype),
        memory=memory,
        probability_fn=wrapped_probability_fn,
        memory_sequence_length=memory_sequence_length,
        score_mask_value=score_mask_value,
        name=name)
    self._num_units = num_units
    self._normalize = normalize
    self._name = name
    self._score_bias_init = score_bias_init

  def __call__(self, query, state):
    
    with variable_scope.variable_scope(
        None, "bahdanau_monotonic_attention", [query]):
      processed_query = self.query_layer(query) if self.query_layer else query
      score = _bahdanau_score(processed_query, self._keys, self._normalize)
      score_bias = variable_scope.get_variable(
          "attention_score_bias", dtype=processed_query.dtype,
          initializer=self._score_bias_init)
      score += score_bias
    alignments = self._probability_fn(score, state)
    next_state = alignments
    return alignments, next_state


class LuongMonotonicAttention(_BaseMonotonicAttentionMechanism):
 

  def __init__(self,
               num_units,
               memory,
               memory_sequence_length=None,
               scale=False,
               score_mask_value=None,
               sigmoid_noise=0.,
               sigmoid_noise_seed=None,
               score_bias_init=0.,
               mode="parallel",
               dtype=None,
               name="LuongMonotonicAttention"):
    
    # Set up the monotonic probability fn with supplied parameters
    if dtype is None:
      dtype = dtypes.float32
    wrapped_probability_fn = functools.partial(
        _monotonic_probability_fn, sigmoid_noise=sigmoid_noise, mode=mode,
        seed=sigmoid_noise_seed)
    super(LuongMonotonicAttention, self).__init__(
        query_layer=None,
        memory_layer=layers_core.Dense(
            num_units, name="memory_layer", use_bias=False, dtype=dtype),
        memory=memory,
        probability_fn=wrapped_probability_fn,
        memory_sequence_length=memory_sequence_length,
        score_mask_value=score_mask_value,
        name=name)
    self._num_units = num_units
    self._scale = scale
    self._score_bias_init = score_bias_init
    self._name = name

  def __call__(self, query, state):
    
    with variable_scope.variable_scope(None, "luong_monotonic_attention",
                                       [query]):
      score = _luong_score(query, self._keys, self._scale)
      score_bias = variable_scope.get_variable(
          "attention_score_bias", dtype=query.dtype,
          initializer=self._score_bias_init)
      score += score_bias
    alignments = self._probability_fn(score, state)
    next_state = alignments
    return alignments, next_state


class AttentionWrapperState(
    collections.namedtuple("AttentionWrapperState",
                           ("cell_state", "attention", "time", "alignments",
                            "alignment_history", "attention_state"))):
   

  def clone(self, **kwargs):
     
    def with_same_shape(old, new):
      """Check and set new tensor's shape."""
      if isinstance(old, ops.Tensor) and isinstance(new, ops.Tensor):
        return tensor_util.with_same_shape(old, new)
      return new

    return nest.map_structure(
        with_same_shape,
        self,
        super(AttentionWrapperState, self)._replace(**kwargs))


def hardmax(logits, name=None):
  """Returns batched one-hot vectors.
  The depth index containing the `1` is that of the maximum logit value.
  Args:
    logits: A batch tensor of logit values.
    name: Name to use when creating ops.
  Returns:
    A batched one-hot tensor.
  """
  with ops.name_scope(name, "Hardmax", [logits]):
    logits = ops.convert_to_tensor(logits, name="logits")
    if tensor_shape.dimension_value(logits.get_shape()[-1]) is not None:
      depth = tensor_shape.dimension_value(logits.get_shape()[-1])
    else:
      depth = array_ops.shape(logits)[-1]
    return array_ops.one_hot(
        math_ops.argmax(logits, -1), depth, dtype=logits.dtype)


def _compute_attention(attention_mechanism, cell_output, attention_state,
                       attention_layer):
  """Computes the attention and alignments for a given attention_mechanism."""
  alignments, next_attention_state = attention_mechanism(
      cell_output, state=attention_state)

  # Reshape from [batch_size, memory_time] to [batch_size, 1, memory_time]
  expanded_alignments = array_ops.expand_dims(alignments, 1)
   
  context = math_ops.matmul(expanded_alignments, attention_mechanism.values)
  context = array_ops.squeeze(context, [1])

  if attention_layer is not None:
    attention = attention_layer(array_ops.concat([cell_output, context], 1))
  else:
    attention = context

  return attention, alignments, next_attention_state


class AttentionWrapper(rnn_cell_impl.RNNCell):
  """Wraps another `RNNCell` with attention.
  """

  def __init__(self,
               cell,
               attention_mechanism,
               attention_layer_size=None,
               alignment_history=False,
               cell_input_fn=None,
               output_attention=True,
               initial_cell_state=None,
               name=None,
               attention_layer=None):
    
    super(AttentionWrapper, self).__init__(name=name)
    rnn_cell_impl.assert_like_rnncell("cell", cell)
    if isinstance(attention_mechanism, (list, tuple)):
      self._is_multi = True
      attention_mechanisms = attention_mechanism
      for attention_mechanism in attention_mechanisms:
        if not isinstance(attention_mechanism, AttentionMechanism):
          raise TypeError(
              "attention_mechanism must contain only instances of "
              "AttentionMechanism, saw type: %s"
              % type(attention_mechanism).__name__)
    else:
      self._is_multi = False
      if not isinstance(attention_mechanism, AttentionMechanism):
        raise TypeError(
            "attention_mechanism must be an AttentionMechanism or list of "
            "multiple AttentionMechanism instances, saw type: %s"
            % type(attention_mechanism).__name__)
      attention_mechanisms = (attention_mechanism,)

    if cell_input_fn is None:
      cell_input_fn = (
          lambda inputs, attention: array_ops.concat([inputs, attention], -1))
    else:
      if not callable(cell_input_fn):
        raise TypeError(
            "cell_input_fn must be callable, saw type: %s"
            % type(cell_input_fn).__name__)

    if attention_layer_size is not None and attention_layer is not None:
      raise ValueError("Only one of attention_layer_size and attention_layer "
                       "should be set")

    if attention_layer_size is not None:
      attention_layer_sizes = tuple(
          attention_layer_size
          if isinstance(attention_layer_size, (list, tuple))
          else (attention_layer_size,))
      if len(attention_layer_sizes) != len(attention_mechanisms):
        raise ValueError(
            "If provided, attention_layer_size must contain exactly one "
            "integer per attention_mechanism, saw: %d vs %d"
            % (len(attention_layer_sizes), len(attention_mechanisms)))
      self._attention_layers = tuple(
          layers_core.Dense(
              attention_layer_size,
              name="attention_layer",
              use_bias=False,
              dtype=attention_mechanisms[i].dtype)
          for i, attention_layer_size in enumerate(attention_layer_sizes))
      self._attention_layer_size = sum(attention_layer_sizes)
    elif attention_layer is not None:
      self._attention_layers = tuple(
          attention_layer
          if isinstance(attention_layer, (list, tuple))
          else (attention_layer,))
      if len(self._attention_layers) != len(attention_mechanisms):
        raise ValueError(
            "If provided, attention_layer must contain exactly one "
            "layer per attention_mechanism, saw: %d vs %d"
            % (len(self._attention_layers), len(attention_mechanisms)))
      self._attention_layer_size = sum(
          tensor_shape.dimension_value(layer.compute_output_shape(
              [None,
               cell.output_size + tensor_shape.dimension_value(
                   mechanism.values.shape[-1])])[-1])
          for layer, mechanism in zip(
              self._attention_layers, attention_mechanisms))
    else:
      self._attention_layers = None
      self._attention_layer_size = sum(
          tensor_shape.dimension_value(attention_mechanism.values.shape[-1])
          for attention_mechanism in attention_mechanisms)

    self._cell = cell
    self._attention_mechanisms = attention_mechanisms
    self._cell_input_fn = cell_input_fn
    self._output_attention = output_attention
    self._alignment_history = alignment_history
    with ops.name_scope(name, "AttentionWrapperInit"):
      if initial_cell_state is None:
        self._initial_cell_state = None
      else:
        final_state_tensor = nest.flatten(initial_cell_state)[-1]
        state_batch_size = (
            tensor_shape.dimension_value(final_state_tensor.shape[0])
            or array_ops.shape(final_state_tensor)[0])
        error_message = (
            "When constructing AttentionWrapper %s: " % self._base_name +
            "Non-matching batch sizes between the memory "
            "(encoder output) and initial_cell_state.  Are you using "
            "the BeamSearchDecoder?  You may need to tile your initial state "
            "via the tf.contrib.seq2seq.tile_batch function with argument "
            "multiple=beam_width.")
        with ops.control_dependencies(
            self._batch_size_checks(state_batch_size, error_message)):
          self._initial_cell_state = nest.map_structure(
              lambda s: array_ops.identity(s, name="check_initial_cell_state"),
              initial_cell_state)

  def _batch_size_checks(self, batch_size, error_message):
    return [check_ops.assert_equal(batch_size,
                                   attention_mechanism.batch_size,
                                   message=error_message)
            for attention_mechanism in self._attention_mechanisms]

  def _item_or_tuple(self, seq):
    """Returns `seq` as tuple or the singular element.
    Which is returned is determined by how the AttentionMechanism(s) were passed
    to the constructor.
    Args:
      seq: A non-empty sequence of items or generator.
    Returns:
       Either the values in the sequence as a tuple if AttentionMechanism(s)
       were passed to the constructor as a sequence or the singular element.
    """
    t = tuple(seq)
    if self._is_multi:
      return t
    else:
      return t[0]

  @property
  def output_size(self):
    if self._output_attention:
      return self._attention_layer_size
    else:
      return self._cell.output_size

  @property
  def state_size(self):
    """The `state_size` property of `AttentionWrapper`.
    Returns:
      An `AttentionWrapperState` tuple containing shapes used by this object.
    """
    return AttentionWrapperState(
        cell_state=self._cell.state_size,
        time=tensor_shape.TensorShape([]),
        attention=self._attention_layer_size,
        alignments=self._item_or_tuple(
            a.alignments_size for a in self._attention_mechanisms),
        attention_state=self._item_or_tuple(
            a.state_size for a in self._attention_mechanisms),
        alignment_history=self._item_or_tuple(
            a.alignments_size if self._alignment_history else ()
            for a in self._attention_mechanisms))  # sometimes a TensorArray

  def zero_state(self, batch_size, dtype):
    
    with ops.name_scope(type(self).__name__ + "ZeroState", values=[batch_size]):
      if self._initial_cell_state is not None:
        cell_state = self._initial_cell_state
      else:
        cell_state = self._cell.zero_state(batch_size, dtype)
      error_message = (
          "When calling zero_state of AttentionWrapper %s: " % self._base_name +
          "Non-matching batch sizes between the memory "
          "(encoder output) and the requested batch size.  Are you using "
          "the BeamSearchDecoder?  If so, make sure your encoder output has "
          "been tiled to beam_width via tf.contrib.seq2seq.tile_batch, and "
          "the batch_size= argument passed to zero_state is "
          "batch_size * beam_width.")
      with ops.control_dependencies(
          self._batch_size_checks(batch_size, error_message)):
        cell_state = nest.map_structure(
            lambda s: array_ops.identity(s, name="checked_cell_state"),
            cell_state)
      initial_alignments = [
          attention_mechanism.initial_alignments(batch_size, dtype)
          for attention_mechanism in self._attention_mechanisms]
      return AttentionWrapperState(
          cell_state=cell_state,
          time=array_ops.zeros([], dtype=dtypes.int32),
          attention=_zero_state_tensors(self._attention_layer_size, batch_size,
                                        dtype),
          alignments=self._item_or_tuple(initial_alignments),
          attention_state=self._item_or_tuple(
              attention_mechanism.initial_state(batch_size, dtype)
              for attention_mechanism in self._attention_mechanisms),
          alignment_history=self._item_or_tuple(
              tensor_array_ops.TensorArray(
                  dtype,
                  size=0,
                  dynamic_size=True,
                  element_shape=alignment.shape)
              if self._alignment_history else ()
              for alignment in initial_alignments))

  def call(self, inputs, state):
    
    if not isinstance(state, AttentionWrapperState):
      raise TypeError("Expected state to be instance of AttentionWrapperState. "
                      "Received type %s instead."  % type(state))

    # Step 1: Calculate the true inputs to the cell based on the
    # previous attention value.
    cell_inputs = self._cell_input_fn(inputs, state.attention)
    cell_state = state.cell_state
    cell_output, next_cell_state = self._cell(cell_inputs, cell_state)

    cell_batch_size = (
        tensor_shape.dimension_value(cell_output.shape[0]) or
        array_ops.shape(cell_output)[0])
    error_message = (
        "When applying AttentionWrapper %s: " % self.name +
        "Non-matching batch sizes between the memory "
        "(encoder output) and the query (decoder output).  Are you using "
        "the BeamSearchDecoder?  You may need to tile your memory input via "
        "the tf.contrib.seq2seq.tile_batch function with argument "
        "multiple=beam_width.")
    with ops.control_dependencies(
        self._batch_size_checks(cell_batch_size, error_message)):
      cell_output = array_ops.identity(
          cell_output, name="checked_cell_output")

    if self._is_multi:
      previous_attention_state = state.attention_state
      previous_alignment_history = state.alignment_history
    else:
      previous_attention_state = [state.attention_state]
      previous_alignment_history = [state.alignment_history]

    all_alignments = []
    all_attentions = []
    all_attention_states = []
    maybe_all_histories = []
    for i, attention_mechanism in enumerate(self._attention_mechanisms):
      attention, alignments, next_attention_state = _compute_attention(
          attention_mechanism, cell_output, previous_attention_state[i],
          self._attention_layers[i] if self._attention_layers else None)
      alignment_history = previous_alignment_history[i].write(
          state.time, alignments) if self._alignment_history else ()

      all_attention_states.append(next_attention_state)
      all_alignments.append(alignments)
      all_attentions.append(attention)
      maybe_all_histories.append(alignment_history)

    attention = array_ops.concat(all_attentions, 1)
    next_state = AttentionWrapperState(
        time=state.time + 1,
        cell_state=next_cell_state,
        attention=attention,
        attention_state=self._item_or_tuple(all_attention_states),
        alignments=self._item_or_tuple(all_alignments),
        alignment_history=self._item_or_tuple(maybe_all_histories))

    if self._output_attention:
      return attention, next_state
    else:
      return cell_output, next_state