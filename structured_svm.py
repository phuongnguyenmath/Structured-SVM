import tensorflow as tf
print(tf.__version__)
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
#from tensorflow.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import rnn_cell
#from tensorflow.python.ops import variable_scope as vs


def crf_unary_score(tag_indices, sequence_lengths, tensor_potentials):
    """Computes the unary scores of tag sequences.
    Args:
        tag_indices: A [batch_size, max_seq_len] matrix of tag indices.
        sequence_lengths: A [batch_size] vector of true sequence lengths.
        potentials: A [batch_size, max_seq_len, num_tags] tensor of unary potentials.
    Returns:
        unary_scores: A [batch_size] vector of unary scores.
    """
    batch_size = array_ops.shape(tensor_potentials)[0]
    max_seq_len = array_ops.shape(tensor_potentials)[1]
    num_tags = array_ops.shape(tensor_potentials)[2]
    
    ### P note: tranform to 3-dim tensor to 1-dim tensor
    flattened_inputs = array_ops.reshape(tensor_potentials, [-1])

    ### P note: ?????
    offsets = array_ops.expand_dims(math_ops.range(batch_size) * max_seq_len * num_tags, 1)
    offsets += array_ops.expand_dims(math_ops.range(max_seq_len) * num_tags, 0)
    # Use int32 or int64 based on tag_indices' dtype.
    if tag_indices.dtype == dtypes.int64:
        offsets = math_ops.cast(offsets, dtypes.int64)
        
    ### P note: prepare index list for choose suitable items in flattened_inputs
    flattened_tag_indices = array_ops.reshape(offsets + tag_indices, [-1])
    
    unary_scores = array_ops.reshape(
                                    array_ops.gather(flattened_inputs, flattened_tag_indices),
                                    [batch_size, max_seq_len])
    
    ### P note: compare the length with max_seq_len  
    masks = array_ops.sequence_mask(sequence_lengths,
                                  maxlen=array_ops.shape(tag_indices)[1],
                                  dtype=dtypes.float32)
    
    unary_scores = math_ops.reduce_sum(unary_scores * masks, 1)
    return unary_scores


###############################################################################
def crf_binary_score(tag_indices, sequence_lengths, transition_params):
    """Computes the binary scores of tag sequences.
    Args:
        tag_indices: A [batch_size, max_seq_len] matrix of tag indices.
        sequence_lengths: A [batch_size] vector of true sequence lengths.
        transition_params: A [num_tags, num_tags] matrix of binary potentials.
    Returns:
        binary_scores: A [batch_size] vector of binary scores.
    """
    
    # Get shape information.
    num_tags = transition_params.get_shape()[0]
    num_transitions = array_ops.shape(tag_indices)[1] - 1

    # Truncate by one on each side of the sequence to get the start and end
    # indices of each transition.
    
    ### P note: Delete the last column of tag_indices
    start_tag_indices = array_ops.slice(tag_indices, [0, 0],
                                      [-1, num_transitions])
    
    ### P note: Delete the first column of tag_indices
    end_tag_indices = array_ops.slice(tag_indices, [0, 1], [-1, num_transitions])
    
    # Encode the indices in a flattened representation.
    ### P note: convert A(i,j) to A(i)*num_tags + A(j) 
    flattened_transition_indices = start_tag_indices * num_tags + end_tag_indices
    flattened_transition_params = array_ops.reshape(transition_params, [-1])

    # Get the binary scores based on the flattened representation.
    binary_scores = array_ops.gather(flattened_transition_params,
                                   flattened_transition_indices)
    
    ### P note: compare the length with max_seq_len 
    masks = array_ops.sequence_mask(sequence_lengths,
                                  maxlen=array_ops.shape(tag_indices)[1],
                                  dtype=dtypes.float32)
    
    ### P note: the number of transition  = lenth - 1
    ### Delete the first column of masks matrix
    truncated_masks = array_ops.slice(masks, [0, 1], [-1, -1])
    binary_scores = math_ops.reduce_sum(binary_scores * truncated_masks, 1)
    return binary_scores


###############################################################################

def crf_sequence_score(tensor_potentials, tag_indices, sequence_lengths,
                       transition_params, balance_param = 1):
    """Computes the unnormalized score for a tag sequence.
    Args:
        inputs: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
        to use as input to the CRF layer.
        tag_indices: A [batch_size, max_seq_len] matrix of tag indices for which we
        compute the unnormalized score.
        sequence_lengths: A [batch_size] vector of true sequence lengths.
        transition_params: A [num_tags, num_tags] transition matrix.
    Returns:
        sequence_scores: A [batch_size] vector of unnormalized sequence scores.
    """
    
    
    unary_scores = crf_unary_score(tag_indices, sequence_lengths, tensor_potentials)
    binary_scores = crf_binary_score(tag_indices, sequence_lengths,
                                     transition_params)
    sequence_scores = unary_scores + balance_param*binary_scores
    return sequence_scores


###############################################################################
 ### UNDERSTAND

def viterbi_decode(score, transition_params, balance_param = 1):
    """Decode the highest scoring sequence of tags outside of TensorFlow.
    This should only be used at test time.
    Args:
        score: A [seq_len, num_tags] matrix of unary potentials.  # potential of only ONE SAMPLE
        ### matrix potential of W = (W1,W2,...,Wk) X = (X1,X2,..., Xm)
        [[X1W1, X1W2,...,X1Wk],
        [X2W1,X2W2,...,X2Wk],
        ...,
        [XmW1,XmW2,...,XmWk]]
        transition_params: A [num_tags, num_tags] matrix of binary potentials.
        Returns:
            viterbi: A [seq_len] list of integers containing the highest scoring tag indices.
            viterbi_score: A float containing the score for the Viterbi sequence.
    """
    
    ### P note: trellis  = Viterbi matrix
    ### P note: backpointers = matrix of argmax value
    trellis = np.zeros_like(score)
    backpointers = np.zeros_like(score, dtype=np.int32)
    trellis[0] = score[0]
    
    ### P note: 
    for t in range(1, score.shape[0]):
        #np.expand_dims(trellis[t - 1], 1) is unitary socre at xt ----> COLUMNS
        v = np.expand_dims(trellis[t - 1], 1) + balance_param*transition_params 
        ### pi[t,s] = max_{sk in S} {pi[t-1,sk] + tranistion_params(sk,s)} + unary_score(t-1,s)
        ### s ROW
        ### sk COLUMN
        trellis[t] = score[t] + np.amax(v, 0)  # Maxima along the first axis COLUMNS
        backpointers[t] = np.argmax(v, 0)
        
    viterbi = [np.argmax(trellis[-1])]
    
    for bp in reversed(backpointers[1:]):
        viterbi.append(bp[viterbi[-1]])
        
    viterbi.reverse()
    
    viterbi_score = np.max(trellis[-1])
    return viterbi, viterbi_score

###############################################################################
    

def viterbi_LAD(score, transition_params, sequence_label, balance_param = 1):   ### WRITE   ### MUST CHECK
    """Decode the highest scoring sequence of tags outside of TensorFlow.
    This should only be used at test time.
    Args:
        score: A [seq_len, num_tags] matrix of unary potentials.
        transition_params: A [num_tags, num_tags] matrix of binary potentials.
    Returns:
        viterbi: A [seq_len] list of integers containing the highest scoring tag indices.
        viterbi_score: A float containing the score for the Viterbi sequence.
    """
    trellis = np.zeros_like(score)
    backpointers = np.zeros_like(score, dtype=np.int32)
    num_tags = transition_params.shape[0]
    state = np.arange(num_tags)
    trellis[0] = (state != sequence_label[0])*1 + score[0] 
    
    for t in range(1, score.shape[0]):
        v = (state != sequence_label[t])*1 + np.expand_dims(trellis[t - 1], 1) + balance_param*transition_params #### REWRITE
        trellis[t] = score[t] + np.max(v, 0)
        backpointers[t] = np.argmax(v, 0)

    viterbi = [np.argmax(trellis[-1])]
    for bp in reversed(backpointers[1:]):
        viterbi.append(bp[viterbi[-1]])
    
    viterbi.reverse()
    
    viterbi_score = np.max(trellis[-1])
    return viterbi, viterbi_score

###############################################################################
class CrfDecodeForwardRnnCell(rnn_cell.RNNCell): 
    """Computes the forward decoding in a linear-chain CRF.
    """
    def __init__(self, transition_params, balance_param = 1):
        """Initialize the CrfDecodeForwardRnnCell.
        Args:
            transition_params: A [num_tags, num_tags] matrix of binary
            potentials. This matrix is expanded into a
            [1, num_tags, num_tags] in preparation for the broadcast
            summation occurring within the cell.
        """
        self._transition_params = array_ops.expand_dims(transition_params, 0)  ### P note: [1, num_tags, num_tags]
        self._num_tags = tensor_shape.dimension_value(transition_params.shape[0])
        self._balance_param = balance_param
    @property
    def state_size(self):
        return self._num_tags
    
    @property
    def output_size(self):
        return self._num_tags
    
    
    def __call__(self, inputs, state, scope=None):
        """Build the CrfDecodeForwardRnnCell.
        Args:
            inputs: A [batch_size, num_tags] matrix of unary potentials.
            state: A [batch_size, num_tags] matrix containing the previous step's
            score values. ### P note: is trellis in Viterbi_decode
            scope: Unused variable scope of this cell.
            
        Returns:
            backpointers: A [batch_size, num_tags] matrix of backpointers.
            new_state: A [batch_size, num_tags] matrix of new score values.
        """
        
        # For simplicity, in shape comments, denote:
        # 'batch_size' by 'B', 'max_seq_len' by 'T' , 'num_tags' by 'O' (output).
        state = array_ops.expand_dims(state, 2)                         # [B, O, 1]
        
        # This addition op broadcasts self._transitions_params along the zeroth
        # dimension and state along the second dimension.
        # [B, O, 1] + [1, O, O] -> [B, O, O]
        transition_scores = state + self._balance_param*self._transition_params             # [B, O, O]
        new_state = inputs + math_ops.reduce_max(transition_scores, [1])  # [B, O]
        backpointers = math_ops.argmax(transition_scores, 1)
        backpointers = math_ops.cast(backpointers, dtype=dtypes.int32)    # [B, O]
        return backpointers, new_state

###############################################################################
class CrfDecodeBackwardRnnCell(rnn_cell.RNNCell):
    """Computes backward decoding in a linear-chain CRF.
    """
    
    def __init__(self, num_tags):
        """Initialize the CrfDecodeBackwardRnnCell.
        Args:
            num_tags: An integer. The number of tags.
        """
        self._num_tags = num_tags
        
    @property
    def state_size(self):
        return 1
    
    @property
    def output_size(self):
        return 1
    
    def __call__(self, inputs, state, scope=None):
        """Build the CrfDecodeBackwardRnnCell.
        Args:
            inputs: A [batch_size, num_tags] matrix of
            backpointer of next step (in time order).
            state: A [batch_size, 1] matrix of tag index of next step.
            scope: Unused variable scope of this cell.
        Returns: 
            new_tags, new_tags: A pair of [batch_size, num_tags]
            tensors containing the new tag indices.
        """
        
        state = array_ops.squeeze(state, axis=[1])                # [B]
        batch_size = array_ops.shape(inputs)[0]
        b_indices = math_ops.range(batch_size)                    # [B]
        indices = array_ops.stack([b_indices, state], axis=1)     # [B, 2]
        new_tags = array_ops.expand_dims( 
                                    gen_array_ops.gather_nd(inputs, indices),  # [B]
                                         axis=-1)          # [B, 1]
        return new_tags, new_tags


###############################################################################
def crf_decode(potentials, transition_params, sequence_length, balance_param = 1):
    """Decode the highest scoring sequence of tags in TensorFlow.
    This is a function for tensor.
    Args:
        potentials: A [batch_size, max_seq_len, num_tags] tensor of unary potentials.
        transition_params: A [num_tags, num_tags] matrix of binary potentials.
        sequence_length: A [batch_size] vector of true sequence lengths.
        
    Returns:
        decode_tags: A [batch_size, max_seq_len] matrix, with dtype `tf.int32`.
        Contains the highest scoring tag indices.
        best_score: A [batch_size] vector, containing the score of `decode_tags`.
    """
    """Decoding of highest scoring sequence."""
    # For simplicity, in shape comments, denote:
    # 'batch_size' by 'B', 'max_seq_len' by 'T' , 'num_tags' by 'O' (output).
    num_tags = tensor_shape.dimension_value(potentials.shape[2])
    
    # Computes forward decoding. Get last score and backpointers.
    crf_fwd_cell = CrfDecodeForwardRnnCell(transition_params, balance_param)
    initial_state = array_ops.slice(potentials, [0, 0, 0], [-1, 1, -1])    # [B,1, O]
    initial_state = array_ops.squeeze(initial_state, axis=[1])  # [B, O]
    inputs = array_ops.slice(potentials, [0, 1, 0], [-1, -1, -1])  # [B, T-1, O]
    
    # Sequence length is not allowed to be less than zero.
    sequence_length_less_one = math_ops.maximum(
                            constant_op.constant(0, dtype=sequence_length.dtype),
                            sequence_length - 1)
    backpointers, last_score = rnn.dynamic_rnn(  # [B, T - 1, O], [B, O]
                                    crf_fwd_cell,
                                    inputs=inputs,
                                    sequence_length=sequence_length_less_one,
                                    initial_state=initial_state,
                                    time_major=False,
                                    dtype=dtypes.int32)
    backpointers = gen_array_ops.reverse_sequence(                   # [B, T - 1, O]
                            backpointers, sequence_length_less_one, seq_dim=1)
    
    # Computes backward decoding. Extract tag indices from backpointers.
    crf_bwd_cell = CrfDecodeBackwardRnnCell(num_tags)
    initial_state = math_ops.cast(math_ops.argmax(last_score, axis=1),  # [B]
                                  dtype=dtypes.int32)
    initial_state = array_ops.expand_dims(initial_state, axis=-1)  # [B, 1]
    decode_tags, _ = rnn.dynamic_rnn(  # [B, T - 1, 1]
                                crf_bwd_cell,
                                inputs=backpointers,
                                sequence_length=sequence_length_less_one,
                                initial_state=initial_state,
                                time_major=False,
                                dtype=dtypes.int32)
    decode_tags = array_ops.squeeze(decode_tags, axis=[2])  # [B, T - 1]
    decode_tags = array_ops.concat([initial_state, decode_tags],   # [B, T]
                                   axis=1)
    decode_tags = gen_array_ops.reverse_sequence(  # [B, T]
                                    decode_tags, sequence_length, seq_dim=1)
    best_score = math_ops.reduce_max(last_score, axis=1)  # [B]
    
    return decode_tags, best_score

###############################################################################
class CrfLADForwardRnnCell(rnn_cell.RNNCell):
  """Computes the forward decoding in a linear-chain CRF.
  """

  def __init__(self, transition_params, balance_param = 1):
    """Initialize the CrfDecodeForwardRnnCell.
    Args:
      transition_params: A [num_tags, num_tags] matrix of binary
        potentials. This matrix is expanded into a
        [1, num_tags, num_tags] in preparation for the broadcast
        summation occurring within the cell.
    """
    self._transition_params = array_ops.expand_dims(transition_params, 0)
    self._num_tags = tensor_shape.dimension_value(transition_params.shape[0])
    self._balance_param = balance_param

  @property
  def state_size(self):
    return self._num_tags

  @property
  def output_size(self):
    return self._num_tags

  def __call__(self, inputs, state, scope=None):      ### ??? NEED CHECK ???
    
    """Build the CrfDecodeForwardRnnCell.
    Args:
      inputs: A [batch_size, num_tags + 1 ] matrix    inputs = concatenate ([potential, tag], axis = 1)
             potential: A [batch_size, num_tags] matrix of unary potentials.
             tag : [batch_size] vector of label y
      state: A [batch_size, num_tags] matrix containing the previous step's  ### of RNN
            score values.  ### P note: is trellis in Viterbi_decode
      
      scope: Unused variable scope of this cell.
    Returns:
      backpointers: A [batch_size, num_tags] matrix of backpointers.
      new_state: A [batch_size, num_tags] matrix of new score values.
    """
    # For simplicity, in shape comments, denote:
    # 'batch_size' by 'B', 'max_seq_len' by 'T' , 'num_tags' by 'O' (output).
    
    batch_size = tensor_shape.dimension_value(state.shape[0])
    num_tags = tensor_shape.dimension_value(state.shape[1])
    
    state = array_ops.expand_dims(state, 2)                       # [B, O, 1]
    
    potential = inputs[:,0:num_tags]
    
    tag = inputs[:,-1]
    tag = tf.expand_dims(tag, 1)
    
    state_tag = tf.range(num_tags, dtype=dtypes.float32) + tf.zeros([batch_size, num_tags], dtype=dtypes.float32)
    
    tag_diff = state_tag - tag
    
    tag_score = tf.div_no_nan(tag_diff,tag_diff)

    #tag_score = tf.reshape(tag_score, shape = [batch_size, 1]) + tf.zeros([batch_size, num_tags], dtype=dtypes.int32)
                                 ##########
    # This addition op broadcasts self._transitions_params along the zeroth
    # dimension and state along the second dimension.
    # [B, O, 1] + [1, O, O] -> [B, O, O]
    tag_score = array_ops.expand_dims(tag_score, 2)
    
    transition_scores = tag_score + state + self._balance_param*self._transition_params             # [B, O, O]
    
    
    new_state = potential + math_ops.reduce_max(transition_scores, [1])  # [B, O]
    backpointers = math_ops.argmax(transition_scores, 1)
    
    backpointers = math_ops.cast(backpointers, dtype=dtypes.int32)    # [B, O]
    return backpointers, new_state


###############################################################################
class CrfLADBackwardRnnCell(rnn_cell.RNNCell):
    """Computes backward decoding in a linear-chain CRF.
    """
    def __init__(self, num_tags):
        """Initialize the CrfDecodeBackwardRnnCellAD.
        Args:
            num_tags: An integer. The number of tags.
            
        """
        self._num_tags = num_tags
        
    @property
    def state_size(self):
        return 1
    
    @property
    def output_size(self):
        return 1
    
    def __call__(self, inputs, state, scope=None):
        """Build the CrfDecodeBackwardRnnCell.
        Args:
            inputs: A [batch_size, num_tags] matrix of 
            backpointer of next step (in time order).
            state: A [batch_size, 1] matrix of tag index of next step.
            scope: Unused variable scope of this cell.
        Returns:
            new_tags, new_tags: A pair of [batch_size, num_tags]
            tensors containing the new tag indices.
        """
        state = array_ops.squeeze(state, axis=[1])                # [B]
        batch_size = array_ops.shape(inputs)[0]
        b_indices = math_ops.range(batch_size)                    # [B]
        indices = array_ops.stack([b_indices, state], axis=1)     # [B, 2]
        new_tags = array_ops.expand_dims(
                gen_array_ops.gather_nd(inputs, indices),             # [B]
                axis=-1)                                              # [B, 1]
        
        return new_tags, new_tags
##############################################################################s
def ssvm_LAD(potentials, tag_indices, sequence_length, transition_params, balance_param = 1): 
    """Decode the highest scoring sequence of tags in TensorFlow.
    This is a function for tensor.
    Args:
        inputs/potentials: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
        transition_params: A [num_tags, num_tags] matrix of binary potentials.
        sequence_length: A [batch_size] vector of true sequence lengths.
        tag_indices: A [batch_size, max_seq_len] maxtrix
    Returns:
        decode_tags: A [batch_size, max_seq_len] matrix, with dtype `tf.int32`.
        Contains the highest scoring tag indices.
        best_score: A [batch_size] vector, containing the score of `decode_tags`
    """
    
    """Decoding of highest scoring sequence."""
    # For simplicity, in shape comments, denote:
    # 'batch_size' by 'B', 'max_seq_len' by 'T' , 'num_tags' by 'O' (output).
    #batch_size = tensor_shape.dimension_value(potentials.shape[0])
    #max_seq_len = tensor_shape.dimension_value(potentials.shape[1])  
    
    
    num_tags = tensor_shape.dimension_value(potentials.shape[2])
    
    tag_indices = tf.expand_dims(tag_indices, axis = 2)
    tag_indices = tf.cast(tag_indices, tf.float32)
    potentialstags = tf.concat([potentials, tag_indices], axis = 2)
    
    ######
    # Computes forward decoding. Get last score and backpointers.
    crf_fwd_cell = CrfLADForwardRnnCell(transition_params, balance_param)
    initial_state = array_ops.slice(potentials, [0, 0, 0], [-1, 1, -1])    # [B,1, O]
    initial_state = array_ops.squeeze(initial_state, axis=[1])  # [B, O]
    inputs = array_ops.slice(potentialstags, [0, 1, 0], [-1, -1, -1])  # [B, T-1, O]
    
    # Sequence length is not allowed to be less than zero.
    sequence_length_less_one = math_ops.maximum(
                         constant_op.constant(0, dtype=sequence_length.dtype),
                         sequence_length - 1)
    backpointers, last_score = rnn.dynamic_rnn(  # [B, T - 1, O], [B, O]
                                        crf_fwd_cell,
                                        inputs=inputs,
                                        sequence_length=sequence_length_less_one,
                                        initial_state=initial_state,
                                        time_major=False,
                                        dtype=dtypes.int32)
    backpointers = gen_array_ops.reverse_sequence(                   # [B, T - 1, O]
                                        backpointers, sequence_length_less_one, seq_dim=1)
    
    # Computes backward decoding. Extract tag indices from backpointers.
    crf_bwd_cell = CrfLADBackwardRnnCell(num_tags)
    initial_state = math_ops.cast(math_ops.argmax(last_score, axis=1),  # [B]
                                  dtype=dtypes.int32)
    initial_state = array_ops.expand_dims(initial_state, axis=-1)  # [B, 1]
    
    decode_tags, _ = rnn.dynamic_rnn(  # [B, T - 1, 1]
                                    crf_bwd_cell,
                                    inputs=backpointers,
                                    sequence_length=sequence_length_less_one,
                                    initial_state=initial_state,
                                    time_major=False,
                                    dtype=dtypes.int32)
    decode_tags = array_ops.squeeze(decode_tags, axis=[2])  # [B, T - 1]
    decode_tags = array_ops.concat([initial_state, decode_tags],   # [B, T]
                                   axis=1)
    decode_tags = gen_array_ops.reverse_sequence(  # [B, T]
                                                decode_tags, sequence_length, seq_dim=1)
    best_score = math_ops.reduce_max(last_score, axis=1)  # [B]
    
    return decode_tags, best_score

###############################################################################
def ssvm_loss_function(tensor_potentials, tag_indices,
                       sequence_lengths, transition_params, balance_param = 1):
    """Computes the log-likelihood of tag sequences in a CRF.
    Args:
        potentials: A [batch_size, max_seq_len, num_tags] tensor of unary potentials
        to use as input to the CRF layer.
        tag_indices: A [batch_size, max_seq_len] matrix of tag indices for which we
        compute the log-likelihood.
        sequence_lengths: A [batch_size] vector of true sequence lengths.
        transition_params: A [num_tags, num_tags] transition matrix, if available.
    Returns:
        log_likelihood: A [batch_size] `Tensor` containing the log-likelihood of
        each example, given the sequence of tag indices.
        transition_params: A [num_tags, num_tags] transition matrix. This is either
        provided by the caller or created in this function.
    """
    
    # Get shape information.
    #num_tags = tensor_shape.dimension_value(tensor_potentials.shape[2])
    # Get the transition matrix if not provided.
    #if transition_params is None:
    #  transition_params = vs.get_variable("transitions", [num_tags, num_tags])
    sequence_scores = crf_sequence_score(tensor_potentials, tag_indices, sequence_lengths,
                                       transition_params, balance_param)
    LAD_sequences, LAD_scores = ssvm_LAD(tensor_potentials, tag_indices, sequence_lengths,
                                       transition_params, balance_param)  ### REWRITE
    
    # Normalize the scores to get the log-likelihood per example.
    ssvm_loss = LAD_scores - sequence_scores
    return ssvm_loss
###############################################################################

