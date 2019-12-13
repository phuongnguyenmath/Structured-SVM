import numpy as np
import tensorflow as tf
from structured_svm import *

# Data settings.
num_examples = 10
num_words = 50
num_features = 100
num_tags = 5

# Random features.
x = np.random.rand(num_examples, num_words, num_features).astype(np.float32)

# Random tag indices representing the gold sequence.
y = np.random.randint(num_tags, size=[num_examples, num_words]).astype(np.int32)

# All sequences in this example have the same length, but they can be variable in a real model.
sequence_lengths = np.full(num_examples, num_words - 1, dtype=np.int32)


##### Hyperparameter
balance_param = 1.0

#writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

# Train and evaluate the model.
with tf.Graph().as_default():
    with tf.Session() as session:
        # Add the data to the TensorFlow graph.
        x_t = tf.constant(x)
        y_t = tf.constant(y)
        sequence_lengths_t = tf.constant(sequence_lengths)
        
        # Compute unary scores from a linear layer.
        weights = tf.get_variable("weights", [num_features, num_tags])
        transition_params = tf.get_variable("transition_params", [num_tags, num_tags])
        matricized_x_t = tf.reshape(x_t, [-1, num_features])
        matricized_unary_scores = tf.matmul(matricized_x_t, weights)
        unary_scores = tf.reshape(matricized_unary_scores,
                                  [num_examples, num_words, num_tags])
    
    
        ssvm_loss = ssvm_loss_function(unary_scores, y_t, 
                                       sequence_lengths_t, transition_params, balance_param)    
    
    
        # Compute the log-likelihood of the gold sequences and keep the transition
        # params for inference at test time.
    
        # Compute the viterbi sequence and score.
        viterbi_sequence, viterbi_score = crf_decode(unary_scores, 
                        transition_params, sequence_lengths_t,balance_param)
    
        # Add a training op to tune the parameters. 
        #loss = tf.reduce_mean(ssvm_loss) 
        
        # Regularization
        loss = tf.reduce_mean(ssvm_loss) + tf.nn.l2_loss(weights) + tf.nn.l2_loss(transition_params)  
        
        train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    
        session.run(tf.global_variables_initializer())
    
        mask = (np.expand_dims(np.arange(num_words), axis=0) <
                np.expand_dims(sequence_lengths, axis=1))
        total_labels = np.sum(sequence_lengths)
    
        # Train for a fixed number of iterations.
        for i in range(1000):
            tf_viterbi_sequence, _ = session.run([viterbi_sequence, train_op])
            if i % 100 == 0:
                correct_labels = np.sum((y == tf_viterbi_sequence) * mask)
                accuracy = 100.0 * correct_labels / float(total_labels)
                print("Accuracy: %.2f%%" % accuracy)