"""
#################################################################################
The following code contains the implementation of question 1 using Vanilla RNN

################################################################################
"""


import numpy as np
from my_rnn import RNN
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve,  roc_auc_score,auc
import warnings
warnings.simplefilter("ignore", UserWarning)

 
def del_all_flags(FLAGS):
    flags_dict = FLAGS._flags()    
    keys_list = [keys for keys in flags_dict]    
    for keys in keys_list:
        FLAGS.__delattr__(keys)


def onehot_encoding(df_categorical):
    result=df_categorical.copy()
    for feature_name in result.columns:
        one_hot = pd.get_dummies(result[feature_name])
        result = result.drop(feature_name,axis = 1)
        result = result.join(one_hot,lsuffix='_caller', rsuffix='_other')
    return result

"""
prepare batch of training
"""
def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

"""
Fuction definition for LSTM Model
"""
def train_vanilla_rnn():
    del_all_flags(tf.flags.FLAGS)

    flags = tf.app.flags
    FLAGS = flags.FLAGS
    
    # Data loading params
    
    tf.flags.DEFINE_float("dev_sample_percentage", .3, "Percentage of the training data to use for validation")
    tf.flags.DEFINE_integer("max_sentence_length", 120, "Max sentence length in train/test data (Default: 100)")
    
    # Model Hyperparameters
    tf.flags.DEFINE_string("word2vec", None, "Word2vec file with pre-trained embeddings")
    tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (Default: 300)")
    tf.flags.DEFINE_integer("hidden_size", 128, "Dimensionality of character embedding (Default: 128)")
    tf.flags.DEFINE_float("dropout_keep_prob", 0.7, "Dropout keep probability (Default: 0.5)")
    tf.flags.DEFINE_float("l2_reg_lambda", 3.0, "L2 regularization lambda (Default: 3.0)")
    
    # Training parameters
    tf.flags.DEFINE_integer("batch_size", 256, "Batch Size (Default: 64)")
    tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (Default: 100)")
    tf.flags.DEFINE_integer("display_every", 10, "Number of iterations to display training info.")
    tf.flags.DEFINE_integer("evaluate_every", 137, "Evaluate model on dev set after this many steps")
    tf.flags.DEFINE_integer("num_checkpoints", 1, "Number of checkpoints to store")
    tf.flags.DEFINE_float("learning_rate", 1e-3, "Which learning rate to start with. (Default: 1e-3)")
    
    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
    
    tf.flags.DEFINE_string("cell_type", "vanilla", "Type of rnn cell. Choose 'vanilla' or 'lstm' or 'gru' (Default: vanilla)")

    
    with tf.device('/cpu:0'):
        train_x = pd.read_csv('train_data.csv',header=0)
        y=pd.read_csv('train_labels.csv',header=None)
        
        test_x=pd.read_csv('test_data.csv',header=0)
        test_yy=pd.read_csv('test_labels.csv',header=None)
    
    """
    #################################################################
    one hot encoding of the label
    #################################################################
    """
    x=train_x.values
    y=onehot_encoding(y)
    y=y.values
    
    vocab_size=10000
    
    #test data
    test_x=test_x.values
    test_yy=onehot_encoding(test_yy)
    test_yy=test_yy.values
    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    x_train= x_shuffled[:]
    y_train= y_shuffled[:]
    
    
    test_data_x=test_x[:]
    test_data_y=test_yy[:]
    
    print("Training Using Vanialla RNN in Progress, wait until the iteration ends, Thanks")    

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
            allow_soft_placement=FLAGS.allow_soft_placement,
            log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        
        
        with sess.as_default():
            rnn = RNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=vocab_size,
                embedding_size=FLAGS.embedding_dim,
                cell_type=FLAGS.cell_type,
                hidden_size=FLAGS.hidden_size,
                l2_reg_lambda=FLAGS.l2_reg_lambda
            )

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(rnn.loss, global_step=global_step)

          
            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", rnn.loss)
            acc_summary = tf.summary.scalar("accuracy", rnn.accuracy)
            
                         # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])

            # Dev summaries
            test_set_summary = tf.summary.merge([loss_summary, acc_summary])

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
           


            # Initialize all variables
            sess.run(tf.global_variables_initializer())



            # Generate batches
            batches = batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                # Train
                feed_dict = {
                    rnn.input_text: x_batch,
                    rnn.input_y: y_batch,
                    rnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, rnn.loss, rnn.accuracy], feed_dict)
                st=int(step/10)
                # Training log display
                if step % FLAGS.display_every == 0:
                    print("iteration {}, loss {:g}, acc {:g}".format(st, loss, accuracy))

              
                """"
                #########################################################################################################
                
                              Evaluate on a Test Set
                
                ########################################################################################################
                """
                if step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation Resul on a test set:")
                    feed_dict_dev = {
                        rnn.input_text: test_data_x,
                        rnn.input_y: test_data_y,
                        rnn.dropout_keep_prob: 1.0
                    }
                    summaries_dev, loss, accuracy,test_y,test_predicted = sess.run(
                        [test_set_summary, rnn.loss, rnn.accuracy,rnn.true_values,rnn.predicted_value], feed_dict_dev)
                    
                    

                    print("step {}, loss {:g}, acc {:g}\n".format(step, loss, accuracy))
                    
                    
                    """
                    ##########################################################
                    Plot ROC curve and the end of Training and test phase
                    #########################################################
                    """
                    #roc plot
                   
                    #get probability score
                    test_score=np.amax(test_predicted,axis=1)
                    rocAuc = roc_auc_score(test_y, test_score)
                    falsePositiveRate, truePositiveRate, _ = roc_curve(test_y, test_score)

                    plt.figure()
                    
                    plt.plot(falsePositiveRate, truePositiveRate, color='green',
                             lw=1, label='AUC= %0.2f)' % rocAuc)
                    plt.plot([0, 1], [0, 1], color='red', lw=3, linestyle='--',label = 'Random')
                    plt.xlim([-0.05, 1.05])
                    plt.ylim([-0.05, 1.05])
                    plt.xlabel('False Positive Rate')
                    plt.ylabel('True Positive Rate')
                    plt.title('Receiver Operating Characteristic (Vaniall RNN,test)')
                    plt.legend(loc="lower right")
                    plt.show()
                   
                    
                    """
                    plot precision and recall curve
                    """
                    #get precision and recall values
                    precision, recall, thresholds = precision_recall_curve(test_y, test_predicted[:,0], pos_label=0)
                    # average precision score
                     # precision auc
                    pr_auc = auc(recall, precision)
                    # plot
                    plt.figure(dpi=50)
                    plt.plot(recall, precision, lw=1, color='blue', label=f'AUPRC={pr_auc:.3f}')
                    plt.fill_between(recall, precision, -1, alpha=0.5)
                    plt.title('Preciion Recall Curve for Vanillaa RNN')
                    plt.xlabel('Recall')
                    plt.ylabel('Precision')
                    plt.xlim([-0.05, 1.05])
                    plt.ylim([-0.05, 1.05])
                    plt.legend()
                    plt.show()