"""
################################################################################################################

        Answer for Homework4, Question number (2)
        English to French Machine Translator
        
        Model Structure
        
        Input=>Word_id=>Embedding=> Encoder (LSTM)=>Attention => Decoder
        
        Laste Edited, June 5,2019
        
        Author: Gaddisa Olani
###############################################################################################################

"""



import re
import string
import numpy as np
import tensorflow as tf
from imp import reload
from unicodedata import normalize
import my_sequence_model as tests
import preprocessing_2 as preprocess
from tensorflow.python.layers.core import Dense


#Read the data and do the preprocssing task
def sentence_to_seq(sentence, vocab_to_int):
    sentence = sentence.lower()
    return [vocab_to_int.get(word, vocab_to_int['<UNK>']) for word in sentence.split()]


"""
Translate new input senetence
"""
def translate_unseen_data():
    translate_sentence = 'new jersey is sometimes quiet during autumn , and it is snowy in april .'
    translate_sentence = sentence_to_seq(translate_sentence, source_vocab_to_int)
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # Load saved model
        loader = tf.train.import_meta_graph(load_path + '.meta')
        loader.restore(sess, load_path)
    
        input_data = loaded_graph.get_tensor_by_name('input:0')
        logits = loaded_graph.get_tensor_by_name('predictions:0')
        target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
        source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
        keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
    
        translate_logits = sess.run(logits, {input_data: [translate_sentence]*batch_size,
                                             target_sequence_length: [len(translate_sentence)*2]*batch_size,
                                             source_sequence_length: [len(translate_sentence)]*batch_size,
                                             keep_prob: 1.0})[0]
    
    print('Source Language (English)')
    print('  Indices of words:      {}'.format([i for i in translate_sentence]))
    print('  Input words: {}'.format([source_int_to_vocab[i] for i in translate_sentence]))
    
    print('\nTarget Language(French)')
    print('  Indices of words:      {}'.format([i for i in translate_logits]))
    print('  French words: {}'.format(" ".join([target_int_to_vocab[i] for i in translate_logits])))

"""
Convert text to ids
"""
def text_to_ids(source_text, target_text, source_vocab_to_int, target_vocab_to_int):
    source_id_text = [[source_vocab_to_int.get(word, source_vocab_to_int[word]) for word in line.split()] for line in source_text.split('\n')]
    target_id_text = [[target_vocab_to_int.get(word, target_vocab_to_int[word]) for word in line.split()] + [target_vocab_to_int['<EOS>']] for line in target_text.split('\n')]
    return source_id_text, target_id_text


"""
Build the Neural Network
"""
def model_inputs():
    input_ = tf.placeholder(tf.int32, [None, None], name="input")
    targets = tf.placeholder(tf.int32, [None, None])
    learning_rate = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    target_sequence_length = tf.placeholder(tf.int32, [None], name="target_sequence_length")
    max_target_len = tf.reduce_max(target_sequence_length, name="max_target_len")
    source_sequence_length = tf.placeholder(tf.int32, [None], name="source_sequence_length")
    
    return input_, targets, learning_rate, keep_prob, target_sequence_length, max_target_len, source_sequence_length


#Preprocess target data for encoding
def process_decoder_input(target_data, target_vocab_to_int, batch_size):
    ending = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    dec_input = tf.concat([tf.fill([batch_size, 1], target_vocab_to_int['<GO>']), ending], 1)
    return dec_input



#Create encoding layer using LSTM
def encoding_layer(rnn_inputs, rnn_size, num_layers, keep_prob, 
                   source_sequence_length, source_vocab_size, 
                   encoding_embedding_size):

    enc_embed_input = tf.contrib.layers.embed_sequence(rnn_inputs, source_vocab_size, encoding_embedding_size)
    def make_cell(rnn_size): 
        #tf.contrib.rnn.GRUCell(self._num_hidden)
        return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2)), input_keep_prob=keep_prob)

    enc_cell = tf.contrib.rnn.MultiRNNCell([make_cell(rnn_size) for _ in range(num_layers)])
    
    enc_output, enc_state = tf.nn.dynamic_rnn(enc_cell, enc_embed_input, sequence_length=source_sequence_length, dtype=tf.float32)
    return enc_output, enc_state


#Create a decoding layer for training
def decoding_layer_train(encoder_state, dec_cell, dec_embed_input, target_sequence_length, max_summary_length, output_layer, keep_prob):
    training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_embed_input,sequence_length=target_sequence_length,time_major=False)
    dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, input_keep_prob=keep_prob)
    training_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,training_helper,encoder_state,output_layer) 
    training_decoder_output = tf.contrib.seq2seq.dynamic_decode(training_decoder,impute_finished=True,maximum_iterations=max_summary_length)[0]
    
    return training_decoder_output


#Create a decoding layer for inference
def decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, start_of_sequence_id,end_of_sequence_id, max_target_sequence_length,vocab_size, output_layer, batch_size, keep_prob):
   
    start_tokens = tf.tile([target_vocab_to_int['<GO>']], [batch_size], name='start_tokens')

    inference_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(dec_embeddings,start_tokens,target_vocab_to_int['<EOS>'])

    inference_decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell,inference_helper,encoder_state,output_layer)

    inference_decoder_output = tf.contrib.seq2seq.dynamic_decode(inference_decoder,impute_finished=True,maximum_iterations=max_target_sequence_length)[0]
    
    return inference_decoder_output

# save a list of clean sentences to file
def save_prediction_sentences(sentences, filename):
    from pickle import dump
    dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)

#Create decoding layer
def decoding_layer(dec_input, encoder_state,target_sequence_length, max_target_sequence_length,rnn_size, num_layers, target_vocab_to_int, target_vocab_size,batch_size, keep_prob, decoding_embedding_size):
    dec_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, decoding_embedding_size], -1.0, 1.0))
    embedded = tf.nn.embedding_lookup(dec_embeddings, dec_input)
    # RNN Cell, with dropout
    def make_cell():
        return tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(rnn_size, initializer=tf.random_uniform_initializer(-0.1, 0.1)), output_keep_prob=keep_prob)
    
    dec_cell = tf.contrib.rnn.MultiRNNCell([make_cell() for _ in range(num_layers)])
    # Define a linear dense layer to map outputs to words
    output_layer = Dense(target_vocab_size, kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
    # Training & Inference Decoder
    with tf.variable_scope("decode") as decoding_scope:
        training_decoder_output = decoding_layer_train(encoder_state, dec_cell, embedded, target_sequence_length, max_target_sequence_length, output_layer, keep_prob)
        decoding_scope.reuse_variables()
        inference_decoder_output = decoding_layer_infer(encoder_state, dec_cell, dec_embeddings, target_vocab_to_int['<GO>'], target_vocab_to_int['<EOS>'], max_target_sequence_length, target_vocab_size, output_layer, batch_size, keep_prob)
        
            
    return training_decoder_output, inference_decoder_output


#Build the Sequence-to-Sequence part of the neural network
def seq2seq_model(input_data, target_data, keep_prob, batch_size,source_sequence_length, target_sequence_length,max_target_sentence_length,source_vocab_size, target_vocab_size,enc_embedding_size, dec_embedding_size,rnn_size, num_layers, target_vocab_to_int):
    _, enc_state = encoding_layer(input_data, rnn_size, num_layers, keep_prob,source_sequence_length,source_vocab_size, enc_embedding_size)
    dec_input = process_decoder_input(target_data, target_vocab_to_int, batch_size)
    training_dec_logits_out, inference_dec_logits_out = decoding_layer(dec_input,enc_state,target_sequence_length,max_target_sentence_length,rnn_size,num_layers,target_vocab_to_int, target_vocab_size,batch_size,keep_prob,dec_embedding_size) 
    return training_dec_logits_out, inference_dec_logits_out



def pad_sentence_batch(sentence_batch, pad_int):
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]

def get_batches(sources, targets, batch_size, source_pad_int, target_pad_int):
    for batch_i in range(0, len(sources)//batch_size):
        start_i = batch_i * batch_size

        # Slice the right amount for the batch
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]

        # Pad
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # Need the lengths for the _lengths parameters
        pad_targets_lengths = []
        for target in pad_targets_batch:
            pad_targets_lengths.append(len(target))

        pad_source_lengths = []
        for source in pad_sources_batch:
            pad_source_lengths.append(len(source))

        yield pad_sources_batch, pad_targets_batch, pad_source_lengths, pad_targets_lengths
        

"""
###############################################################

    TRAIN THE NETWORK

###############################################################
"""
def get_accuracy(target, logits):
    max_seq = max(target.shape[1], logits.shape[1])
    if max_seq - target.shape[1]:
        target = np.pad(
            target,
            [(0,0),(0,max_seq - target.shape[1])],
            'constant')
    if max_seq - logits.shape[1]:
        logits = np.pad(logits,[(0,0),(0,max_seq - logits.shape[1])],'constant')
    return np.mean(np.equal(target, logits))


def begin_trainig():                                                                                                                                                                                                          
    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())
    
        for epoch_i in range(epochs):
            for batch_i, (source_batch, target_batch, sources_lengths, targets_lengths) in enumerate(
                    get_batches(train_source, train_target, batch_size,source_vocab_to_int['<PAD>'],target_vocab_to_int['<PAD>'])):
    
                _, loss = sess.run([train_op, cost],{input_data: source_batch,targets: target_batch,lr: learning_rate,target_sequence_length: targets_lengths,source_sequence_length: sources_lengths,keep_prob: keep_probability})
    
    
                if batch_i % display_step == 0 and batch_i > 0:
                    batch_train_logits = sess.run(inference_logits,{input_data: source_batch,source_sequence_length: sources_lengths,target_sequence_length: targets_lengths,keep_prob: 1.0})
                    batch_valid_logits = sess.run(inference_logits,{input_data: valid_sources_batch,source_sequence_length: valid_sources_lengths,target_sequence_length: valid_targets_lengths,keep_prob: 1.0})
                    train_acc = get_accuracy(target_batch, batch_train_logits)
                    valid_acc = get_accuracy(valid_targets_batch, batch_valid_logits)
                    loss_list.append(loss)
                    valid_acc_list.append(valid_acc)
                    print('Epoch {:>3} Batch {:>4}/{} - Train Accuracy: {:>6.4f}, Validation Accuracy: {:>6.4f}, Loss: {:>6.4f}'
                          .format(epoch_i, batch_i, len(source_int_text) // batch_size, train_acc, valid_acc, loss))
    
        # Save Model
        saver = tf.train.Saver()
        saver.save(sess, save_path)
        print('Congratulations English to Frech Translator Model is Trained and Saved')

# read test file
def read_testfile(lines):
	cleaned = list()
	# prepare regex for char filtering
	re_print = re.compile('[^%s]' % re.escape(string.printable))
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for line in lines:
		# normalize unicode characters
		line = normalize('NFD', line).encode('ascii', 'ignore')
		line = line.decode('UTF-8')
		# tokenize on white space
		line = line.split()
		# convert to lower case
		line = [word.lower() for word in line]
		# remove punctuation from each token
		line = [word.translate(table) for word in line]
		# remove non-printable chars form each token
		line = [re_print.sub('', w) for w in line]
		# remove tokens with numbers in them
		line = [word for word in line if word.isalpha()]
		# store as string
		cleaned.append(' '.join(line))
	return cleaned
if __name__ == "__main__":
    
    """
    ################################################################################################
    
         Select all and run it to get the answer for question 2
         The last line of code in this file is to show the accuracy and sample output
    
    ################################################################################################
    """
    
    source_path = 'data/en.txt'
    target_path = 'data/fr.txt'
    source_text = preprocess.load_data(source_path)
    target_text = preprocess.load_data(target_path)
    
    
    print("Loading the en.txt and fr.txt completed: ")
    
    print("Begin Data Preprocssing")
    
    tests.test_text_to_ids(text_to_ids)

    #Preprocess all the data and save it
    preprocess.preprocess_and_save_data(source_path, target_path, text_to_ids)
    (source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = preprocess.load_preprocess()
    
    tests.test_model_inputs(model_inputs)
    tests.test_process_encoding_input(process_decoder_input)

    #Implement encoding_layer() to create a Encoder RNN layer:
    reload(tests)
    tests.test_encoding_layer(encoding_layer)
    tests.test_decoding_layer_train(decoding_layer_train)
    tests.test_decoding_layer_infer(decoding_layer_infer)
    tests.test_decoding_layer(decoding_layer)
    tests.test_seq2seq_model(seq2seq_model)
    
    epochs = 10
    batch_size = 256
    rnn_size = 256
    num_layers = 2
    encoding_embedding_size = 256
    decoding_embedding_size = 256
    learning_rate = .0001
    keep_probability = .9
    display_step = 10
    
    #Build the tensor graph
    save_path = 'checkpoints/dev'
    (source_int_text, target_int_text), (source_vocab_to_int, target_vocab_to_int), _ = preprocess.load_preprocess()
    max_target_sentence_length = max([len(sentence) for sentence in source_int_text])
    
    train_graph = tf.Graph()
    
    
    
    with train_graph.as_default():
        input_data, targets, lr, keep_prob, target_sequence_length, max_target_sequence_length, source_sequence_length = model_inputs()
        input_shape = tf.shape(input_data)
    
        train_logits, inference_logits = seq2seq_model(tf.reverse(input_data, [-1]),targets,keep_prob,batch_size,source_sequence_length,target_sequence_length,max_target_sequence_length,len(source_vocab_to_int),len(target_vocab_to_int),encoding_embedding_size,decoding_embedding_size,rnn_size,num_layers,target_vocab_to_int)
        training_logits = tf.identity(train_logits.rnn_output, name='logits')
        inference_logits = tf.identity(inference_logits.sample_id, name='predictions')
    
        masks = tf.sequence_mask(target_sequence_length, max_target_sequence_length, dtype=tf.float32, name='masks')
    
        with tf.name_scope("optimization"):
            # Loss function
            cost = tf.contrib.seq2seq.sequence_loss(
                training_logits,
                targets,
                masks)
    
            # Optimizer
            optimizer = tf.train.AdamOptimizer(lr)
    
            # Gradient Clipping
            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)
        # Split data to training and validation sets
    train_source = source_int_text[batch_size:]
    train_target = target_int_text[batch_size:]
    valid_source = source_int_text[:batch_size]
    valid_target = target_int_text[:batch_size]
    (valid_sources_batch, valid_targets_batch, valid_sources_lengths, valid_targets_lengths ) = next(get_batches(valid_source,valid_target,batch_size,source_vocab_to_int['<PAD>'],target_vocab_to_int['<PAD>']))
    loss_list=[]
    valid_acc_list=[]
    
    begin_trainig()
    
    # Plot the loss and accuracy
    import matplotlib.pyplot as plt
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    ax1.plot(loss_list, color='green')
    ax1.set_title('Training Loss')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('Loss value')
    ax2.plot(valid_acc_list)
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Test Accuracy')
    plt.show()
    
    #Save parameters for prediction
    preprocess.save_params(save_path)
    _, (source_vocab_to_int, target_vocab_to_int), (source_int_to_vocab, target_int_to_vocab) = preprocess.load_preprocess()
    load_path = preprocess.load_params()
    
    #Translate unseen data
    tests.test_sentence_to_seq(sentence_to_seq)
    translate_unseen_data()
    
