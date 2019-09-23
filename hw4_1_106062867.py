"""
###################################################################################################
    Last updated on un Jun  2 15:40:24 2019
    This code contains all the answer to question 1 from a to g
    main function
    
    Structure of my CNN Network
        Input Layer
        Embedding layer(with relu)
        Recurrent Layer
        Droupout
        Dense Layer
        output layer(with Sigmoid)
    AdamOptimizer with cross entropy loss is used for learning
    
    The code for each model is kept separatelt but you can run each of them one by one
    by slecting them inside the main function
    
    Thank you!
    
    @author: gaddisa olani
###################################################################################################
"""
import lstm_model
import lstm_model2
import gru_model
import vanilla_rnn_model as vanilla
import warnings
warnings.simplefilter("ignore", UserWarning)
lstm_model

if __name__ == "__main__":
    
    """
    ################################################################################################
    #### Answer for the firts quetsion usin Vanillaa RNN, LSTM and GRU      ########################
    ####   The accuracy will be shown on each iteration but the evaluation  ########################
    ###     will be done at the end of iteration                            ########################
    ###     The ROC Curve and PC will be shown at the end of itaration      ########################
    ###        RUN EACH OF THEM ONE BY ONE                                      ####################
    ################################################################################################
    """
    lstm_model.train_LSTM()
    vanilla.train_vanilla_rnn()
    gru_model.train_gru()
    
    """
    ###############################################################################################
    After Changing the Length of sequence to 256 (as exmaple for LSTM)
    ##############################################################################################
    """
    lstm_model2.train_LSTM_length_256()
    