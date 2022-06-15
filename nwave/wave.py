'''
==========
Date: Apr 8, 2022
Maintantainer: Xinyi Zhong (xinyi.zhong@yale.edu)
==========
Encapsulate dictlearner and neurodynamics to form a wave object

'''

class Wave():
    '''
    
    Methods
    -------
    train_dictionary : 
    get_excitation : 
    
    Attributes
    -------
    dict_learner : DictLearner 
    neuron_dynmic_model : NeuronDynamicsModel
    '''

    def __init__(self, dict_learner, neuron_dynamic_model) -> None:
        self.dict_learner = dict_learner
        self.neuron_dynamic_model = neuron_dynamic_model

    def train_dictionary(self, word_batch):
        # Get stimulus 
        stimulus = self.dict_learner.perceive_to_get_stimulus(word_batch) 
        # Get neuron activity
        activation = self.neuron_dynamic_model.stimulate(stimulus)
        # Neuron model evolve and reset
        self.neuron_dynamic_model.evolve()
        self.neuron_dynamic_model.reset()
        # Update codebook 
        l0, l1, e2 = self.dict_learner.update_codebook(word_batch, activation)

    def save(self, fpath):
        self.neuron_dynamic_model.dump(fpath)
        self.dict_learner.dump(fpath)
    
    def load(self, fpath):
        self.neuron_dynamic_model.load(fpath)
        self.dict_learner.load(fpath)
