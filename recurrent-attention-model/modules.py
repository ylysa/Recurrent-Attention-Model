import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class QuestionNetwork(nn.Module):
    """
    This network combines the question and the image representation.
    question: current question
    feature_vector: image representation
    Returns:
    g_t: combination of the question-answer and image representation vector
    """
    def __init__(self, img_size, answer_question_size, hidden_size_img, hidden_size_answer_question):
        super().__init__()

        # image layer
        self.imageLayer = nn.Linear(img_size, hidden_size_img)
        self.imageLayer2 = nn.Linear(hidden_size_img, hidden_size_img + hidden_size_answer_question)

        # question-answer layer
        self.QALayer = nn.Linear(answer_question_size, hidden_size_answer_question)
        self.QALayer2 = nn.Linear(hidden_size_answer_question, hidden_size_img + hidden_size_answer_question)

    def forward(self, question, feature_vector):

        img_out = F.relu(self.imageLayer(feature_vector))
        img_out = F.relu(self.imageLayer2(img_out))

        aq_out = F.relu(self.QALayer(question))
        aq_out = F.relu(self.QALayer2(aq_out))

        g_t = F.relu(img_out + aq_out)
        g_t = g_t.unsqueeze(dim=0)
        g_t = g_t.unsqueeze(dim=0)
        return g_t

class CoreNetwork(nn.Module):
    """
    This network outputs the prediction.
    aq_t: combination of the question-answer and image representation vector
    h_t_prev: previous hidden states of the LSTM
    Returns:
    h_t: tuple with the output of the LSTM (hidden state of the model) and hidden states of the LSTM
    """
    def __init__(self, input_size, hidden_size):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.aq_layer = nn.Linear(input_size, hidden_size)
        self.hidden = nn.Linear(hidden_size, hidden_size)

    def forward(self, aq_t, h_t_prev):
        h1 = self.aq_layer(aq_t)
        h2 = self.hidden(h_t_prev)
        h_t = F.relu(h1 + h2)

        return h_t

class PredictionNetwork(nn.Module):
    """
    This network outputs the prediction based on the hidden state.
    h_t: the hidden state vector of the core network
    sample: set True for training, set False for evaluation during validation/testing
    Returns:
    a_t: the prediction
    """
    def __init__(self, input_size, output_size):
        super().__init__()

        self.fc = nn.Linear(input_size, output_size)


    def forward(self, h_t):
        a_t = F.log_softmax(self.fc(h_t.squeeze(0)), dim=-1)
        return a_t

class ActionSelectionNetwork(nn.Module):
    """
    This network chooses an action at each time step.
    h_t: the hidden state vector of the core network
    sample: set True for training, set False for evaluation during validation/testing
    Returns:
    action: a new action
    log_prob: the probability of the chosen action
    """
    def __init__(self, input_size, output_size):
        super().__init__()


        hid_size = input_size // 2
        self.fc = nn.Linear(input_size, hid_size)
        self.fc_lt = nn.Linear(hid_size, output_size)

    def forward(self, h_t, sample=True):
        feat = F.relu(self.fc(h_t.squeeze(0).detach()))
        mu = self.fc_lt(feat)
        l_t1 = F.softmax(mu, dim=-1)


        if sample:
            #sample an action
            categorical_dist = Categorical(l_t1)
            action = categorical_dist.sample()
            log_prob = categorical_dist.log_prob(action)

        else:
            #for evaluation take the best action
            log_prob, action = torch.max(l_t1, dim=-1)
            log_prob = torch.log(log_prob)

        return log_prob, action


class BaselineNetwork(nn.Module):
    """
    This network regresses the baseline in the
    reward function to reduce the variance of
    the gradient update.
    h_t: the hidden state vector of the core network.

    Returns:
    b_t: The baseline for the current time step.
    """

    def __init__(self, input_size, output_size):
        super().__init__()

        self.fc = nn.Linear(input_size, output_size)

    def forward(self, h_t):
        b_t = self.fc(h_t.squeeze(0).detach())
        return b_t