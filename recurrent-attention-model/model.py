import torch.nn as nn
import modules


class RecurrentAttention(nn.Module):
    """
    # img_size = size of the image representation
    # num_questions - the number of possible actions to take
    # answer_size - the size of the answer
    # num_predictions - number of prediction that can be made at the end
    # hidden_size_img - hidden size for the image representation
    # hidden_size_qa - hidden size for the qustion-answer pair representation
    # hidden_size_rnn - size of the hidden state
    #
    """
    def __init__(
            self, img_size, num_questions, num_predictions, hidden_size_img, hidden_size_qa, hidden_size_rnn
    ):


        super().__init__()

        self.sensor = modules.QuestionNetwork(img_size, num_questions, hidden_size_img, hidden_size_qa)
        self.rnn = nn.LSTM(256, hidden_size_rnn, batch_first=True)
        self.locator = modules.ActionSelectionNetwork(hidden_size_rnn, 8)
        self.classifier = modules.PredictionNetwork(hidden_size_rnn, num_predictions)

        self.baseliner = modules.BaselineNetwork(hidden_size_rnn, 1)

    def forward(self, question, h_t_prev, feature_vector, evaluate=False):
        """
        # question: the current question
        # h_t_prev: the previous state of the model
        # feature_vector: vector representation of the image
        # evaluate: set False for training, set True for evaluation during validation/testing
        Returns:
        # out: the hidden state of the model
        # h_t: LSTM hidden states
        # action: selected action for the current time step
        # b_t: the baseline for the current time step
        # action_log_pi: propability of the action
        """
        aq_img = self.sensor(question, feature_vector)
        out, h_t = self.rnn(aq_img, h_t_prev)
        action_log_pi, action = self.locator(out, sample=not evaluate)
        b_t = self.baseliner(out).squeeze()

        return out, h_t, action, b_t, action_log_pi


    # target prediction
    def getPrediction(self, out):
        log_probas = self.classifier(out)
        return log_probas
