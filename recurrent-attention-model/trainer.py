import os
import time
import shutil
import random
import numpy
import generate_dataset
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from model import RecurrentAttention
from utils import AverageMeter

class Trainer:
    def __init__(self, config, dataset, valdataset, testdataset, images, valimages, testimages):
        self.config = config

        if config.use_gpu and torch.cuda.is_available():
            print(torch.cuda.device_count())

            self.device = torch.device("cuda:1")
        else:
            self.device = torch.device("cpu")
            print("set cpu")

        self.tb = SummaryWriter()
        self.dataset = dataset
        self.valdataset = valdataset
        self.testdataset = testdataset
        self.images = images
        self.valimages = valimages
        self.testimages = testimages

        self.qa_hidden = config.qa_hidden
        self.img_hidden = config.img_hidden

        # core network params
        self.hidden_size = config.hidden_size

        # reinforce params
        self.std = config.std
        self.M = config.M

        # data params
        if config.is_train:
            self.num_train = len(self.dataset)
            self.num_valid = len(self.valdataset)
        else:
            self.num_test = len(self.testdataset)

        # training params
        self.epochs = config.epochs
        self.pretraining_epochs = config.pretraining_epochs
        self.start_epoch = 0
        self.momentum = config.momentum
        self.lr = config.init_lr

        # misc params
        self.best = config.best
        self.ckpt_dir = config.ckpt_dir
        self.logs_dir = config.logs_dir
        self.best_valid_acc = 0.0
        self.counter = 0
        self.lr_patience = config.lr_patience
        self.train_patience = config.train_patience
        self.resume = config.resume
        self.print_freq = config.print_freq
        self.model_name = "ram_model"

        self.plot_dir = "./plots/" + self.model_name + "/"
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        # build RAM model
        self.num_questions = 8
        self.num_prediction = 4
        self.max_actions = 14
        self.model = RecurrentAttention(img_size=24, num_questions=self.num_questions, num_predictions=self.num_prediction,
                                        hidden_size_img=self.img_hidden, hidden_size_qa=self.qa_hidden, hidden_size_rnn=self.hidden_size)
        self.model.to(self.device)

        # initialize optimizer and scheduler
        #  self.optimizer = torch.optim.Adam(
        #      self.model.parameters(), lr=self.config.init_lr
        #      )
        self.optimizer = torch.optim.Adam([{'params': self.model.rnn.parameters()},
                                            {'params': self.model.classifier.parameters()},
                                            {'params': self.model.sensor.parameters()},
                                            {'params': self.model.baseliner.parameters()},
                                            {'params': self.model.locator.parameters()}], lr=self.config.init_lr)

        #self.scheduler = ReduceLROnPlateau(
        #    self.optimizer, "min", patience=self.lr_patience
        #)

    def reset(self):
        h_t = torch.zeros(
            1,
            self.hidden_size,
            dtype=torch.float,
            device=self.device,
            requires_grad=True,
        )
        l_t = torch.zeros(
            self.num_questions,
            dtype=torch.float,
            device=self.device,
            requires_grad=True,
        )
        return h_t, l_t

    def train(self):
        """Train the model on the training set.

        A checkpoint of the model is saved after each epoch
        and if the validation accuracy is improved upon,
        a separate ckpt is created for use on the test set.
        """
        # load the most recent checkpoint
        if self.resume:
            self.load_checkpoint(best=False)

        print(
            "\n[*] Train on {} samples, validate on {} samples".format(
                self.num_train, self.num_valid
            )
        )
        accs=[]
        val = []
        eplen = []
        eplenval = []
        reward = []
        latencyreward = []
        loss = []
        val_loss = []

        try:
            for epoch in range(self.start_epoch, self.epochs):
                if epoch < self.pretraining_epochs:
                    for param in self.optimizer.param_groups[4]['params']:
                        param.requires_grad = False
                else:
                    for param in self.optimizer.param_groups[4]['params']:
                        param.requires_grad = True
                    for param in self.optimizer.param_groups[0]['params']:
                        param.requires_grad = False
                    for param in self.optimizer.param_groups[1]['params']:
                        param.requires_grad = False
                    for param in self.optimizer.param_groups[2]['params']:
                        param.requires_grad = False
                    for param in self.optimizer.param_groups[3]['params']:
                        param.requires_grad = False
                print(
                    "\nEpoch: {}/{} - LR: {:.6f}".format(
                        epoch + 1, self.epochs, self.optimizer.param_groups[0]["lr"]
                    )
                )

                # train for 1 epoch
                c = list(zip(self.dataset, self.images))
                random.shuffle(c)
                self.dataset, self.images = zip(*c)
                self.dataset, self.images = numpy.asarray(self.dataset), numpy.asarray(self.images)
                train_loss, train_acc, elen, rewards, latencyrewards = self.train_one_epoch(epoch)

                self.tb.add_scalar("Loss", train_loss, epoch)
                self.tb.add_scalar("Accuracy", train_acc, epoch)
                self.tb.add_scalar("Reward", rewards, epoch)
                self.tb.add_scalar("Latency reward", latencyrewards, epoch)
                self.tb.add_scalar("Episode length", elen, epoch)

                # evaluate on validation set
                valid_loss, valid_acc, elen_val = self.validate(epoch)
                self.tb.add_scalar("Validation loss", valid_loss, epoch)
                self.tb.add_scalar("Validation accuracy", valid_acc, epoch)
                self.tb.add_scalar("Validation episode length", elen_val, epoch)
                print(f"eplen train: {elen} eplen val:{elen_val}")
                accs.append(train_acc)
                val.append(valid_acc)
                reward.append(rewards)
                latencyreward.append(latencyrewards)
                loss.append(train_loss)
                val_loss.append(valid_loss)
                eplen.append(elen)
                eplenval.append(elen_val)
                # # reduce lr if validation loss plateaus
                #  self.scheduler.step(-valid_acc)

                is_best = valid_acc > self.best_valid_acc
                msg1 = "train loss: {:.3f} - train acc: {:.3f} "
                msg2 = "- val loss: {:.3f} - val acc: {:.3f} - val err: {:.3f}"
                if is_best:
                    self.counter = 0
                    msg1 += " [*]"
                msg = msg1 + msg2
                print(
                    msg.format(
                        train_loss, train_acc, valid_loss, valid_acc, 100 - valid_acc
                    )
                )

                # check for improvement
                #   if not is_best:
                #      self.counter += 1
                #   if self.counter > self.train_patience:
                #      print("[!] No improvement in a while, stopping training.")
                #       return
                self.best_valid_acc = max(valid_acc, self.best_valid_acc)
                self.save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        "model_state": self.model.state_dict(),
                        "optim_state": self.optimizer.state_dict(),
                        "best_valid_acc": self.best_valid_acc,
                    },
                    is_best,
                )

        except KeyboardInterrupt:
            # Introduce a line break after ^C is displayed so save message
            # is on its own line.
            print("interrupted..")
        finally:
            print("save your model")

    def checkQuestion(self, sentence):
        if len(sentence) == 1:
            return True
        elif len(sentence) == 2:
            if sentence[0] > 1 and sentence[1] < 2:
                return True
            elif sentence[0] > 3 and 4 > sentence[1] > 1:
                return True
            else:
                return False
        elif len(sentence) == 3:
            if sentence[0] > 3 and 1 < sentence[1] < 4 and sentence[2] < 2:
                return True
            else:
                return False
        else:
            return False

    def getContrast(self, sentence):
        try:
            attribute1 = sentence[0]
        except:
            attribute1 = -1
        try:
            attribute2 = sentence[1]
        except:
            attribute2 = -1
        contrast = [0., 0.]
        if attribute1 == 2 or attribute2 == 2:
            contrast = [1., 0.]
        elif attribute1 == 3 or attribute2 == 3:
            contrast = [0., 1.]
        return torch.tensor(contrast)

    def getColor(self, sentence):
        color = torch.tensor([0., 0.])
        if len(sentence) == 1 and sentence[0] < 2:
            color = F.one_hot(sentence[0], 2)
        elif len(sentence) == 2:
            if sentence[1] < 2:
                color = F.one_hot(sentence[1], 2)
        elif len(sentence) == 3:
            if sentence[2] < 2:
                color = F.one_hot(sentence[2], 2)
        return color

    def getContext(self, sentence):
        context = [0., 0.]
        if sentence[0] == 4:
            context = [1., 0.]
        elif sentence[0] == 5:
            context = [0., 1.]
        return torch.tensor(context)

    def getAnswer(self, img, selected, question):
        aqmap ={
            "[4, 2, 0]": 'foreground light blue',
            "[4, 2, 1]": 'foreground light green',
            "[4, 3, 0]": 'foreground dark blue',
            "[4, 3, 1]": 'foreground dark green',
            "[5, 2, 0]": 'background light blue',
            "[5, 2, 1]": 'background light green',
            "[5, 3, 0]": 'background dark blue',
            "[5, 3, 1]": 'background dark green',
            "[2, 0]": 'light blue',
            "[2, 1]": 'light green',
            "[3, 0]": 'dark blue',
            "[3, 1]": 'dark green',
            "[0]": 'blue',
            "[1]": 'green',
            "[4, 0]": 'foreground blue',
            "[4, 1]": 'foreground green',
            "[5, 0]": 'background blue',
            "[5, 1]": 'background green',
            "[4, 2]": 'foreground light',
            "[4, 3]": 'foreground dark',
            "[5, 2]": 'background light',
            "[5, 3]": 'background dark'
        }
        answers = []
        if len(question) == 1 and question[0] > 1:
            answers.append(1)
        else:
            question = [x.item() for x in question]
            question = aqmap.get(str(question))
            s = img[selected.item()].values()
            s = list(s)
            answer = 1
            if any(e not in s[1] for e in question.split()):
                answer = 0
            if answer == 0:
                if all(e in s[2] for e in question.split()):
                    answer = 1
            answers.append(answer)
        return answers

    def train_one_epoch(self, epoch):
        self.model.train()
        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()
        rewards = AverageMeter()
        latencyrewards = AverageMeter()

        tic = time.time()

        eplen = []
        with tqdm(total=self.num_train) as pbar:

            for i in range(self.num_train):
                self.optimizer.zero_grad()
                x = self.dataset[i]
                y = torch.tensor([numpy.random.randint(0, 4)])
                vector = self.images[i]

                if epoch < self.pretraining_epochs:
                    q = generate_dataset.generateSeqQA(x, y)
                # initialize location vector and hidden state
                h_t, l_t = self.reset()
                h_t = h_t.reshape((1, 1, 256)), torch.zeros(256).reshape((1, 1, 256))
                vector = torch.tensor(vector)
                vector.requires_grad = True
                vector = vector.to(self.device).to(dtype=torch.float)

                log_pi = []
                baselines = []
                sentence = []
                questionCount = 0
                endOfQuestion = False

                for t in range(1, self.max_actions):
                    out, h_t, action, b_t, action_log_pi = self.model(l_t, h_t, vector, evaluate=False)
                    if epoch < self.pretraining_epochs:
                        action = q[t-1]
                        action = torch.tensor([action])
                    baselines.append(b_t)
                    log_pi.append(action_log_pi)

                    #last action used as stop
                    if (7) == action:
                        break
                    if not (action == 6):
                        sentence.append(action)
                    else:
                        endOfQuestion = True
                    if self.checkQuestion(sentence):
                        context = self.getContext(sentence)
                        contrast = self.getContrast(sentence)
                        color = self.getColor(sentence)
                        if endOfQuestion:
                            answer = self.getAnswer(x, y, sentence)
                            answer = F.one_hot(torch.tensor(answer), 2)
                        else:
                            answer = torch.tensor([0., 0.])
                    else:
                        context = torch.tensor([0., 0.])
                        contrast = torch.tensor([0., 0.])
                        color = torch.tensor([0., 0.])
                        answer = torch.tensor([0., 0.])
                    context = context.squeeze(0)
                    contrast = contrast.squeeze(0)
                    color = color.squeeze(0)
                    answer = answer.squeeze(0)
                    l_t = torch.cat((context, contrast), dim=0)
                    l_t = torch.cat((l_t, color), dim=0)
                    l_t = torch.cat((l_t, answer), dim=0)
                    l_t = l_t.to(self.device).to(dtype=torch.float)
                    if len(sentence) == 4 or endOfQuestion:
                        sentence = []
                        questionCount += 1
                        endOfQuestion = False

                log_probas = self.model.getPrediction(out)

                baselines = torch.stack(baselines)
                log_pi = torch.stack(log_pi)
                predicted = torch.max(log_probas, 1)[1]

                R = (predicted.detach().to("cpu") == y).float()

                eplen.append(t)

                #latency reward penalizes the number of actions
                #latency_reward = 1./(t+1.)
                #latency reward penalizes the number of questions
                latency_reward = 1./(questionCount+2.)
                if R == 1.:
                    R = torch.tensor([1.+latency_reward])
                else:
                    R = torch.tensor([-1.])
                    #R = torch.tensor([-1+latency_reward])
                R = R.unsqueeze(1).repeat(1, t)

                # compute losses for differentiable modules
                loss_action = F.nll_loss(log_probas.to("cpu"), y)
                loss_baseline = F.mse_loss(baselines.to("cpu"), R[0])

                # compute reinforce loss
                adjusted_reward = R[0] - baselines.detach().to("cpu")
                loss_reinforce = torch.sum(-log_pi * adjusted_reward[0])

                # sum up into a hybrid loss
                if epoch < self.pretraining_epochs:
                    loss = loss_action + loss_baseline
                else:
                    loss = loss_reinforce


                # compute accuracy
                correct = (predicted.to("cpu") == y).float()
                acc = 100 * (correct.sum() / len(y))

                # store
                losses.update(loss.item(), 1)
                accs.update(acc.item(), 1)

                # compute gradients and update SGD
                loss.backward()
                self.optimizer.step()

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc - tic)

                pbar.set_description(
                    (
                        "{:.1f}s - loss: {:.3f} - acc: {:.3f}".format(
                            (toc - tic), loss.item(), acc.item()
                        )
                    )
                )
                pbar.update(1)
            eplen = numpy.mean(numpy.asarray(eplen).reshape(-1,))
            return losses.avg, accs.avg, eplen, rewards.avg, latencyrewards.avg

    @torch.no_grad()
    def validate(self, epoch):
        """Evaluate the RAM model on the validation set.
        """
        losses = AverageMeter()
        accs = AverageMeter()
        eplen = []

        for i in range(self.num_valid):
            x = self.valdataset[i]
            y = torch.tensor([numpy.random.randint(0, 4)])
            vector = self.valimages[i]
            if epoch < self.pretraining_epochs:
                q = generate_dataset.generateSeqQA(x, y)

            # initialize location vector and hidden state
            h_t, l_t = self.reset()
            h_t = h_t.reshape((1, 1, 256)), torch.zeros(256).reshape((1, 1, 256))
            vector = torch.tensor(vector)

            vector.requires_grad = False
            vector = vector.to(self.device).to(dtype=torch.float)

            log_pi = []
            baselines = []
            sentence = []
            questionCount = 0
            endOfQuestion = False
            for t in range(1, self.max_actions):
                out, h_t, action, b_t, action_log_pi = self.model(l_t, h_t, vector, evaluate=True)
                # store
                baselines.append(b_t)
                log_pi.append(action_log_pi)
                if epoch < self.pretraining_epochs:
                    action = q[t-1]
                    action = torch.tensor([action])

                if (7) == action:
                    break
                if not (action == 6):
                    sentence.append(action)
                else:
                    endOfQuestion = True

                if self.checkQuestion(sentence):
                    context = self.getContext(sentence)
                    contrast = self.getContrast(sentence)
                    color = self.getColor(sentence)
                    if endOfQuestion:
                        answer = self.getAnswer(x, y, sentence)
                        answer = F.one_hot(torch.tensor(answer), 2)
                    else:
                        answer = torch.tensor([0., 0.])
                else:
                    context = torch.tensor([0., 0.])
                    contrast = torch.tensor([0., 0.])
                    color = torch.tensor([0., 0.])
                    answer = torch.tensor([0., 0.])
                context = context.squeeze(0)
                contrast = contrast.squeeze(0)
                color = color.squeeze(0)
                answer = answer.squeeze(0)

                l_t = torch.cat((context, contrast), dim=0)
                l_t = torch.cat((l_t, color), dim=0)
                l_t = torch.cat((l_t, answer), dim=0)

                if len(sentence) == 4 or endOfQuestion:
                    sentence = []
                    questionCount += 1
                    endOfQuestion = False

            log_probas = self.model.getPrediction(out)

            baselines = torch.stack(baselines)
            log_pi = torch.stack(log_pi)

            eplen.append(t)

            # average
            log_probas = log_probas.view(self.M, -1, log_probas.shape[-1])
            log_probas = torch.mean(log_probas, dim=0)
            baselines = baselines.contiguous().view(self.M, -1, baselines.shape[-1])
            baselines = torch.mean(baselines, dim=0)

            log_pi = log_pi.contiguous().view(self.M, -1, log_pi.shape[-1])
            log_pi = torch.mean(log_pi, dim=0)

            # calculate reward
            predicted = torch.max(log_probas, 1)[1]
            R = (predicted.detach().to("cpu") == y).float()


            latency_reward = 1./(questionCount+2.)
            #latency_reward = 1./(t+1.)
            if R == 1:
                R = torch.tensor([1.+latency_reward])
            else:
                R = torch.tensor([-1.])
               # R = torch.tensor([-1.+latency_reward])

            R = R.unsqueeze(1).repeat(1, t)

            # compute losses for differentiable modules
            loss_action = F.nll_loss(log_probas.to("cpu"), y)

            loss_baseline = F.mse_loss(baselines.to("cpu"), R)


            # compute reinforce loss
            adjusted_reward = R - baselines.detach().to("cpu")
            loss_reinforce = torch.sum(-log_pi.to("cpu") * adjusted_reward[0])

            # sum up into a hybrid loss
            if epoch < self.pretraining_epochs:
                loss = loss_action + loss_baseline
            else:
                loss = loss_reinforce
            # compute accuracy
            correct = (predicted.to("cpu") == y).float()
            acc = 100 * (correct.sum() / len(y))
            # store
            losses.update(loss.item(), 1)
            accs.update(acc.item(), 1)

        eplen = numpy.mean(numpy.asarray(eplen).reshape(-1,))
        return losses.avg, accs.avg, eplen

    @torch.no_grad()
    def test(self):
        correct = 0

        # load the most recent checkpoint
        self.load_checkpoint(best=False)
        for i in range(self.num_test):
            x = self.testdataset[i]
            y = torch.tensor([numpy.random.randint(0, 4)])
            vector = self.testimages[i]

            h_t, l_t = self.reset()
            h_t = h_t.reshape((1, 1, 256)), torch.zeros(256).reshape((1, 1, 256))
            vector = torch.tensor(vector).to(dtype=torch.float)

            sentence = []
            questionCount = 0
            endOfQuestion = False

            for t in range(1, self.max_actions):
                out, h_t, action, b_t, action_log_pi = self.model(l_t, h_t, vector, evaluate=True)
                if (7) == action:
                    break
                if not (action == 6):
                    sentence.append(action)
                else:
                    endOfQuestion = True
                if self.checkQuestion(sentence):
                    context = self.getContext(sentence)
                    contrast = self.getContrast(sentence)
                    color = self.getColor(sentence).squeeze(0)
                    if endOfQuestion:
                        answer = self.getAnswer(x, y, sentence)
                        answer = (F.one_hot(torch.tensor(answer), 2)).squeeze(0)
                    else:
                        answer = torch.tensor([0., 0.])
                else:
                    context = torch.tensor([0., 0.])
                    contrast = torch.tensor([0., 0.])
                    color = torch.tensor([0., 0.])
                    answer = torch.tensor([0., 0.])
                l_t = torch.cat((context, contrast), dim=0)
                l_t = torch.cat((l_t, color), dim=0)
                l_t = torch.cat((l_t, answer), dim=0)

                if len(sentence) == 4 or endOfQuestion:
                    sentence = []
                    questionCount += 1
                    endOfQuestion = False
            log_probas = self.model.getPrediction(out)
            pred = log_probas.data.max(1, keepdim=True)[1]
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()

        perc = (100.0 * correct)/self.num_test
        error = 100 - perc
        print(
            "[*] Test Acc: {}/{} ({:.2f}% - {:.2f}%)".format(
                correct, self.num_test, perc, error
            )
        )

    def save_checkpoint(self, state, is_best):
        filename = self.model_name + "_ckpt.pth.tar"
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)
        if is_best:
            filename = self.model_name + "_model_best.pth.tar"
            shutil.copyfile(ckpt_path, os.path.join(self.ckpt_dir, filename))

    def load_checkpoint(self, best=False):
        """
        best: if set to True, loads the model with the best validation accuracy.
              if set to False, loads the most recent model.
        """
        print("[*] Loading model from {}".format(self.ckpt_dir))

        filename = self.model_name + "_ckpt.pth.tar"
        if best:
            filename = self.model_name + "_model_best.pth.tar"
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)

        # load variables from checkpoint
        self.start_epoch = ckpt["epoch"]
        self.best_valid_acc = ckpt["best_valid_acc"]
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optim_state"])

        if best:
            print(
                "[*] Loaded {} checkpoint @ epoch {} "
                "with best valid acc of {:.3f}".format(
                    filename, ckpt["epoch"], ckpt["best_valid_acc"]
                )
            )
        else:
            print("[*] Loaded {} checkpoint @ epoch {}".format(filename, ckpt["epoch"]))
