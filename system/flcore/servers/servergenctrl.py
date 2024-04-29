# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import copy
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flcore.clients.clientgen import clientGen
from flcore.servers.serverbase import Server
from threading import Thread

import torch
import copy
import math
import pdb
import numpy as np
class KalmanFilter:
    def __init__(self, num_states, num_measurements, num_controls):
        self.A = np.eye(num_states)   # State transition matrix
        self.B = np.zeros((num_states, num_controls))  # Control matrix
        self.H = np.eye(num_measurements, num_states)  # Measurement matrix

        self.Q = np.eye(num_states)   # Process noise covariance
        self.R = np.eye(num_measurements)   # Measurement noise covariance
        self.P = np.eye(num_states)   # Initial state covariance

        self.x = np.zeros((num_states, 1))   # Initial state estimate
        self.u = np.zeros((num_controls, 1))   # Control input

    def predict(self):
        self.x = np.dot(self.A, self.x) + np.dot(self.B, self.u)
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q

    def update(self, z):
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = self.x + np.dot(K, y)
        self.P = np.dot((np.eye(self.P.shape[0]) - np.dot(K, self.H)), self.P)
def smooth_control_values(control_values):
    num_states = 1  # Number of states
    num_measurements = 1  # Number of measurements
    num_controls = 1  # Number of control inputs

    kf = KalmanFilter(num_states, num_measurements, num_controls)
    # smoothed_values = []
    # for val in control_values:
    kf.predict()
    kf.update(control_values)
    smoothed_values = kf.x[0, 0]

    return smoothed_values

class FedGenCtrl(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientGen)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.Privacy_Budget_left = []

        self.generative_model = Generative(
                                    args.noise_dim, 
                                    args.num_classes, 
                                    args.hidden_dim, 
                                    self.clients[0].feature_dim, 
                                    self.device
                                ).to(self.device)
        self.generative_optimizer = torch.optim.Adam(
            params=self.generative_model.parameters(),
            lr=args.generator_learning_rate, betas=(0.9, 0.999),
            eps=1e-08, weight_decay=0, amsgrad=False)
        self.generative_learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.generative_optimizer, gamma=args.learning_rate_decay_gamma)
        self.loss = nn.CrossEntropyLoss()
        
        self.qualified_labels = []
        for client in self.clients:
            for yy in range(self.num_classes):
                self.qualified_labels.extend([yy for _ in range(int(client.sample_per_class[yy].item()))])
        for client in self.clients:
            client.qualified_labels = self.qualified_labels

        self.server_epochs = args.server_epochs
        self.localize_feature_extractor = args.localize_feature_extractor
        if self.localize_feature_extractor:
            self.global_model = copy.deepcopy(args.model.head)
        

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            loss_threshold = torch.tensor(0.0005)  # beta * C_v
            loss_min = 0.005 #0.05 ## 0.01 is better than 0.05

            # if i%self.eval_gap == 0:
            #     print(f"\n-------------Round number: {i}-------------")
            #     print("\nEvaluate global model")
            #     self.evaluate()

            print(f"\n-------------Round number: {i}-------------")
            print("\nEvaluate global model")
            train_losses, test_losses, test_accs = self.evaluate()
            # losses, test_acc, test_num, auc = self.test_metrics()

            if self.CRD == 'CRD' and i >= 1:
                self.loss_last = test_losses[-2]

                if self.crossfade == True:
                    # test_loss_cf = (1 - i / self.global_rounds) * test_losses[-1] + i / self.global_rounds * \
                    #                test_losses[-2] ## linear crossfade
                    test_loss_cf = np.aqrt(1 - i / self.global_rounds) * test_losses[-1] + np.aqrt(
                        i / self.global_rounds) * \
                                   test_losses[-2]  ## square root crossfade
                    delta_loss = test_loss_cf - self.loss_last
                else:
                    delta_loss = test_losses[-1] - self.loss_last

                delta_loss_thrd = 0.025  # 0.05, 0.01
                if self.clipping == True:
                    delta_loss_cap = min(max(delta_loss, -delta_loss_thrd), delta_loss_thrd)
                else:
                    delta_loss_cap = delta_loss

                if delta_loss_cap > loss_threshold:
                    ### controller ###
                    if self.kalman_filter == True:
                        self.global_rounds = copy.deepcopy(min(self.global_rounds, math.floor(
                            math.ceil(smooth_control_values(iter + (test_losses[-1] - loss_min) / delta_loss_cap)))))
                    else:
                        self.global_rounds = copy.deepcopy(
                            min(self.global_rounds,
                                math.floor(math.ceil(i + (test_losses[-1] - loss_min) / delta_loss))))
                    # threshold_epochs_kf = copy.deepcopy(min(self.global_rounds, math.floor(
                    #     math.ceil(smooth_control_values(iter + (test_losses[-1] - loss_min) / delta_loss)))))
                    # pdb.set_trace()
                    # self.model = model_origin
                    # self.optimizer = torch.optim.SGD(model_origin.parameters(), lr=self.learning_rate)
                    print("*****Adjust Global Rounds *****", self.global_rounds)
                else:
                    print("*****Don't Adjust Global Rounds *****", self.global_rounds)
                    # break
            else:
                print("*****Don't Adjust Global Rounds *****", self.global_rounds)


            for client in self.selected_clients:
                client.train()

            # threads = [Thread(target=client.train)
            #            for client in self.selected_clients]
            # [t.start() for t in threads]
            # [t.join() for t in threads]

            self.receive_models()
            if self.dlg_eval and i%self.dlg_gap == 0:
                self.call_dlg(i)
            self.train_generator()
            # self.aggregate_parameters()
            self.aggregate_parameters_dp()

            self.Budget.append(time.time() - s_t)
            # print('-'*50, self.Budget[-1])
            self.Privacy_Budget_left.append(self.privacy_budget)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])
            print('-' * 25, 'privacy cost', '-' * 25, self.Privacy_Budget_left[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

            if i > self.global_rounds:
                print("Training Terminates!")
                break

        print("\nBest accuracy.")
        # self.print_(max(self.rs_test_acc), max(
        #     self.rs_train_acc), min(self.rs_train_loss))
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientGen)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()


    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model, self.generative_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                if self.localize_feature_extractor:
                    self.uploaded_models.append(client.model.head)
                else:
                    self.uploaded_models.append(client.model)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def train_generator(self):
        self.generative_model.train()

        for _ in range(self.server_epochs):
            labels = np.random.choice(self.qualified_labels, self.batch_size)
            labels = torch.LongTensor(labels).to(self.device)
            z = self.generative_model(labels)

            logits = 0
            for w, model in zip(self.uploaded_weights, self.uploaded_models):
                model.eval()
                if self.localize_feature_extractor:
                    logits += model(z) * w
                else:
                    logits += model.head(z) * w

            self.generative_optimizer.zero_grad()
            loss = self.loss(logits, labels)
            loss.backward()
            self.generative_optimizer.step()
        
        self.generative_learning_rate_scheduler.step()

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model, self.generative_model, self.qualified_labels)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()


# based on official code https://github.com/zhuangdizhu/FedGen/blob/main/FLAlgorithms/trainmodel/generator.py
class Generative(nn.Module):
    def __init__(self, noise_dim, num_classes, hidden_dim, feature_dim, device) -> None:
        super().__init__()

        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.device = device

        self.fc1 = nn.Sequential(
            nn.Linear(noise_dim + num_classes, hidden_dim), 
            nn.BatchNorm1d(hidden_dim), 
            nn.ReLU()
        )

        self.fc = nn.Linear(hidden_dim, feature_dim)

    def forward(self, labels):
        batch_size = labels.shape[0]
        eps = torch.rand((batch_size, self.noise_dim), device=self.device) # sampling from Gaussian

        y_input = F.one_hot(labels, self.num_classes)
        z = torch.cat((eps, y_input), dim=1)

        z = self.fc1(z)
        z = self.fc(z)

        return z