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
import torch
import numpy as np
import time
from flcore.clients.clientbase import Client
from utils.privacy import *


class clientDPSUR(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

    def train(self):
        trainloader = self.load_train_data()
        # self.model.to(self.device)
        self.model.train()
        model_origin = copy.deepcopy(self.model)

        start_time = time.time()
        loss_threshold = torch.tensor(-0.001)  # beta * C_v
        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max_local_epochs // 2)

        for epoch in range(max_local_epochs):
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            losses, test_acc, test_num, auc = self.test_metrics()
            loss_client = losses / test_num
            if self.loss_last:
                if loss_client - self.loss_last > loss_threshold:
                    self.optimizer = torch.optim.SGD(model_origin.parameters(), lr=self.learning_rate)
                    print("*****reject*****", loss_client - self.loss_last, loss_client, self.loss_last)
                else:
                    print("accept", loss_client - self.loss_last, loss_client, self.loss_last)
                    self.loss_last = loss_client
                    break
            else:
                self.loss_last = loss_client
                print("init", loss_client)
                break

        # self.model.cpu()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

        # if self.privacy:
        #     eps, DELTA = get_dp_params(privacy_engine)
        #     print(f"Client {self.id}", f"epsilon = {eps:.2f}, sigma = {DELTA}")
        #
        #     for param, param_dp in zip(model_origin.parameters(), self.model.parameters()):
        #         param.data = param_dp.data.clone()
        #     self.model = model_origin
        #     self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

