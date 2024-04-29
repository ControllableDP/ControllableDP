# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang
import pdb

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

import torch
import os
import numpy as np
import h5py
import copy
import time
import random
from utils.data_utils import read_client_data
from utils.dlg import DLG
from utils.dpsur_privacy import compute_eps, compute_rdp

class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.model = args.model
        self.model_n = args.model_n
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 100
        self.auto_break = args.auto_break
        self.LDP_mechanism = args.LDP_mechanism
        self.privacy_budget_org = args.privacy_budget
        self.privacy_budget = args.privacy_budget * np.ones((self.num_join_clients, 1))
        self.rdp_t = np.zeros((155, self.num_join_clients))
        self.noise_scale = 0
        self.clipthr = args.clipthr
        self.batch_size = args.batch_size
        self.delta = args.delta
        self.device = args.device
        self.CRD = args.CRD
        self.crossfade = args.crossfade
        self.kalman_filter = args.kalman_filter
        self.clipping = args.clipping

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []
        self.rs_test_loss = []
        self.rs_privacy_budget = []
        self.rs_noise_scale = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new

    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow)
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

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
            # pdb.set_trace()
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            # if client_time_cost <= self.time_threthold:
            #     tot_samples += client.train_samples
            #     self.uploaded_ids.append(client.id)
            #     self.uploaded_weights.append(client.train_samples)
            #     self.uploaded_models.append(client.model)
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_weights.append(client.train_samples)
            self.uploaded_models.append(client.model)

        # pdb.set_trace()
        for i, w in enumerate(self.uploaded_weights): # update model weights
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def noise_laplace(self, indx):
        orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64)) + [128, 256, 512]
        # trainloader = self.load_train_data()
        # q_s = self.num_join_clients / self.num_clients
        train_data = read_client_data(self.dataset, indx, is_train=True)
        delta_s1 = 2 * self.clipthr / len(train_data) #  (self.batch_num_per_client * self.batch_size) # self.num_items_train
        noise_scale = 2 * delta_s1 * self.join_ratio * self.global_rounds / self.privacy_budget[indx]
        # self.rdp_t[:,indx] += compute_rdp(self.batch_size / len(train_data), noise_scale, 1, orders)
        # epsilon, best_alpha = compute_eps(orders, self.rdp_t[:, indx], self.delta)
        # self.privacy_budget[indx] -= epsilon

        self.privacy_budget[indx] -= 2 * self.join_ratio * delta_s1 / noise_scale
        return noise_scale, self.privacy_budget[indx]

    def noise_gaussian(self, indx):
        orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(11, 64)) + [128, 256, 512]
        # trainloader = self.load_train_data()
        # q_s = self.num_join_clients / self.num_clients
        train_data = read_client_data(self.dataset, indx, is_train=True)
        delta_s = 2 * self.clipthr / len(train_data) # (self.batch_num_per_client * self.batch_size) # 2*args.clipthr/args.num_items_train
        noise_scale = delta_s * np.sqrt(2 * self.join_ratio * self.global_rounds * np.log(1 / self.delta)) / self.privacy_budget[indx]
        # pdb.set_trace()
        # self.rdp_t[:,indx] = compute_rdp(self.batch_size / len(train_data), noise_scale / delta_s, 1, orders).astype('float64')
        # epsilon, best_alpha = compute_eps(orders, self.rdp_t[:,indx].reshape((155, 1)), self.delta)
        # pdb.set_trace()
        # self.privacy_budget[indx] -= epsilon

        self.privacy_budget[indx] -= 1 * self.join_ratio * (delta_s ** 2) / (noise_scale** 2) # self.join_ratio is q_s

        return noise_scale, self.privacy_budget[indx]

    def aggregate_parameters_dp(self):
        assert (len(self.uploaded_models) > 0)

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()

        indx = 0
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            # pdb.set_trace()
            self.add_parameters_dp(w, client_model, indx)
            indx += 1

        self.rs_privacy_budget.append(self.privacy_budget[0])
        self.rs_noise_scale.append(self.noise_scale)

    def add_parameters_dp(self, w, client_model, indx):
        # pdb.set_trace()
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            if self.LDP_mechanism == 'gaussian':
                self.noise_scale, self.privacy_budget[indx] = self.noise_gaussian(indx)
                # print(self.privacy_budget[indx])
                # pdb.set_trace()
                # client_param_noised = client_param.data.clone() + torch.from_numpy(np.random.normal(0, self.noise_scale[0], client_param.data.clone().size())).to(self.device)
                client_param_noised = client_param.data.clone() + (
                        self.noise_scale[0] * torch.randn_like(client_param.data))

            elif self.LDP_mechanism == 'laplace':
                self.noise_scale, self.privacy_budget[indx] = self.noise_laplace(indx)
                client_param_noised = client_param.data.clone() + torch.from_numpy(np.random.laplace(0, self.noise_scale[0], client_param.data.clone().size())).to(self.device)  # np.random.normal(0, noise_scale, w[k][i].size())
            server_param.data += client_param_noised * w


            # print('noise_scale', noise_scale)
        # return self.privacy_budget

    # if args.LDP_mechanism == 'gaussian':
    #     noise = np.random.normal(0, noise_scale, w[k][i].size())
    # elif args.LDP_mechanism == 'laplace':
    #     noise = np.random.laplace(0, noise_scale, w[k][i].size())  # np.random.normal(0, noise_scale, w[k][i].size())
    #
    # if args.gpu != -1:
    #     noise = torch.from_numpy(noise).float().cuda()
    # else:
    #     noise = torch.from_numpy(noise).float()
    #
    # w_noise[k][i] = w_noise[k][i] + noise

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def model_exists(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        return os.path.exists(model_path)
        
    def save_results(self):
        algo = (self.dataset + "_" + self.algorithm+ "_"  + self.model_n + "_CRD_" + self.CRD + "_CFR_" + self.crossfade +
                "_KF_" + self.kalman_filter + "_" + self.LDP_mechanism + "_Dclip_" + self.clipping + "_pb_" + str(self.privacy_budget_org))
        result_path = "../results/"
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = result_path + "{}.h5".format(algo)
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                hf.create_dataset('rs_test_loss', data=self.rs_test_loss)
                hf.create_dataset('rs_privacy_budget', data=self.rs_privacy_budget)
                hf.create_dataset('rs_noise_scale', data=self.rs_noise_scale)

    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_loss = []
        tot_correct = []
        tot_auc = []
        for c in self.clients:
            closs, ct, ns, auc = c.test_metrics()
            tot_loss.append(closs)
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc, tot_loss

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None, tloss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()
        # pdb.set_trace()

        test_loss = sum(stats[4]) * 1.0 / sum(stats[1])
        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        if tloss == None:
            self.rs_test_loss.append(test_loss)
        else:
            tloss.append(test_loss)
        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Loss: {:.4f}".format(test_loss))
        print("Averaged Test Accurancy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))
        return self.rs_train_loss, self.rs_test_loss, self.rs_test_acc

    def print_(self, test_acc, test_auc, train_loss, test_loss):
        print("Average Test Accurancy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))
        print("Average Test Loss: {:.4f}".format(test_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt != None and div_value != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt != None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value != None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
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

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_loss = []
        tot_correct = []
        tot_auc = []
        for c in self.new_clients:
            closs, ct, ns, auc = c.test_metrics()
            tot_loss.append(closs)
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc, tot_loss
