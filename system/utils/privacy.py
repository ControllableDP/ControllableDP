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

from opacus import PrivacyEngine

MAX_GRAD_NORM = 1.0
DELTA = 1e-5

def initialize_dp(model, optimizer, data_loader, dp_sigma):
    privacy_engine = PrivacyEngine()
    model, optimizer, data_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier = dp_sigma, 
        max_grad_norm = MAX_GRAD_NORM,
    )

    return model, optimizer, data_loader, privacy_engine


def get_dp_params(privacy_engine):
    return privacy_engine.get_epsilon(delta=DELTA), DELTA


# def Privacy_account_laplace(args, threshold_epochs):
#     q_s = args.num_Chosenusers/args.num_users
#     delta_s1 = 2 * args.clipthr /args.num_items_train
#     noise_scale = 2 * delta_s1 * q_s * threshold_epochs / args.privacy_budget
#     return noise_scale
#
# def Privacy_account(args, threshold_epochs):
#     q_s = args.num_Chosenusers/args.num_users
#     delta_s = 2 * args.clipthr /args.num_items_train # 2*args.clipthr/args.num_items_train
#
#     # if args.dp_mechanism != 'CRD':
#     if args.num_niose_prop > 1:
#         noise_scale = delta_s * np.sqrt(
#             2 * q_s * (args.num_niose_prop - pow(args.num_niose_prop, 1 - threshold_epochs)) * np.log(
#                 1 / args.delta)) \
#                       / (args.privacy_budget * np.sqrt(args.num_niose_prop - 1))
#     elif args.num_niose_prop == 1.0:
#         noise_scale = delta_s * np.sqrt(
#             2 * q_s * threshold_epochs * np.log(1 / args.delta)) / args.privacy_budget
#     else:
#         noise_scale = delta_s * np.sqrt(
#             2 * q_s * (- args.num_niose_prop + pow(args.num_niose_prop, 1 - threshold_epochs)) * np.log(
#                 1 / args.delta)) \
#                       / (args.privacy_budget * np.sqrt(- args.num_niose_prop + 1))
#
#     return noise_scale