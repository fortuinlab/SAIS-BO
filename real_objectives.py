from typing import Optional

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import torch
from botorch.test_functions.base import BaseTestProblem
from botorch.utils.transforms import unnormalize

import xgboost as xgb

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import cross_val_score, StratifiedKFold

dtype = torch.double
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Pest Control
    # Original code from Yuchen Lily Li https://github.com/yucenli/bnn-bo
    # and QUVA Deep Vision Lab https://github.com/QUVA-Lab/COMBO
    # Modifications made by Qian Xie, 2024 

    # Further modifications made by the authors of
    # Standard Acquisition is Sufficient for Asynchronous Bayesian Optimization

PESTCONTROL_N_CHOICE = 5
PESTCONTROL_N_STAGES = 25


def _pest_spread(curr_pest_frac, spread_rate, control_rate, apply_control):
    if apply_control:
        next_pest_frac = (1.0 - control_rate) * curr_pest_frac
    else:
        next_pest_frac = spread_rate * (1 - curr_pest_frac) + curr_pest_frac
    return next_pest_frac


def _pest_control_score(x, seed=None):
    U = 0.1
    n_stages = x.size
    n_simulations = 100

    init_pest_frac_alpha = 1.0
    init_pest_frac_beta = 30.0
    spread_alpha = 1.0
    spread_beta = 17.0 / 3.0

    control_alpha = 1.0
    control_price_max_discount = {1: 0.2, 2: 0.3, 3: 0.3, 4: 0.0}
    tolerance_develop_rate = {1: 1.0 / 7.0, 2: 2.5 / 7.0, 3: 2.0 / 7.0, 4: 0.5 / 7.0}
    control_price = {1: 1.0, 2: 0.8, 3: 0.7, 4: 0.5}
    # below two changes over stages according to x
    control_beta = {1: 2.0 / 7.0, 2: 3.0 / 7.0, 3: 3.0 / 7.0, 4: 5.0 / 7.0}

    payed_price_sum = 0
    above_threshold = 0

    if seed is not None:
        init_pest_frac = np.random.RandomState(seed).beta(init_pest_frac_alpha, init_pest_frac_beta, size=(n_simulations,))
    else:
        init_pest_frac = np.random.beta(init_pest_frac_alpha, init_pest_frac_beta, size=(n_simulations,))
    curr_pest_frac = init_pest_frac
    for i in range(n_stages):
        if seed is not None:
            spread_rate = np.random.RandomState(seed).beta(spread_alpha, spread_beta, size=(n_simulations,))
        else:
            spread_rate = np.random.beta(spread_alpha, spread_beta, size=(n_simulations,))
        do_control = x[i] > 0
        if do_control:
            if seed is not None:
                control_rate = np.random.RandomState(seed).beta(control_alpha, control_beta[x[i]], size=(n_simulations,))            
            else:
                control_rate = np.random.beta(control_alpha, control_beta[x[i]], size=(n_simulations,))
            next_pest_frac = _pest_spread(curr_pest_frac, spread_rate, control_rate, True)
            # torelance has been developed for pesticide type 1
            control_beta[x[i]] += tolerance_develop_rate[x[i]] / float(n_stages)
            # you will get discount
            payed_price = control_price[x[i]] * (
                    1.0 - control_price_max_discount[x[i]] / float(n_stages) * float(np.sum(x == x[i])))
        else:
            next_pest_frac = _pest_spread(curr_pest_frac, spread_rate, 0, False)
            payed_price = 0
        payed_price_sum += payed_price
        above_threshold += np.mean(curr_pest_frac > U)
        curr_pest_frac = next_pest_frac

    return payed_price_sum + above_threshold

def pest_control_price(x):
    control_price_max_discount = torch.tensor([0.0, 0.2, 0.3, 0.3, 0.0])
    control_price = torch.tensor([0.0, 1.0, 0.8, 0.7, 0.5])
    
    # Convert squeezed tensor to integer tensor
    x_int = x.int()
    x_int = torch.atleast_2d(x_int)
    
    # Calculate the number of times each element appears in x
    x_counts = torch.eq(x_int.unsqueeze(1), x_int).sum(dim=0)
    
    # Calculate payed prices using vectorized operations
    payed_prices = control_price[x_int] * (
        1.0 - control_price_max_discount[x_int] / float(x_int.size(0)) * x_counts
    )
    
    # Sum up the payed prices
    payed_price_sum = payed_prices.sum()
    
    return torch.clamp_min(payed_price_sum, 5.)


class PestControl(BaseTestProblem):
    """
	Pest Control Problem.

	"""

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False
    ) -> None:
        self.dim = PESTCONTROL_N_STAGES
        self._bounds = np.repeat([[0], [5 - 1e-6]], PESTCONTROL_N_STAGES, axis=1).T
        self.num_objectives = 1

        super().__init__(
            noise_std=noise_std,
            negate=negate,
        )

        self.categorical_dims = np.arange(self.dim)

    def evaluate_true(self, X):
        res = torch.stack([self._compute(x) for x in X]).to(X)
        res += 1e-6 * torch.randn_like(res)
        return res

    def _compute(self, x):
        evaluation = _pest_control_score((x.cpu() if x.is_cuda else x).numpy(), seed=1)
        return torch.tensor(evaluation)
    

def pest_builder(d=25):

    """Build pest control problem"""

    if d != 25:
        raise ValueError("this is a 25D problem")
    
    
    def f(x):
        """pest"""
        x = torch.atleast_2d(x)
        choice_X = torch.floor(5*x)
        choice_X[choice_X == 5] = 4
        return PestControl(negate=True)(choice_X)
    
    return f


def xgb_builder(d = 9):

    if d != 9:
        raise ValueError("this is a 9D problem")
    
    bounds = torch.tensor([[0.001, 0.01, 0.01, 0.01, 0, 0, 0, 1, 1],
                           [0.99, 0.99, 0.99, 0.99, 100, 100, 100, 5e4, 1e2]],
                            dtype = dtype, device = device)

    X, y = load_breast_cancer(return_X_y=True)

    def f(x):
            
            x = unnormalize(x, bounds)

            params = {
            'objective': 'binary:logistic',
            'learning_rate': x[0].item(),  
            'subsample': x[1].item(),                 
            'colsample_bytree': x[2].item(),     
            'colsample_bynode': x[3].item(),
            'gamma': x[4].item(),                         
            'reg_alpha': x[5].item(),                    
            'reg_lambda': x[6].item(),                 
            'n_estimators': int(x[7].item()),
            "max_depth": int(x[8].item()),
            'eval_metric': 'logloss'
            }

            model = xgb.XGBClassifier(**params)
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

            
            return scores.mean()
    
    return f


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010))
])

dataset = torchvision.datasets.CIFAR10(root='./CIFAR_data', train=True,
                                       download=True, transform=transform)

class CNN(nn.Module):
    def __init__(self, filter_sizes):
        super(CNN, self).__init__()

        self.conv_layers = nn.ModuleList()
        in_channels = 3
        for filters in filter_sizes:
            self.conv_layers.append(
                nn.Conv2d(in_channels, filters, kernel_size=3, padding=1)
            )
            in_channels = filters

        self.pool = nn.MaxPool2d(2, 2)

        self.fc = nn.Linear(filter_sizes[-1] * 4 * 4, 10)

    def forward(self, x):
        for idx, conv in enumerate(self.conv_layers):
            x = F.relu(conv(x))
            if idx % 2 == 1:  
                x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def cifar_builder(d = 9):

    if d != 9:
        raise ValueError("this is a 9D problem")
    
    bounds = torch.tensor([[0.001, 0.001, 1, 16, 16, 16, 16, 16, 16],
                           [0.999, 0.999, 5000, 256, 256, 256, 256, 256, 256]],
                            dtype = dtype, device = device)
    
    train_size = 25000  # half of training set
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])
    val_loader = DataLoader(val_set, batch_size=256, shuffle=False, num_workers=0)

    def train_and_evaluate(x):

        print("normalized", x)
        x = unnormalize(x, bounds)
        print("un-normalized", x)
       

        lr = x[0]
        momentum = x[1]
        batch_size = int(x[2])
        filter_sizes = [int(x[i]) for i in range(3, 9)]

        print(filter_sizes)

        epochs = 20

        model = CNN(filter_sizes).to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

        train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=0)

        print("starting training")

        model.train()
        for epoch in range(epochs):
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        print("starting validation")

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        acc = correct / total
        return acc

    return train_and_evaluate


realOBJECTIVES = {
    "pest": pest_builder,
    "xgb": xgb_builder,
    "cifar": cifar_builder,
}

