import warnings; warnings.simplefilter('ignore')
import os
import copy
import time
import math
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import sklearn.metrics as metrics
from sklearn.preprocessing import MinMaxScaler


class LSTM(torch.nn.Module):
  def __init__(self, input_dim, hidden_dim, batch_size, output_dim, num_layers, dropout, **_):
    super(LSTM, self).__init__()
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.batch_size = batch_size
    self.output_dim = output_dim
    self.num_layers = num_layers
    self.input_linear = nn.Linear(input_dim, hidden_dim) # TODO nn.LSTM or Linear
    #self.middle_linear = nn.Linear(hidden_dim, hidden_dim)
    #self.linear = nn.Linear(hidden_dim, output_dim)
    #self.dropout = nn.Dropout(dropout)

    # Define the LSTM layer
    self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

    # Define the output layer
    self.linear = nn.Linear(self.hidden_dim, output_dim)

  def init_hidden(self):
    # This is what we'll initialize the hidden state as
    return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
            torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)) # Why two times...

    #self.apply(self.init_weights)

  def init_weights(self, model):
    if type(model) == nn.Linear:
      nn.init.uniform_(model.weight, 0, 0.001)

  def apply_dropout(self):
    def apply_drops(m):
      if type(m) == nn.Dropout:
        m.train()
    self.apply(apply_drops)

  def forward(self, input):
    # Forward pass through LSTM layer
    # shape of lstm_out: [input_size, batch_size, hidden_dim]
    # shape of self.hidden: (a, b), where a and b both
    # have shape (num_layers, batch_size, hidden_dim).

    lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))
    #lstm_out, self.hidden = self.lstm(1, self.batch_size, -1)
    
    # Only take the output from the final timestep
    # Can pass on the entirety of the lstm_out to the next layer if it is a seq2seq prediction
    y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
    return y_pred.view(-1)

## Class to computes and stores the average and current value
class AverageMeter(object):
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def update(self, val, n=1):
    self.val = val
    self.sum += val*n
    self.count += n
    self.avg = self.sum/self.count

## Dataset
class Dataset(object):
  def __init__(self, X, y):
    assert len(X) == len(y)
    self.X = X
    self.y = y

  def __len__(self):
    return len(self.X)

  def __getitem__(self, index):
    x = torch.Tensor(self.X[index])
    y = torch.Tensor(self.y[index])
    return x, y

## Getter for dataloaders for all datasets
def get_datasets(data, batch_size, shuffle, num_workers=0, y_scaler=None, X_scaler=None, **_):
  # Prepare for predictions with single loader
  if X_scaler and y_scaler:
    X_test = X_scaler.transform(data['X_test'])
    y_test = y_scaler.transform(data['y_test'])
    pred_generator = DataLoader(Dataset(X_test, y_test), shuffle=False, batch_size=batch_size, num_workers=num_workers)
    return None, None, None, pred_generator, None, y_scaler

  # Prepare for training with all loaders
  X_scaler = MinMaxScaler(feature_range=(0, 10))
  y_scaler = MinMaxScaler(feature_range=(0, 10))
  
  X_train = X_scaler.fit_transform(data['X_train'])
  y_train = y_scaler.fit_transform(data['y_train'])
  
  X_val = X_scaler.transform(data['X_val'])
  y_val = y_scaler.transform(data['y_val'])
  
  X_test = X_scaler.transform(data['X_test'])
  y_test = y_scaler.transform(data['y_test'])

  training_generator = DataLoader(Dataset(X_train, y_train), shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
  validation_generator = DataLoader(Dataset(X_val, y_val), shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
  test_generator = DataLoader(Dataset(X_test, y_test), shuffle=shuffle, batch_size=batch_size, num_workers=num_workers)
  pred_generator = DataLoader(Dataset(X_test, y_test), shuffle=False, batch_size=batch_size, num_workers=num_workers)

  return training_generator, validation_generator, test_generator, pred_generator, X_scaler, y_scaler


filename = './MLP_model.joblib'

## Train model
def train(config, data):
  if (os.path.exists(filename)):
    return
  params = { 
    'shuffle': False, # TODO Set to true when LSTM
    'num_workers': 4,
    'input_dim':len(data['X_train'].columns),
    'output_dim': len(data['y_train'].columns),
    'num_layers': 4,
    'batch_size': 1,
    'hidden_dim': 69,
    'learning_rate': 1e-3,
    'epochs': 1000,
    'dropout': 0.2,
    'log_nth': 1,
    'mode': 'train',
  }

  # Activate gpu optimization
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

  torch.manual_seed(42)

  model = LSTM(**params).to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
  criterion = nn.MSELoss(reduction='sum')

  model_dict, val_score = fit(data, model, device, params, config, optimizer, criterion)
  torch.save(model_dict, filename)


## Prediction
def predict(config, data):

  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
  
  state = torch.load(filename)
  params = state['params']
  params['mode'] = 'predict'
  params['X_scaler'] = state['X_scaler']
  params['y_scaler'] = state['y_scaler']
  
  model = LSTM(**params).to(device)
  model.load_state_dict(state['model_dict'])

  pred_np, score = fit(data, model, device, params, config)
  pred_df = pd.DataFrame(index=data['X_test'].index)
  print(pred_df.shape)
  pred_df['MLP'] = pred_np[::config['window'], :].flatten()
  pred_df['True'] = data['y_test'].iloc[::config['window'], :].values.flatten()
  rmse = math.sqrt(metrics.mean_squared_error(pred_df['True'], pred_df['MLP']))
  r2 = metrics.r2_score(pred_df['True'], pred_df['MLP'])
  return pred_df, rmse, r2

## Pytorch Pipe 🐥
def fit(data, model, device, params, config, optimizer=None, criterion=None):
  training_generator, validation_generator, test_generator, pred_generator, X_scaler, y_scaler = get_datasets(data, **params)

  ## Run single training batch with backprop {loss}
  def runBatches(generator, isTrainMode):
    losses = AverageMeter()

    for i, (X, y) in enumerate(generator):
      X, y = Variable(X, requires_grad=True).to(device), Variable(y).to(device)
      output = model.forward(X)
      y_pred = output
      loss = criterion(output, y)
      if isTrainMode:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
      losses.update(loss.item())

    return losses.avg, y_pred, loss

  ## Run single prediction batch {y_true, y_pred}
  def predict(generator):
    model.eval()
    y_trues = []
    y_preds = []
    
    for i, (X, y) in enumerate(generator):
      X, y = X.to(device), y.to(device)
      output = model.forward(X)
      y_trues = np.append(y_trues, y.cpu().numpy())
      y_preds = np.append(y_preds, output.detach().cpu().numpy())
    print(len(y_preds))
    return np.array(y_trues), np.array(y_preds)

  ## Do Training
  if params['mode'] == 'train':
    print("Sjukeste yoloskriptet")

    start_time = datetime.datetime.now()
    train_scores = []
    val_scores = []

    best_model_dict = copy.deepcopy(model.state_dict())
    best_score = 999
    for epoch in range(params['epochs']):

      # Initialize hidden state
      # Don't do this if you want your LSTM ti be stateful
      model.hidden = model.init_hidden()

      # Training
      model.train()
      train_score, y_pred, loss = runBatches(generator=training_generator, isTrainMode=True)
      train_scores.append(train_score)

      # Validation
      model.eval()
      val_score, y_pred, loss = runBatches(generator=validation_generator, isTrainMode=False)
      val_scores.append(val_score)

      # Keep the best model
      if val_score < best_score:
        best_score = val_score
        best_model_dict = copy.deepcopy(model.state_dict())

      time = (datetime.datetime.now() - start_time).total_seconds()
      
      if not epoch%params['log_nth']:
        print('e {e:<3} time: {t:<4.0f} train: {ts:<4.2f} val: {vs:<4.2f}'.format(e=epoch, t=time, ts=train_score, vs=val_score))

    # Test the trained model
    test_score, y_pred, loss = runBatches(generator=test_generator, isTrainMode=False)
    trues, preds = predict(generator=pred_generator)

    # Return results, model and params for saving
    result_dict = {
      'model_dict': best_model_dict,
      'params': params,
      'train_scores': train_scores,
      'val_scores': val_scores,
      'X_scaler': X_scaler,
      'y_scaler': y_scaler,
    }
    return result_dict, best_score

  ## Do Predictions
  if params['mode'] == 'predict':
    trues, preds = predict(generator=pred_generator)
    score = math.sqrt(metrics.mean_squared_error(trues, preds))
    return y_scaler.inverse_transform(preds.reshape(-1, config['window'])), score

