import torch
import json
from argparse import ArgumentParser
from datetime import datetime
from tqdm import tqdm
from math import ceil

import core
from core.loader.data_loader import *
from core.models import load_empty_model
from core.metrics import RunningMetrics


def train_test_split(dataset: ATLAS2_Train_Dataset) -> tuple:
    train_size = int(0.8*len(dataset))
    test_size = len(dataset) - train_size

    train_data, test_data = torch.utils.data.random_split(dataset, [train_size, test_size])
    return train_data, test_data


def store_results(args: ArgumentParser, results: dict) -> None:
    # Creating the results folder if it does not exist
    if not os.path.exists(args.results_path):
        os.makedirs(args.results_path)

    results_folder = os.path.join(args.results_path, datetime.now().isoformat('#'))
    os.makedirs(results_folder)

    # Storing metadata
    with open(os.path.join(results_folder, f'metadata.json'), 'w') as json_buffer:
        json.dump(vars(args), json_buffer, indent=4)

    model  = results['model']
    scores = {key: value for key, value in results.items() if key != 'model'}

    # Storing model weights
    torch.save(model.state_dict(), os.path.join(results_folder, 'model' + '.pt'))

    # Storing metric scores
    with open(os.path.join(results_folder, 'scores' + '.json'), 'w') as json_buffer:
        json.dump(scores, json_buffer, indent=4)

    print(f'\nModel and scores saved in {args.results_path}')


def run(args: ArgumentParser) -> dict:
    if not os.path.isdir(args.data_path):
        raise FileNotFoundError(f'Folder {args.data_path} does not exist.')
    
    print('')
    print('Data path:    ', args.data_path)
    print('Results path: ', args.results_path)
    print('')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print('Weighted loss ENABLED' if args.weighted_loss else 'Weighted loss DISABLED')

    labeled_data = ATLAS2_Train_Dataset(data_path=args.data_path)

    if args.weighted_loss:
        class_weights = torch.tensor(labeled_data.get_class_weights(), device=device, requires_grad=False)
        class_weights = class_weights.float()
    else:
        class_weights = None

    loss_map = {
        'cel': ('CrossEntropyLoss', {'reduction': 'sum', 'weight': class_weights})
    }

    # Defining the loss function
    loss_name, loss_args = loss_map[args.loss_function]
    criterion = getattr(core.loss, loss_name)(**loss_args)

    # Splitting the data into train and test
    train_data, test_data = train_test_split(labeled_data)

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size)

    print('\nCreating model...')
    print('Architecture:   ', args.architecture.upper())
    print('Optimizer:      ', args.optimizer)
    print('Loss function:  ', args.loss_function)
    print('Learning rate:  ', args.learning_rate)
    print('Device:         ', device)
    print('Batch size:     ', args.batch_size)
    print('N. of epochs:   ', args.n_epochs)

    print('')
    print('Number of training examples:', len(train_data))
    print('Number of test examples:    ', len(test_data))
    
    model = load_empty_model(args.architecture, labeled_data.get_n_classes())
    model = model.to(device)

    # Defining the optimizer
    optimizer_map = {
        'adam': torch.optim.Adam,
        'sgd' : torch.optim.SGD
    }

    OptimizerClass = optimizer_map[args.optimizer]
    optimizer      = OptimizerClass(
        params=model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    # Initializing metrics
    train_metrics = RunningMetrics(n_classes=labeled_data.get_n_classes(), bf1_threshold=2)
    test_metrics  = RunningMetrics(n_classes=labeled_data.get_n_classes(), bf1_threshold=2)

    # Storing the loss for each epoch
    train_loss_list = []
    test_loss_list  = []

    for epoch in range(args.n_epochs):
        # Training phase
        model.train()
        train_loss = 0

        print(datetime.now().strftime('\n%Y/%m/%d %H:%M:%S'))
        print(f'Training on epoch {epoch + 1}/{args.n_epochs}\n')

        for images, labels in tqdm(train_loader, ascii=' >='):
            optimizer.zero_grad()

            images = images.type(torch.FloatTensor).to(device)
            labels = labels.type(torch.FloatTensor).to(device)

            outputs = model(images)

            # Updating the running metrics
            train_metrics.update(images=outputs, targets=labels)

            # Computing the loss and updating weights
            loss = criterion(images=outputs, targets=labels.long())
            train_loss += loss.item()
            loss.backward()

            optimizer.step()
        
        train_loss = train_loss / ceil((len(train_loader) / args.batch_size))
        train_scores = train_metrics.get_scores()

        print(f'Train loss: {train_loss}')
        print(f'Train mIoU: {train_scores["mean_iou"]}')

        train_loss_list.append(train_loss)
        train_metrics.reset()
    
    # Testing phase
    with torch.no_grad():
        model.eval()
        test_loss = 0

        print(datetime.now().strftime('\n%Y/%m/%d %H:%M:%S'))
        print('Testing the model...\n')

        for images, labels in tqdm(test_loader, ascii=' >='):
            images = images.type(torch.FloatTensor).to(device)
            labels = labels.type(torch.FloatTensor).to(device)

            outputs = model(images)

            # Updating the running metrics
            test_metrics.update(images=outputs, targets=labels)

            # Computing the loss
            loss = criterion(images=outputs, targets=labels.long())
            test_loss += loss.item()
        
        test_loss = test_loss / ceil((len(test_loader) / args.batch_size))
        test_scores = test_metrics.get_scores()

        print(f'Test loss: {test_loss}')
        print(f'Test mIoU: {test_scores["mean_iou"]}')

        test_loss_list.append(test_loss)
        test_metrics.reset()
    
    results = {
        'model'        : model,
        'train_scores' : train_scores,
        'test_scores'  : test_scores,
        'train_losses' : train_loss_list,
        'test_losses'  : test_loss_list
    }
    
    return results


if __name__ == '__main__':
    parser = ArgumentParser(description='Hyperparameters')

    parser.add_argument('-a', '--architecture',
        dest='architecture',
        type=str,
        default='unet3d',
        help='Architecture to use [segnet, unet, deconvnet]',
        choices=['segnet', 'unet', 'unet3d', 'deconvnet']
    )
    parser.add_argument('-p', '--data-path',
        dest='data_path',
        type=str,
        help='Path to the folder containing the dataset and its labels in .npy format'
    )
    parser.add_argument('-b', '--batch-size',
        dest='batch_size',
        type=int,
        default=16,
        help='Batch Size'
    )
    parser.add_argument('-d', '--device',
        dest='device',
        type=str,
        default='cuda:0',
        help='Device to train on [cuda:n]'
    )
    parser.add_argument('-v', '--cross-validation',
        dest='cross_validation',
        action='store_true',
        default=False,
        help='Whether to use 5-fold cross validation'
    )
    parser.add_argument('-L', '--loss-function',
        dest='loss_function',
        type=str,
        default='cel',
        help='Loss function to use [cel (Cross_Entropy Loss)]',
        choices=['cel']
    )
    parser.add_argument('-o', '--optimizer',
        dest='optimizer',
        type=str,
        default='adam',
        help='Optimizer to use [adam, sgd (Stochastic Gradient Descent)]',
        choices=['adam', 'sgd']
    )
    parser.add_argument('-l', '--learning-rate',
        dest='learning_rate',
        type=float,
        default=1e-4,
        help='Learning rate'
    )
    parser.add_argument('-w', '--weight-decay',
        dest='weight_decay',
        type=float,
        default=1e-5,
        help='L2 regularization. Value 0 indicates no weight decay'
    )
    parser.add_argument('-W', '--weighted-loss',
        dest='weighted_loss',
        action='store_true',
        default=True,
        help='Whether to use class weights in the loss function'
    )
    parser.add_argument('-e', '--n-epochs',
        dest='n_epochs',
        type=int,
        default=20,
        help='Number of epochs'
    )
    parser.add_argument('-s', '--store-results',
        dest='store_results',
        action='store_true',
        default=True,
        help='Whether to store the model weights and metrics'
    )
    parser.add_argument('-r', '--results-path',
        dest='results_path',
        type=str,
        default=os.path.join(os.getcwd(), 'results'),
        help='Directory for storing execution results'
    )

    args = parser.parse_args(args=None)
    results = run(args)

    if args.store_results:
        store_results(args, results)
