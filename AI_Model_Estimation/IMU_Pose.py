import torch
torch.set_printoptions(precision=20)
torch.set_default_dtype(torch.float64)
from torch import nn
from os import path
from RosbagDatasets import ROSbagIMUGT
from torch.utils.data import ConcatDataset
import time
from matplotlib import pyplot as plt
import numpy as np
import pickle

OUTPUT_DIR = '/home/hazem/Desktop/Training/Output/'
TRAINING_BAGS_DIR = '/home/hazem/Desktop/Training/Bags/training_bags/'
TESTING_BAGS_DIR = '/home/hazem/Desktop/Training/Bags/testing_bags/'
NUM_EPOCHS = 100
MINIBATCH_SIZE = 32
IMU_SEQUENCE_LENGTH = 10
HIDDEN_SIZE = 128
NORMALIZE_DATA = True


class MinMaxNormalizer:
    def __init__(self, feature_range=(-1, 1)):
        self.transform_dict = None
        self.feature_range = feature_range

    def fit_transform(self, datasets):
        # loop through datasets and determine global min and max value per feature
        rawdata = np.concatenate([dset.imudata for dset in datasets.datasets])
        imu_dmax = rawdata.max(axis=0)
        imu_dmin = rawdata.min(axis=0)

        rawdata = np.concatenate([dset.d_gt_dist_angle_items for dset in datasets.datasets])
        distangle_dmax = rawdata.max(axis=0)
        distangle_dmin = rawdata.min(axis=0)

        rawdata = np.concatenate([dset.delta_yaw_items for dset in datasets.datasets])
        deltayaw_dmax = rawdata.max()
        deltayaw_dmin = rawdata.min()

        self.transform_dict = {'imudata': [imu_dmin, imu_dmax],
                               'dist_angle': [distangle_dmin, distangle_dmax],
                               'delta_yaw': [deltayaw_dmin, deltayaw_dmax]}

    def transform(self, data):
        data['imudata'] = self.__apply_transform__(data['imudata'], self.transform_dict['imudata'])
        data['d_gt_dist_angle'] = self.__apply_transform__(data['d_gt_dist_angle'], self.transform_dict['dist_angle'])
        data['delta_yaw'] = self.__apply_transform__(data['delta_yaw'], self.transform_dict['delta_yaw'])

        return data

    # undoes transformation (assumes torch tensor input)
    def inverse_transform(self, datascaled, dminmax):
        data = torch.tensor((dminmax[1] - dminmax[0]), device=datascaled.device) * (datascaled - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0]) \
               + torch.tensor(dminmax[0], device=datascaled.device)
        return data

    def __apply_transform__(self, data, dminmax):
        datastd = (data - dminmax[0]) / (dminmax[1] - dminmax[0])
        datascaled = datastd * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]

        return datascaled


def load_normalizer(dataset, normalizer):
    try:
        for dset in dataset.datasets:
            dset.normalizer = normalizer
    except:
        dataset.normalizer = normalizer

    return dataset


class IPPU(nn.Module):
    def __init__(self):
        super(IPPU, self).__init__()

        self.LSTM = nn.LSTM(input_size=6, hidden_size=HIDDEN_SIZE, num_layers=2, batch_first=True,
                            bidirectional=True, dropout=0.50)
        # TODO: infer input channel size from input data
        self.fc1 = nn.Linear(HIDDEN_SIZE, 2)

    def forward(self, imudata):
        _, (h_out, _) = self.LSTM(imudata)
        x = self.fc1(h_out[-1, :])
        return x


def compute_loss(outputs, gt_data, criterion):
    # compute loss according to criterion
    loss = criterion(outputs, gt_data)

    return loss


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()

    print('Training IMU')

    running_loss = 0.0

    for i, data in enumerate(dataloader, 0):
        imu_data = data[1]
        imu_data = imu_data.to(device, torch.float64)
        gt_delta = data[0]
        gt_delta = gt_delta.to(device)

        outputs = model(imu_data)

        loss = compute_loss(outputs, gt_delta, criterion)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        # if (i % 100 == 0) & (i > 0):
        #     print('-----------------------------------------')
        #     print("IPPU: Loss at iteration %d: %.5f" % (i, loss))

    print('Epoch loss average: {:.4f}'.format(running_loss/(i+1)))
    print('-' * 10)

    return running_loss/(i+1)


def test_model(model, dataloader, criterion, device):
    model.eval()

    print('Testing model')

    running_loss = 0.0

    for i, data in enumerate(dataloader, 0):
        imu_data = data[1]
        imu_data = imu_data.to(device, torch.float64)
        gt_delta = data[0]
        gt_delta = gt_delta.to(device)

        outputs = model(imu_data)

        loss = compute_loss(outputs, gt_delta, criterion)

        running_loss += loss.item()

        # if (i % 100 == 0) & (i > 0):
        #     print('-----------------------------------------')
        #     print("IPPU: Loss at iteration %d: %.5f" % (i, loss))

    print('Test loss average: {:.4f}'.format(running_loss/(i+1)))
    print('-' * 10)

    return running_loss/(i+1)


if __name__ == '__main__':
    # check if we're running on GPU or CPU
    device = torch.device("cuda")
    #device = torch.device("cpu")

    # load global variables to device
    model = IPPU()
    model.to(device)

    # define train dataset, data_transforms, dataset and dataloader
    trainsets = [TRAINING_BAGS_DIR + 'ROS_BAG_1.bag',
                 TRAINING_BAGS_DIR + 'ROS_BAG_2.bag',
                 TRAINING_BAGS_DIR + 'ROS_BAG_3.bag',
                 TRAINING_BAGS_DIR + 'ROS_BAG_4.bag',
                 TRAINING_BAGS_DIR + 'ROS_BAG_5.bag',
                 TRAINING_BAGS_DIR + 'ROS_BAG_6.bag',
                 TRAINING_BAGS_DIR + 'ROS_BAG_7.bag',
                 TRAINING_BAGS_DIR + 'ROS_BAG_8.bag',
                 TRAINING_BAGS_DIR + 'ROS_BAG_9.bag',
                 TRAINING_BAGS_DIR + 'ROS_BAG_10.bag']

    trainset = ConcatDataset([ROSbagIMUGT(dataset, "/imu", "/gazebo/model_states", imu_seq_length=IMU_SEQUENCE_LENGTH)
                              for dataset in trainsets])

    # define test dataset
    testset = ROSbagIMUGT(TESTING_BAGS_DIR + 'ROS_BAG_11.bag', "/imu", "/gazebo/model_states",
                          imu_seq_length=IMU_SEQUENCE_LENGTH)

    # normalize training and testing datasets if desired
    if NORMALIZE_DATA:
        normalizer = MinMaxNormalizer()
        normalizer.fit_transform(trainset)
        trainset = load_normalizer(trainset, normalizer)
        testset = load_normalizer(testset, normalizer)
        file = open(path.join(OUTPUT_DIR, 'MinMaxNormalizer_' + time.strftime('%d%b%Y_%H%M%S') + '.pkl'), 'wb')
        pickle.dump(normalizer, file)
        file.close()

    # create the dataloaders for training and testing data
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=MINIBATCH_SIZE,
                                              shuffle=True, drop_last=True, num_workers=0)

    testloader = torch.utils.data.DataLoader(testset, batch_size=MINIBATCH_SIZE,
                                             shuffle=True, drop_last=True, num_workers=0)

    for param in model.parameters():
        param.requires_grad = True
    params = [p for p in model.parameters() if p.requires_grad]

    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(params, lr=0.0001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)

    train_losses = []
    test_losses = []
    for ii in range(NUM_EPOCHS):
        print('Epoch {}/{}'.format(ii, NUM_EPOCHS - 1))
        print('-' * 20)
        start = time.time()
        train_loss = train_one_epoch(model, trainloader, criterion, optimizer, device)
        train_losses.append(train_loss)
        test_loss = test_model(model, testloader, criterion, device)
        test_losses.append(test_loss)
        lr_scheduler.step()
        stop = time.time()
        print('Time per epoch: %.5f' % (stop - start))

    # save model
    model_name = path.join(OUTPUT_DIR, 'IPPU_' + time.strftime('%d%b%Y_%H%M%S') + '.ptm')
    torch.save(model.state_dict(), model_name)
    print('Model saved as: ' + model_name)

    # plot train - test losses
    plt.figure()
    plt.plot(train_losses, label='Train')
    plt.plot(test_losses, label='Test')
    plt.legend()
    plt.xlabel('Epochs')
    plt.title('Average losses')
    plt.savefig(model_name.split('.ptm')[0]+'_losses.png')
    plt.show()
