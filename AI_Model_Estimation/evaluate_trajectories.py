
from __future__ import absolute_import, division, print_function
from torch.utils.data import SequentialSampler
from IMU_Pose import IPPU, MinMaxNormalizer
import numpy as np
import torch
from RosbagDatasets import ROSbagIMUGT
import matplotlib.pyplot as plt
import pickle

IMU_SEQUENCE_LENGTH = 10


def translation_from_parameters(params):
    tvec = torch.zeros((2, 1), device=params.device)
    tvec[[0]] = torch.cos(params[0, 1])*params[0, 0]
    tvec[[1]] = torch.sin(params[0, 1])*params[0, 0]
    return tvec


def evaluate_trajectories(model, dataloader, device):
    print("-> Computing pose predictions")
    with torch.no_grad():
        pred_tvec = []
        gt_tvec = []
        running_loss = 0
        for i, data in enumerate(dataloader, 0):
            imu_data = data[1]
            imu_data = imu_data.to(device, torch.float64)
            gt_delta = data[0]
            gt_delta = gt_delta.to(device)

            outputs = model(imu_data)

            # check if we need to undo a normalization
            if dataloader.dataset.normalizer:
                outputs = dataloader.dataset.normalizer.inverse_transform(outputs, dataloader.dataset.normalizer.transform_dict['dist_angle'])
                gt_delta = dataloader.dataset.normalizer.inverse_transform(gt_delta, dataloader.dataset.normalizer.transform_dict['dist_angle'])

            pred = translation_from_parameters(outputs)
            gt = translation_from_parameters(gt_delta)
            pred_tvec.append(torch.squeeze(pred).cpu().numpy())
            gt_tvec.append(torch.squeeze(gt).cpu().numpy())

    trajectory = np.cumsum(pred_tvec, axis=0)
    gt_trajectory = np.cumsum(gt_tvec, axis=0)

    # compute MSE along trajectory and plot
    traj_error = np.sqrt(np.sum(np.square(trajectory - gt_trajectory), axis=1))
    plt.figure()
    plt.plot(traj_error)
    plt.ylabel('Error [mm]')
    plt.xlabel('Time')
    plt.show()

    # now print data
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    xs = [x[0] for x in trajectory]
    ys = [x[1] for x in trajectory]
    gt_xs = [x[0] for x in gt_trajectory]
    gt_ys = [x[1] for x in gt_trajectory]
    ax.plot(xs, ys, label='Predicted')
    ax.plot(gt_xs, gt_ys, label='Ground truth')
    ax.set_xlabel('x [mm]')
    ax.set_ylabel('y [mm]')
    ax.set_zlabel('z [mm]')
    plt.legend()
    plt.show()
    # max(this_pose)


if __name__ == '__main__':
    # define model file and bag file with test trajectory
    model_file = '/home/hazem/Desktop/Training/Output/Trial_2/IPPU_02Feb2022_001113.ptm'
    test_file = '/home/hazem/Desktop/Training/Bags/testing_bags/ROS_BAG_11.bag'
    normalizer_file = '/home/hazem/Desktop/Training/Output/Trial_2/MinMaxNormalizer_01Feb2022_215324.pkl'

    # initialize model
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = IPPU()
    model.to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()

    # initialize test trajectory sequentially
    testset = ROSbagIMUGT(test_file, "/imu", "/gazebo/model_states", imu_seq_length=IMU_SEQUENCE_LENGTH)

    if normalizer_file != None:
        normalizer = pickle.load(open(normalizer_file, 'rb'))
        testset.normalizer = normalizer

    testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                             shuffle=False, drop_last=False, num_workers=0)

    # evaluate and plot trajectory compared to ground truth
    evaluate_trajectories(model, testloader, device)
