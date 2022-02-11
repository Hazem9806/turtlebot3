import rosbag
import numpy as np
import torch
from torch.utils.data import Dataset
import pyquaternion as Q


class ROSbagIMUGT(Dataset):

    def __init__(self, bagname, imu_topic, gt_topic, transform=None, imu_seq_length=10, imu_seq_overlap=0.0):
        # store variables
        self.imu_topic = None
        self.gt_topic = None
        self.cam0_bag = None
        self.cam0_topic = None

        self.bagname = bagname
        self.imu_topic = imu_topic
        self.gt_topic = gt_topic

        self.transform = transform
        self.imu_seq_length = imu_seq_length
        self.imu_seq_overlap = imu_seq_overlap
        self.imu_increment = self.imu_seq_length - round(self.imu_seq_length * self.imu_seq_overlap)

        # this will hold the scaling factors for min-max scaling.
        self.normalizer = None

        if bagname is not None:
            self.load_bags()
            self.precompute_items()

    def __len__(self):
        # assuming that length corresponds to actual length and not to max index
        lov = round(self.imu_seq_length * self.imu_seq_overlap)
        return int((self.imudata.shape[0] - lov) / self.imu_increment)

    def __getitem__(self, index):
        # only need to index into the precomputed arrays now
        d_gt_dist_angle = self.d_gt_dist_angle_items[index, :]
        imudata = self.imudata_items[index, :]
        delta_yaw = self.delta_yaw_items[index]
        data = {'imudata': imudata, 'd_gt_dist_angle': d_gt_dist_angle, 'delta_yaw': delta_yaw}
        # apply transformation if parameters have been set
        if self.normalizer != None:
            data = {'imudata': imudata, 'd_gt_dist_angle': d_gt_dist_angle, 'delta_yaw': delta_yaw}
            data = self.normalizer.transform(data)
        else:
            # multiply distances by 1000 so that they are in mm
            data['d_gt_dist_angle'][0] *= 1000.

        return data['d_gt_dist_angle'], data['imudata'], data['delta_yaw']

    def precompute_items(self):
        # precompute the arrays with items
        # return gt difference of xyz positions and attitude as one vector
        self.imudata_items = np.stack([self.imudata[index*self.imu_increment:(index*self.imu_increment+self.imu_seq_length), :] for index in range(self.__len__())])

        # compute relative distance traveled in x-y plane (need to start from last end position!)
        d_Txyz = np.stack([np.array(self.gt_translation_xyz[index*self.imu_increment+self.imu_seq_length-1]) -
                           np.array(self.gt_translation_xyz[max(0, index*self.imu_increment-1)]) for index in range(self.__len__())])
        distance_items = np.sqrt(np.sum(np.square(d_Txyz[:, 0:2]), axis=1))

        # compute change in attitude
        q2 = [Q.Quaternion(self.gt_rotation_wxyz[index * self.imu_increment + self.imu_seq_length - 1]) for index in range(self.__len__())]
        q1 = [Q.Quaternion(self.gt_rotation_wxyz[max(0, index * self.imu_increment - 1)]) for index in range(self.__len__())]

        # change in heading in degrees (i.e. only rotation around world z axis = yaw)
        for ii in range(self.__len__()):
            q2[ii][1] = 0
            q2[ii][2] = 0
            q1[ii][1] = 0
            q1[ii][2] = 0

        self.delta_yaw_items = np.stack([(this_q2 * this_q1.conjugate).radians for this_q2, this_q1 in zip(q2, q1)])

        # compute polar angle associated with displacement
        angle_items = np.arctan2(d_Txyz[:, 1], d_Txyz[:, 0])

        self.d_gt_dist_angle_items = torch.tensor(np.stack((distance_items, angle_items), axis=1), dtype=torch.float64)

    def load_bags(self):
        # always need IMU and GT
        bag = rosbag.Bag(self.bagname)

        imus = bag.read_messages(topics=self.imu_topic)
        gts = bag.read_messages(topics=self.gt_topic)

        # extract all gt poses and timestamps
        translation_xyz = []
        rotation_wxyz = []
        gt_times = []

        for gt in gts:
            translation_xyz.append([gt.message.pose[2].position.x, gt.message.pose[2].position.y,
                                    gt.message.pose[2].position.z])
            rotation_wxyz.append([gt.message.pose[2].orientation.w, gt.message.pose[2].orientation.x,
                                  gt.message.pose[2].orientation.y, gt.message.pose[2].orientation.z])

        # extract all IMU readings and timestamps
        accel_xyz = []
        angvel_xyz = []
        imu_times = []

        for i, imu in enumerate(imus):
            imu_timestamp = imu.message.header.stamp.secs + imu.message.header.stamp.nsecs / 1.e9
            if i == 0:
                starttime = imu_timestamp

            # keep only readings within specified second limits
            accel_xyz.append([imu.message.linear_acceleration.x, imu.message.linear_acceleration.y, \
                              imu.message.linear_acceleration.z])
            angvel_xyz.append([imu.message.angular_velocity.x, imu.message.angular_velocity.y,
                               imu.message.angular_velocity.z])
            imu_times.append(imu_timestamp)

        self.gt_translation_xyz = np.array(translation_xyz)
        self.gt_rotation_wxyz = np.array(rotation_wxyz)
        self.imudata = np.concatenate([angvel_xyz, accel_xyz], axis=1)
        self.imu_times = np.array(imu_times)

