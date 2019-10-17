import numpy as np
from matplotlib import pyplot as plt

class Model:
    Schedule = []

    alpha_F = 0.1
    alpha_P = 0.1
    alpha_E = 0.1

    weight_F = 0
    weight_FE = 0
    weight_E = 0
    weight_P = 0

    def GenerateSchedule(self, scheduleList):
        """
        Generate CS, US Schedule
        :param scheduleList: List of 2 element list.
        ex) [ ['C',10], ['E',10], ['C',10] ]
        :return: np.array with n x 2 dim. (n= numTrial)
        """
        Schedule = np.empty((0,2), int)
        for t in range(len(scheduleList)):
            if type(scheduleList[t][1]) is not int:
                raise Exception('wrong schedule input')
            else :
                scheduleType = scheduleList[t][0]
                trial = scheduleList[t][1]
                # Generate Schedule array based on type
                if scheduleType == 'C': # conditioning
                    Schedule = np.append(Schedule, np.ones((trial,2)), axis=0)
                elif scheduleType == 'E': # extinction
                    Schedule = np.append(Schedule, np.concatenate([np.ones((trial, 1)), np.zeros((trial,1))], axis=1), axis=0)
                elif scheduleType == 'R': # remain
                    Schedule = np.append(Schedule, np.zeros((trial,2)), axis=0)
                elif scheduleType == 'T': # torture
                    Schedule = np.append(Schedule, np.concatenate([np.zeros((trial, 1)), np.ones((trial,1))], axis=1), axis=0)
        self.Schedule = Schedule
    def Run(self):

