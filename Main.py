import numpy as np
from matplotlib import pyplot as plt


class Model:
    def __init__(self, alpha_F=0.7, alpha_P=0.7, alpha_E=0.3):
        np.random.seed()
        self.Schedule = []
        self.alpha_F = alpha_F
        self.alpha_P = alpha_P
        self.alpha_E = alpha_E
        self.wipeMemory()

    def wipeMemory(self):
        # Clear neural activity
        self.F = np.array([0])
        self.P = np.array([0])
        self.E = np.array([0])
        # Clear Weights
        self.w_F = np.array([0])
        self.w_P = np.array([0])
        self.w_E = np.array([0])
        self.w_FE = 1
        # Clear dWeights
        self.d_w_F = np.array([0])
        self.d_w_P = np.array([0])
        self.d_w_E = np.array([0])
        print("""                                         
                                                              _                _ _ 
                                                             (_)              | | |
         _ __ ___   ___ _ __ ___   ___  _ __ _   _  __      ___ _ __   ___  __| | |
        | '_ ` _ \ / _ \ '_ ` _ \ / _ \| '__| | | | \ \ /\ / / | '_ \ / _ \/ _` | |
        | | | | | |  __/ | | | | | (_) | |  | |_| |  \ V  V /| | |_) |  __/ (_| |_|
        |_| |_| |_|\___|_| |_| |_|\___/|_|   \__, |   \_/\_/ |_| .__/ \___|\__,_(_)
                                              __/ |            | |                 
                                             |___/             |_|      
         """)

    def generateSchedule(self, scheduleList, P_prob=0.5):
        """
        Generate CS, US Schedule
        :param scheduleList: string of schedule. 'CharacterNumberCharacterNumber... *'
        F : full conditioning
        P : partial conditioning
        E : Extinction
        R : Rest
        T : Torture
        Ends with '*'
        Example : 'F10E10F10*'
        :param P_prob: probability of US in partial conditioning schedule
        :return: np.array with n x 2 dim. (n= numTrial)
        """
        # Initialize
        Schedule = np.empty((0, 2), int)
        self.ScheduleList = scheduleList
        # Generate Schedule array
        number_buffer = ''
        type_buffer = ''
        for t in range(len(scheduleList)):
            if scheduleList[t].isdigit():
                number_buffer += scheduleList[t]
            else:
                if number_buffer != '' or scheduleList[t] == '*':  # end of number
                    scheduleType = type_buffer
                    trial = int(number_buffer)
                    number_buffer = ''
                    # Generate Schedule array based on type
                    if scheduleType == 'F':  # full conditioning
                        Schedule = np.append(Schedule, np.ones((trial, 2)), axis=0)
                    elif scheduleType == 'P':  # partial conditioning
                        Schedule = np.append(Schedule, np.concatenate([np.ones((trial, 1)),
                                                                       np.random.choice([0, 1], size=(trial, 1),
                                                                                        p=[1 - P_prob, P_prob])],
                                                                      axis=1), axis=0)
                    elif scheduleType == 'E':  # extinction
                        Schedule = np.append(Schedule,
                                             np.concatenate([np.ones((trial, 1)), np.zeros((trial, 1))], axis=1),
                                             axis=0)
                    elif scheduleType == 'R':  # remain
                        Schedule = np.append(Schedule, np.zeros((trial, 2)), axis=0)
                    elif scheduleType == 'T':  # torture
                        Schedule = np.append(Schedule,
                                             np.concatenate([np.zeros((trial, 1)), np.ones((trial, 1))], axis=1),
                                             axis=0)
                    else:
                        raise Exception('Invalid type')
                type_buffer = scheduleList[t]
        self.Schedule = Schedule
        print("""
          _____      _              _       _         _____                           _           _ _ 
         / ____|    | |            | |     | |       / ____|                         | |         | | |
        | (___   ___| |__   ___  __| |_   _| | ___  | |  __  ___ _ __   ___ _ __ __ _| |_ ___  __| | |
         \___ \ / __| '_ \ / _ \/ _` | | | | |/ _ \ | | |_ |/ _ \ '_ \ / _ \ '__/ _` | __/ _ \/ _` | |
         ____) | (__| | | |  __/ (_| | |_| | |  __/ | |__| |  __/ | | |  __/ | | (_| | ||  __/ (_| |_|
        |_____/ \___|_| |_|\___|\__,_|\__,_|_|\___|  \_____|\___|_| |_|\___|_|  \__,_|\__\___|\__,_(_)
        """)

    def run(self):
        numTrial = np.size(self.Schedule, 0)

        for trial in range(numTrial):
            # Append neural activities
            self.P = np.append(self.P, self.w_P[-1] * self.Schedule[trial, 0])
            self.E = np.append(self.E, self.w_E[-1] * self.Schedule[trial, 0])
            self.F = np.append(self.F, self.w_F[-1] * self.Schedule[trial, 0] - self.w_FE * self.E[-1])

            # Calculate weight change
            d_w_F = self.alpha_F * self.Schedule[trial, 0] * max(0, self.Schedule[trial, 1] - self.F[-1])
            d_w_P = self.alpha_P * self.Schedule[trial, 0] * max(0, self.Schedule[trial, 1] - self.P[-1])
            d_w_E = self.alpha_E * self.Schedule[trial, 0] * max(0, self.F[-1] * (self.P[-1] - self.Schedule[trial, 1] - self.E[-1]))

            # Save dWeights
            self.d_w_F = np.append(self.d_w_F, d_w_F)
            self.d_w_P = np.append(self.d_w_P, d_w_P)
            self.d_w_E = np.append(self.d_w_E, d_w_E)

            # Apply weight change
            self.w_F = np.append(self.w_F, self.w_F[-1] + d_w_F)
            self.w_P = np.append(self.w_P, self.w_P[-1] + d_w_P)
            self.w_E = np.append(self.w_E, self.w_E[-1] + d_w_E)

        print("""
         _____   ____  _   _ ______ _ 
        |  __ \ / __ \| \ | |  ____| |
        | |  | | |  | |  \| | |__  | |
        | |  | | |  | | . ` |  __| | |
        | |__| | |__| | |\  | |____|_|
        |_____/ \____/|_| \_|______(_)
        """)

    def plot(self):
        # Generate figure
        fig = plt.figure(figsize=(6,9))
        fig.clf()
        fig.patch.set_facecolor('lightcyan')
        ax = fig.subplots(4,1)
        ax[0].set_xlim(0, np.size(self.Schedule, 0)+1)
        ax[0].set_ylim(0, 2.1)
        ax[0].bar(np.arange(np.size(self.Schedule, 0))+1, self.Schedule[:,0], color='blue', bottom = 1)
        ax[0].bar(np.arange(np.size(self.Schedule, 0))+1, self.Schedule[:,1], color='red')
        ax[0].legend(['CS','US'])
        ax[0].set_title('Schedule : ' + self.ScheduleList)

        ax[1].set_xlim(0, np.size(self.Schedule, 0)+1)
        ax[1].set_ylim(0, 1.1)
        ax[1].plot(self.F, color='red', linewidth=3)
        ax[1].plot(self.P, color='orange')
        ax[1].plot(self.E, color='blue')
        ax[1].legend(['Fear', 'Persistent', 'Extinction'])
        ax[1].set_title('Neural Response')

        ax[2].set_xlim(0, np.size(self.Schedule, 0)+1)
        ax[2].set_ylim(0, 1.1)
        ax[2].plot(self.w_F, color='red')
        ax[2].plot(self.w_P, color='orange')
        ax[2].plot(self.w_E, color='blue')
        ax[2].legend(['W_Fear', 'W_Persistent', 'W_Extinction'])
        ax[2].set_title(
            'Weights : $α_{F} : $' + str(self.alpha_F) + ' $α_{P} : $' + str(self.alpha_P) + ' $α_{E} : $' + str(self.alpha_E) + ' $w_{FE} : $' + str(self.w_FE))

        ax[3].set_xlim(0, np.size(self.Schedule, 0) + 1)
        ax[3].set_ylim(0, 1.1)
        ax[3].plot(self.d_w_F, color='red')
        ax[3].plot(self.d_w_P, color='orange')
        ax[3].plot(self.d_w_E, color='blue')
        ax[3].legend(['Learning signal_Fear', 'Learning signal_Persistent', 'Learning signal_Extinction'])
        ax[3].set_title('Learning Signals = delta weight')
        fig.show()

class Model_E(Model):
    # override
    def __init__(self, alpha_F=0.7, alpha_P=0.7, alpha_E1=0.3, alpha_E2 = 0.3):
        super(Model_E, self).__init__()
        np.random.seed()
        self.Schedule = []
        self.alpha_F = alpha_F
        self.alpha_P = alpha_P
        self.alpha_E1 = alpha_E1
        self.alpha_E2 = alpha_E2
        self.wipeMemory()

t = Model()
t.generateSchedule('F20E20*')
t.run()
t.plot()


