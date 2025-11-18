from controller import Robot, Receiver, Emitter
import sys, struct, math
import numpy as np
import mlp as ntw


class Controller:
    def __init__(self, robot):
        # Robot Parameters
        # Please, do not change these parameters
        self.robot = robot
        self.time_step = 32  # ms
        self.max_speed = 1  # m/s

        # MLP Parameters and Variables
        ### Define below the architecture of your MLP network.
        ### Add the number of neurons for each layer.
        ### The number of neurons should be in between of 1 to 20.
        ### Number of hidden layers should be one or two.
        self.number_input_layer = 11  # 8 proximity + 3 ground sensors
        # Example with one hidden layers: self.number_hidden_layer = [5]
        # Example with two hidden layers: self.number_hidden_layer = [7,5]
        # FILLED: Two hidden layers for feature extraction and decision-making
        self.number_hidden_layer = [8, 6]
        self.number_output_layer = 2

        # Create a list with the number of neurons per layer
        self.number_neuros_per_layer = []
        self.number_neuros_per_layer.append(self.number_input_layer)
        self.number_neuros_per_layer.extend(self.number_hidden_layer)
        self.number_neuros_per_layer.append(self.number_output_layer)

        # Initialize the network
        self.network = ntw.MLP(self.number_neuros_per_layer)
        self.inputs = []

        # Calculate the number of weights of your MLP
        self.number_weights = 0
        for n in range(1, len(self.number_neuros_per_layer)):
            if (n == 1):
                # Input + bias
                self.number_weights += (self.number_neuros_per_layer[n - 1] + 1) * self.number_neuros_per_layer[n]
            else:
                self.number_weights += self.number_neuros_per_layer[n - 1] * self.number_neuros_per_layer[n]

        # Enable Motors
        self.left_motor = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)
        self.velocity_left = 0
        self.velocity_right = 0

        # Enable Proximity Sensors
        self.proximity_sensors = []
        for i in range(8):
            sensor_name = 'ps' + str(i)
            self.proximity_sensors.append(self.robot.getDevice(sensor_name))
            self.proximity_sensors[i].enable(self.time_step)

        # Enable Ground Sensors
        self.left_ir = self.robot.getDevice('gs0')
        self.left_ir.enable(self.time_step)
        self.center_ir = self.robot.getDevice('gs1')
        self.center_ir.enable(self.time_step)
        self.right_ir = self.robot.getDevice('gs2')
        self.right_ir.enable(self.time_step)

        # Enable Emitter and Receiver (to communicate with the Supervisor)
        self.emitter = self.robot.getDevice("emitter")
        self.receiver = self.robot.getDevice("receiver")
        self.receiver.enable(self.time_step)
        self.receivedData = ""
        self.receivedDataPrevious = ""
        self.flagMessage = False

        # Fitness value (initialization fitness parameters once)
        self.fitness_values = []
        self.fitness = 0

    def check_for_new_genes(self):
        if (self.flagMessage == True):
            # Split the list based on the number of layers of your network
            part = []
            for n in range(1, len(self.number_neuros_per_layer)):
                if (n == 1):
                    part.append((self.number_neuros_per_layer[n - 1] + 1) * (self.number_neuros_per_layer[n]))
                else:
                    part.append(self.number_neuros_per_layer[n - 1] * self.number_neuros_per_layer[n])

            # Set the weights of the network
            data = []
            weightsPart = []
            sum = 0
            for n in range(1, len(self.number_neuros_per_layer)):
                if (n == 1):
                    weightsPart.append(self.receivedData[n - 1:part[n - 1]])
                elif (n == (len(self.number_neuros_per_layer) - 1)):
                    weightsPart.append(self.receivedData[sum:])
                else:
                    weightsPart.append(self.receivedData[sum:sum + part[n - 1]])
                sum += part[n - 1]
            for n in range(1, len(self.number_neuros_per_layer)):
                if (n == 1):
                    weightsPart[n - 1] = weightsPart[n - 1].reshape(
                        [self.number_neuros_per_layer[n - 1] + 1, self.number_neuros_per_layer[n]])
                else:
                    weightsPart[n - 1] = weightsPart[n - 1].reshape(
                        [self.number_neuros_per_layer[n - 1], self.number_neuros_per_layer[n]])
                data.append(weightsPart[n - 1])
            self.network.weights = data

            # Reset fitness list
            self.fitness_values = []

    def clip_value(self, value, min_max):
        if (value > min_max):
            return min_max;
        elif (value < -min_max):
            return -min_max;
        return value;

    def sense_compute_and_actuate(self):
        # MLP:
        #   Input == sensory data
        #   Output == motors commands
        output = self.network.propagate_forward(self.inputs)
        self.velocity_left = output[0]
        self.velocity_right = output[1]

        # Multiply the motor values by 3 to increase the velocities
        self.left_motor.setVelocity(self.velocity_left * 3)
        self.right_motor.setVelocity(self.velocity_right * 3)

    def calculate_fitness(self):
        """
        OPTIMIZED FITNESS FUNCTION v2.0

        Key improvements:
        1. Strong line following emphasis with off-center penalty
        2. Context-aware obstacle avoidance (threshold-based)
        3. Speed bonus when following line correctly
        4. Permissive spinning penalty for navigation

        Expected fitness range: -10 to +20
        Good performance: 8-12
        Excellent performance: 12+
        """

        # Ground sensors (normalized 0-1)
        # 0 = white surface (no line), 1 = black surface (line detected)
        left = self.inputs[0]  # Left ground sensor (gs0)
        center = self.inputs[1]  # Center ground sensor (gs1) - MOST IMPORTANT
        right = self.inputs[2]  # Right ground sensor (gs2)

        # Proximity sensors (normalized 0-1)
        # 0 = no obstacle, 1 = very close obstacle
        # Front sensors are most important for obstacle avoidance
        # ps0 = inputs[3], ps1 = inputs[4], ps6 = inputs[9], ps7 = inputs[10]
        front_obstacle = (self.inputs[3] + self.inputs[4] +
                          self.inputs[9] + self.inputs[10]) / 4.0

        ### 1. FORWARD FITNESS - Reward speed and forward movement
        # Encourages robot to move forward rather than backward or stationary
        # Range: -3 to +3 (motors are multiplied by 3)
        # Positive = forward, Negative = backward
        forwardFitness = (self.velocity_left + self.velocity_right) / 2.0

        ### 2. ENHANCED LINE FOLLOWING FITNESS
        # Strong reward for center sensor detecting line
        # Penalty for left/right sensors detecting line (robot is off-center)
        # This encourages the robot to stay centered on the line
        # Range: approximately -3 to +4
        followLineFitness = center * 4.0 - (left + right) * 1.5

        ### 3. SPEED BONUS - Encourage fast movement when on line
        # Rewards robot for moving quickly when it's correctly following the line
        # This prevents slow, cautious behavior when on track
        # Range: 0 to +6
        if center > 0.5:  # Robot is clearly on the line
            speedBonus = forwardFitness * 2.0
        else:  # Robot is off the line or uncertain
            speedBonus = 0

        ### 4. CONTEXT-AWARE COLLISION AVOIDANCE
        # Uses thresholds to only penalize when obstacles are genuinely close
        # This prevents the robot from being overly cautious and staying too far from obstacles
        # Range: -3 to 0
        if front_obstacle > 0.7:  # Very close to obstacle (danger zone)
            # Heavy penalty - must avoid immediately
            avoidCollisionFitness = -(front_obstacle - 0.7) * 10.0
        elif front_obstacle > 0.4:  # Moderately close (caution zone)
            # Light penalty - be aware but can navigate
            avoidCollisionFitness = -(front_obstacle - 0.4) * 2.0
        else:  # Safe distance (< 0.4)
            # No penalty - robot can move freely
            avoidCollisionFitness = 0

        ### 5. ANTI-SPINNING FITNESS
        # Penalizes excessive difference in wheel velocities (spinning in place)
        # But allows reasonable turning for navigation around obstacles
        # Range: -1.5 to 0
        speed_diff = abs(self.velocity_left - self.velocity_right)
        if speed_diff > 2.5:  # Excessive spinning threshold (was 0 before)
            spinningFitness = -(speed_diff - 2.5) * 1.0
        else:  # Reasonable turning or straight movement
            spinningFitness = 0

        ### 6. COMBINED FITNESS with optimized weights
        # Weight distribution:
        # - followLineFitness: 3.0x (highest priority - PRIMARY OBJECTIVE)
        # - speedBonus: 1.0x (encourage fast line following)
        # - forwardFitness: 1.0x (base movement reward)
        # - avoidCollisionFitness: 1.0x (safety, but not overly cautious)
        # - spinningFitness: 0.5x (fine-tuning stability)
        #
        # Total expected range: -10 to +20
        # Poor performance: < 0
        # Moderate: 0-5
        # Good: 5-10
        # Excellent: 10-15
        # Outstanding: 15+
        combinedFitness = (
                forwardFitness * 1.0 +  # Base speed reward
                followLineFitness * 3.0 +  # PRIMARY OBJECTIVE (36% of positive fitness)
                speedBonus * 1.0 +  # Fast line following bonus
                avoidCollisionFitness * 1.0 +  # Context-aware safety
                spinningFitness * 0.5  # Stability fine-tuning
        )

        # Store fitness value and calculate running average
        self.fitness_values.append(combinedFitness)
        self.fitness = np.mean(self.fitness_values)

    def handle_emitter(self):
        # Send the self.fitness value to the supervisor
        data = str(self.number_weights)
        data = "weights: " + data
        string_message = str(data)
        string_message = string_message.encode("utf-8")
        # print("Robot send:", string_message)
        self.emitter.send(string_message)

        # Send the self.fitness value to the supervisor
        data = str(self.fitness)
        data = "fitness: " + data
        string_message = str(data)
        string_message = string_message.encode("utf-8")
        # print("Robot send fitness:", string_message)
        self.emitter.send(string_message)

    def handle_receiver(self):
        if self.receiver.getQueueLength() > 0:
            while (self.receiver.getQueueLength() > 0):
                # Adjust the Data to our model
                # Webots 2022:
                # self.receivedData = self.receiver.getData().decode("utf-8")
                # Webots 2023:
                self.receivedData = self.receiver.getString()

                self.receivedData = self.receivedData[1:-1]
                self.receivedData = self.receivedData.split()
                x = np.array(self.receivedData)
                self.receivedData = x.astype(float)
                # print("Controller handle receiver data:", self.receivedData)
                self.receiver.nextPacket()

            # Is it a new Genotype?
            if (np.array_equal(self.receivedDataPrevious, self.receivedData) == False):
                self.flagMessage = True

            else:
                self.flagMessage = False

            self.receivedDataPrevious = self.receivedData
        else:
            # print("Controller receiver q is empty")
            self.flagMessage = False

    def run_robot(self):
        # Main Loop
        while self.robot.step(self.time_step) != -1:
            # This is used to store the current input data from the sensors
            self.inputs = []

            # Emitter and Receiver
            # Check if there are messages to be sent or read to/from our Supervisor
            self.handle_emitter()
            self.handle_receiver()

            # Read Ground Sensors
            left = self.left_ir.getValue()
            center = self.center_ir.getValue()
            right = self.right_ir.getValue()
            # print("Ground Sensors \n    left {} center {} right {}".format(left,center,right))

            ### Please adjust the ground sensors values to facilitate learning
            min_gs = 0
            max_gs = 1000

            if (left > max_gs): left = max_gs
            if (center > max_gs): center = max_gs
            if (right > max_gs): right = max_gs
            if (left < min_gs): left = min_gs
            if (center < min_gs): center = min_gs
            if (right < min_gs): right = min_gs

            # Normalize the values between 0 and 1 and save data
            self.inputs.append((left - min_gs) / (max_gs - min_gs))
            self.inputs.append((center - min_gs) / (max_gs - min_gs))
            self.inputs.append((right - min_gs) / (max_gs - min_gs))
            # print("Ground Sensors \n    left {} center {} right {}".format(self.inputs[0],self.inputs[1],self.inputs[2]))

            # Read Distance Sensors
            for i in range(8):
                ### Select the distance sensors that you will use
                if (i == 0 or i == 1 or i == 2 or i == 3 or i == 4 or i == 5 or i == 6 or i == 7):
                    temp = self.proximity_sensors[i].getValue()

                    ### Please adjust the distance sensors values to facilitate learning
                    min_ds = 0
                    max_ds = 2400

                    if (temp > max_ds): temp = max_ds
                    if (temp < min_ds): temp = min_ds

                    # Normalize the values between 0 and 1 and save data
                    self.inputs.append((temp - min_ds) / (max_ds - min_ds))
                    # print("Distance Sensors - Index: {}  Value: {}".format(i,self.proximity_sensors[i].getValue()))

            # GA Iteration
            # Verify if there is a new genotype to be used that was sent from Supervisor
            self.check_for_new_genes()
            # Define the robot's actuation (motor values) based on the output of the MLP
            self.sense_compute_and_actuate()
            # Calculate the fitnes value of the current iteration
            self.calculate_fitness()

            # End of the iteration


if __name__ == "__main__":
    # Call Robot function to initialize the robot
    my_robot = Robot()
    # Initialize the parameters of the controller by sending my_robot
    controller = Controller(my_robot)
    # Run the controller
    controller.run_robot()
