from controller import Robot, Receiver, Emitter
import sys, struct, math
import numpy as np
import mlp as ntw
import math


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
        self.current_generation = 0

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
        # Time tracking
        self.step_count = 0
        self.time_on_line = 0
        self.time_off_line = 0
        self.total_distance = 0
        self.distance_on_line = 0
        # Speed tracking
        self.speed_history = []
        self.avg_speed_on_line = 0
        # Collision tracking
        self.collision_count = 0
        self.near_collision_count = 0
        self.time_near_obstacle = 0
        # Line following tracking
        self.consecutive_on_line = 0
        self.consecutive_off_line = 0

        # Fitness value (initialization fitness parameters once)
        self.fitness_values = []
        self.fitness = 0
        self.num_generations = 300
        self.real_speed = 0
        self.is_on_edge = False
        self.action_number = 0
        self.position = 0,0

    def forwardFitness(self):
        """
        前进适应度函数

        参数:
            left_speed: 左轮速度
            right_speed: 右轮速度
            max_speed: 最大速度
        """
        # 奖励机制
        # 1. 两轮速度越快越好（鼓励快速移动）
        speed_reward = (abs(self.velocity_left) + abs(self.velocity_right)) / (2 * self.max_speed)

        # 2. 两轮速度差异越小越好（鼓励直线行驶）
        speed_difference = abs(self.velocity_left - self.velocity_right) / self.max_speed
        straightness_reward = 1.0 - speed_difference

        # 3. 两轮都应该正向旋转（惩罚倒退）
        direction_penalty = 0
        if self.velocity_left < 0 or self.velocity_right < 0:
            direction_penalty = 0.5

        # 综合适应度
        fitness = speed_reward * straightness_reward - direction_penalty
        # if self.real_speed <0.005:
        #     fitness = fitness-0.05
        # print("real speed:",self.real_speed)
        # # if self.is_on_edge:
        # #     fitness = 0.0
        # if self.real_speed>=0.07:
        #     fitness = 1.0
        # elif self.real_speed<0.07 and self.real_speed>=0.06:
        #     fitness = 0.8
        # elif self.real_speed<0.06 and self.real_speed>=0.05:
        #     fitness = 0.6
        # elif self.real_speed<0.05 and self.real_speed>=0.04:
        #     fitness = 0.4
        # elif self.real_speed<0.04 and self.real_speed>=0.02:
        #     fitness = 0.2
        # elif self.real_speed<0.02 and
        #     fitness = 0.1
        # else:
        #     fitness = 0.0
        # if self.real_speed < 0.01 and max(abs(self.velocity_left), abs(self.velocity_right)) > 0.5:
        #     return 0.0
        if  self.real_speed<0.01:
            fitness -= 0.1
        if self.is_on_edge:
            # if self.action_number % 100 == 0:
            #     print("is on the edge......")
            fitness -=0.2
        # if abs(self.velocity_right)!=0
        #     if abs(self.velocity_left)/abs(self.velocity_right)>0.8 and self.velocity_right*self.velocity_left<0:
        #         fitness=0.0



        return max(0, fitness)  # 确保适应度非负

    def followLineFitness(self):
        """
        循线适应度函数

        参数:
            ground_sensors: 地面传感器数组 [left, center, right]
            left_speed: 左轮速度
            right_speed: 右轮速度
            max_speed: 最大速度
        """

        left_sensor, center_sensor, right_sensor = self.left_ir.getValue(), self.center_ir.getValue(), self.right_ir.getValue()
        left_speed, right_speed = self.velocity_left, self.velocity_right
        max_speed = self.max_speed

        # 1. 线条检测奖励（中心传感器检测到线）
        line_detection_reward = 0
        if center_sensor < 500:  # 中心传感器在线上
            line_detection_reward = 1.0
        elif left_sensor < 500 or right_sensor < 500:  # 偏离但还能检测到
            line_detection_reward = 0.5

        # 2. 方向修正奖励
        correction_reward = 0
        if left_sensor < 500 and right_sensor > 500:  # 线在左侧，应该右转
            if right_speed > left_speed:
                correction_reward = 0.8
        elif right_sensor < 500 and left_sensor > 500:  # 线在右侧，应该左转
            if left_speed > right_speed:
                correction_reward = 0.8
        elif center_sensor < 500:  # 线在中心，应该直行
            if abs(left_speed - right_speed) < max_speed * 0.1:
                correction_reward = 1.0

        # 3. 速度奖励（在线上时保持速度）
        speed_reward = 0
        if line_detection_reward > 0.5:
            speed_reward = (left_speed + right_speed) / (2 * max_speed)

        # 4. 丢线惩罚
        lost_line_penalty = 0
        if all(sensor > 500 for sensor in [left_sensor,right_sensor,center_sensor]):
            lost_line_penalty = 1.0  # 完全丢线

        # 综合适应度
        if self.current_generation < 0.3 * self.num_generations:
            fitness = (line_detection_reward * 0.4 +
                       correction_reward * 0.3 +
                       speed_reward * 0.3 -
                       lost_line_penalty)
        else:
            fitness = (line_detection_reward * 0.6 +
                       correction_reward * 0.4  -
                       lost_line_penalty*0.15)
        # if self.real_speed <0.005 and self.is_on_edge:
        #     fitness = 0.0
        if self.is_on_edge :
            fitness = 0.0

        if self.real_speed<0.01:
            return 0.0

        # if self.real_speed < 0.01 and max(abs(self.velocity_left), abs(self.velocity_right)) > 0.5:
        #     fitness -= 0.2
        # if abs(self.velocity_right) != 0:
        #     if abs(self.velocity_left) / abs(
        #             self.velocity_right) > 0.8 and self.velocity_right * self.velocity_left < 0:
        #         fitness = 0.0




        if self.current_generation<0.3*self.num_generations:

            return max(0, fitness)
        else:
            return max(-0.2, fitness)

    # ============================================================================
    # FITNESS FUNCTION 3: AVOID COLLISION FITNESS
    # ============================================================================
    def avoidCollisionFitness(self):
        sensor_values = []
        for sensor in self.proximity_sensors:
            sensor_values.append(sensor.getValue())
        # proximity_sensors =sensor_values
        left_speed, right_speed, danger_threshold = self.velocity_left,self.velocity_right,80
        ps0 = sensor_values[0]
        ps1 = sensor_values[1]
        ps2 = sensor_values[2]
        ps7 =sensor_values[7]
        ps6= sensor_values[6]
        ps5= sensor_values[5]
        front_center = (ps1+ps6)/2
        front_right = (ps0+ps1)/2
        front_left = (ps6+ps7)/2
        right_side = ps2
        left_side = ps5
        front_obstacle = (ps0+ps1+ps6+ps7)/4
        vel_left = self.velocity_left
        vel_right = self.velocity_right
        fitness = 0.0
        reward = 3.0  # Base reward for correct avoidance behavior
        # if self.action_number%100==0:
        #     print("ps0-7:",ps0,ps1,ps2,ps5,ps6,ps7,"vel_left:",vel_left,"vel_right:",vel_right)

        # Define thresholds
        OBSTACLE_CLOSE = 90  # Obstacle is close enough to react
        OBSTACLE_VERY_CLOSE = 270  # Obstacle is very close (danger zone)
        SIDE_OBSTACLE = 90  # Side obstacle detection threshold
        FRONT_CLEAR = 80  # Front is considered clear below this

        # Calculate turning direction
        # Positive = turning right (right wheel faster)
        # Negative = turning left (left wheel faster)
        turn_direction = vel_right - vel_left
        is_turning_left = turn_direction < -0.3
        is_turning_right = turn_direction > 0.3
        is_moving_backward = (vel_left + vel_right) < -0.2
        is_slight_right_turn = 0.1 < turn_direction < 0.5
        is_slight_left_turn = -0.5 < turn_direction < -0.1

        ### SCENARIO 1: Front and Right obstacles → Reward LEFT turn
        # When obstacles are detected in front and to the right,
        # the robot should turn left to avoid collision
        if front_center > OBSTACLE_CLOSE and front_right > OBSTACLE_CLOSE:
            if is_turning_left and not self.is_on_edge:
                fitness += reward * 1.5
                if self.action_number%100==0:
                    print("Avoiding front-right obstacle by turning LEFT")
                # Extra reward if obstacle is very close
                if front_center > OBSTACLE_VERY_CLOSE and not self.is_on_edge:
                    fitness += reward * 0.5
                    if self.action_number%100==0:
                        print("Very close obstacle in front-right, extra reward for LEFT turn")

        ### SCENARIO 2: Front and Left obstacles → Reward RIGHT turn
        # When obstacles are detected in front and to the left,
        # the robot should turn right to avoid collision
        if front_center > OBSTACLE_CLOSE and front_left > OBSTACLE_CLOSE:
            if is_turning_right and not self.is_on_edge:
                fitness += reward * 1.5
                if self.action_number%100==0:
                    print("Avoiding front-left obstacle by turning RIGHT")
                # Extra reward if obstacle is very close
                if front_center > OBSTACLE_VERY_CLOSE:
                    fitness += reward * 0.5
                    if self.action_number%100==0:
                        print("Very close obstacle in front-left, extra reward for RIGHT turn")

        ### SCENARIO 3: Very close obstacle (almost collision) → Reward BACKWARD movement
        # When the robot is about to collide, backing up is the safest option
        if front_center > OBSTACLE_VERY_CLOSE:
            if is_moving_backward and not self.is_on_edge:
                fitness += reward * 2.0
                if self.action_number%100==0:
                    print("Very close obstacle ahead, rewarding BACKWARD movement")
                # Strong reward for backing away from imminent collision

        ### SCENARIO 4: Right side obstacle, front clear → Reward slight RIGHT turn
        # After avoiding an obstacle on the right, the robot should turn slightly right
        # to return to the original trajectory (line)
        if right_side > SIDE_OBSTACLE and front_center < FRONT_CLEAR:
            if is_slight_right_turn and not self.is_on_edge:
                fitness += reward * 5.0
                if self.action_number%100==0:
                    print("Avoided right side obstacle, rewarding slight RIGHT turn")
                # This helps the robot return to the line after avoiding obstacle

        ### SCENARIO 5: Left side obstacle, front clear → Reward slight LEFT turn
        # After avoiding an obstacle on the left, the robot should turn slightly left
        # to return to the original trajectory (line)
        if left_side > SIDE_OBSTACLE and front_center < FRONT_CLEAR:
            if is_slight_left_turn and not self.is_on_edge:
                fitness += reward * 5.0
                if self.action_number%100==0:
                    print("Avoided left side obstacle, rewarding slight LEFT turn")
                # This helps the robot return to the line after avoiding obstacle

        ### PENALTY: Collision risk
        # Penalize if moving forward when obstacle is very close
        if front_center > OBSTACLE_VERY_CLOSE:
            forward_speed = (vel_left + vel_right) / 2.0
            if forward_speed > 0.2:  # Moving forward toward obstacle
                fitness -= reward * 2.0
                # if self.action_number%100==0:
                #     print("Penalty for moving FORWARD toward very close obstacle")
        if self.real_speed < 0.01 and not self.is_on_edge:
            # if self.action_number%100==0:
            #     print("real_speed too low...")
            return 0.0


        return fitness
        # """
        # 避障适应度函数
        #
        # 目标：使机器人能够检测并避开障碍物
        #
        # 原理：
        # - e-puck有8个红外接近传感器，分布在机器人周围
        # - 传感器值越高表示障碍物越近
        # - 前方传感器最重要，侧面次之
        # Args:
        #     proximity_sensors: 8个接近传感器读数 [ps0-ps7]
        #                       ps0, ps1: 右前方
        #                       ps2, ps3: 右侧
        #                       ps4, ps5: 后方
        #                       ps6, ps7: 左侧/左前方
        #     left_speed: 左轮速度
        #     right_speed: 右轮速度
        #     danger_threshold: 危险距离阈值
        #
        # Returns:
        #     适应度得分 [0, 1]
        #
        # 设计要点：
        # 1. 危险检测：识别前方和侧面的障碍物
        # 2. 避障响应：根据障碍物位置调整轮速
        # 3. 预防性奖励：保持安全距离
        # 4. 惩罚碰撞：传感器值过高严重惩罚
        # """
        # sensor_values = []
        # for sensor in self.proximity_sensors:
        #     sensor_values.append(sensor.getValue())
        # proximity_sensors =sensor_values
        # left_speed, right_speed, danger_threshold = self.velocity_left,self.velocity_right,90
        # if len(proximity_sensors) < 8:
        #     return 0.0
        #
        # # 传感器权重（前方最重要）
        # sensor_weights = np.array([
        #     0.2,  # ps0 - 右前
        #     0.2,  # ps1 - 右前
        #     0.1,  # ps2 - 右侧
        #     0.05,  # ps3 - 右后侧
        #     0.05,  # ps4 - 后方
        #     0.05,  # ps5 - 后方
        #     0.1,  # ps6 - 左侧
        #     0.2  # ps7 - 左前
        # ])
        #
        # # 归一化传感器读数 [0, 1]，假设最大值为4096
        # norm_sensors = np.array(proximity_sensors) / 4096.0
        #
        # # 计算加权危险程度
        # danger_level = np.sum(norm_sensors * sensor_weights)
        #
        # # 检测前方障碍物
        # front_sensors = [proximity_sensors[0], proximity_sensors[1],
        #                  proximity_sensors[7]]
        # max_front = max(front_sensors)
        # # print("max_front:",max_front)
        #
        # # 碰撞惩罚
        # if max_front > danger_threshold * 3:
        #     return 0.0  # 严重碰撞
        #
        # # 计算避障得分
        # if max_front < danger_threshold:
        #     # 安全距离，高分
        #     safety_score = 1.0
        # else:
        #     # 有障碍物，根据距离评分
        #     safety_score = 1.0 - (max_front - danger_threshold) / (danger_threshold * 2)
        #     safety_score = max(0.2, safety_score)
        #
        # # 评估避障行为
        # left_obstacle = proximity_sensors[7] > danger_threshold
        # right_obstacle = proximity_sensors[0] > danger_threshold
        #
        # avoidance_score = 0.0
        # if left_obstacle and right_speed < left_speed and left_speed-right_speed>1.5:
        #     # 左侧有障碍，应该右转（右轮慢）
        #     avoidance_score = 1.0
        # elif left_obstacle and right_speed < left_speed and left_speed - right_speed > 1.0:
        #     # 左侧有障碍，应该右转（右轮慢）
        #     avoidance_score = 0.7
        # elif left_obstacle and right_speed < left_speed and left_speed - right_speed > 0.8:
        #     # 左侧有障碍，应该右转（右轮慢）
        #     avoidance_score = 0.6
        # elif left_obstacle and right_speed < left_speed and left_speed - right_speed > 0.6:
        #     # 左侧有障碍，应该右转（右轮慢）
        #     avoidance_score = 0.4
        # elif left_obstacle and right_speed < left_speed and left_speed - right_speed > 0.4:
        #     # 左侧有障碍，应该右转（右轮慢）
        #     avoidance_score = 0.3
        # elif left_obstacle and right_speed < left_speed and left_speed - right_speed > 0.2:
        #     # 左侧有障碍，应该右转（右轮慢）
        #     avoidance_score = 0.2
        # elif left_obstacle and right_speed < left_speed and left_speed - right_speed > 0.1:
        #     # 左侧有障碍，应该右转（右轮慢）
        #     avoidance_score = 0.1
        # if right_obstacle and left_speed < right_speed and right_speed-left_speed>1.5:
        #     # 右侧有障碍，应该左转（左轮慢）
        #     avoidance_score = 1.0
        # elif right_obstacle and left_speed < right_speed and right_speed-left_speed>1.0:
        #     # 右侧有障碍，应该左转（左轮慢）
        #     avoidance_score = 0.7
        # elif right_obstacle and left_speed < right_speed and right_speed-left_speed>0.8:
        #     # 右侧有障碍，应该左转（左轮慢）
        #     avoidance_score = 0.6
        # elif right_obstacle and left_speed < right_speed and right_speed-left_speed>0.6:
        #     # 右侧有障碍，应该左转（左轮慢）
        #     avoidance_score = 0.4
        # elif right_obstacle and left_speed < right_speed and right_speed-left_speed>0.4:
        #     # 右侧有障碍，应该左转（左轮慢）
        #     avoidance_score = 0.3
        # elif right_obstacle and left_speed < right_speed and right_speed-left_speed>0.2:
        #     # 右侧有障碍，应该左转（左轮慢）
        #     avoidance_score = 0.2
        # elif right_obstacle and left_speed < right_speed and right_speed-left_speed>0.1:
        #     # 右侧有障碍，应该左转（左轮慢）
        #     avoidance_score = 0.1
        #
        # # 鼓励在检测到障碍时减速
        # if max_front > danger_threshold:
        #     avg_speed = (abs(left_speed) + abs(right_speed)) / 2.0
        #     speed_reduction = 1.0 - min(avg_speed /self.max_speed, 1.0)
        #     avoidance_score *= (0.7 + 0.3 * speed_reduction)
        #
        # # 综合得分
        # if self.current_generation<0.5*self.num_generations:
        #     fitness = safety_score * 1.0 + avoidance_score * 3
        # else:
        #     fitness = safety_score * 1.0 + avoidance_score * 6
        # if self.is_on_edge:
        #
        #     fitness = 0.0
        # if self.real_speed<0.02 and max(abs(self.velocity_left), abs(self.velocity_right))>0.5:
        #     fitness=0.0
        # if abs(self.velocity_right)!=0:
        #     if abs(self.velocity_left)/abs(self.velocity_right)>0.8 and self.velocity_right*self.velocity_left<0:
        #         fitness=0.0
        # if self.is_on_edge:
        #     fitness =0.0
        #
        # return max(0.0, fitness)


        # return np.clip(fitness, 0.0, 1.0)


    # ============================================================================
    # FITNESS FUNCTION 4: SPINNING FITNESS (PENALTY)
    # ============================================================================

    def spinningFitness(self) :
        """
        旋转惩罚函数

        目标：惩罚原地旋转和无效的振荡行为

        原理：
        - 原地旋转：两轮速度大小相等方向相反
        - 持续振荡：频繁改变转向方向
        - 这些行为浪费时间且无助于任务完成

        Args:
            left_speed: 左轮速度
            right_speed: 右轮速度
            angular_velocity_history: 历史角速度记录

        Returns:惩罚得分 [0, 1]，1表示无惩罚，0表示最大惩罚

        设计要点：
        1. 检测原地旋转：速度相反且大小相近
        2. 检测振荡：频繁改变转向方向
        3. 允许必要的转向：小幅度转向不惩罚
        4. 时间惩罚：持续旋转增加惩罚
        """
        # 计算角速度（简化模型）
        # 正值表示逆时针旋转，负值表示顺时针旋转
        self.action_number +=1
        # if(self.action_number%30==0):
        #     print("self.velocity_left:",self.velocity_left,"self.velocity_right:",self.velocity_right)
        left_speed, right_speed,angular_velocity_history=self.velocity_left,self.velocity_right,None
        angular_velocity = right_speed - left_speed

        # 检测原地旋转
        speed_sum = abs(left_speed) + abs(right_speed)
        speed_diff = abs(abs(left_speed) - abs(right_speed))
        if self.real_speed < 0.0001:
            return 0.0

        if self.real_speed<0.01 and max(abs(self.velocity_left), abs(self.velocity_right))>0.5:
            return 0.0
        if abs(self.velocity_right)!=0:
            if abs(self.velocity_left) / abs(self.velocity_right) > 0.8 and self.velocity_right * self.velocity_left < 0:
                return 0.0


        if speed_sum < 0.1:
            # 几乎静止，不惩罚
            return 1.0

        # 原地旋转检测：速度相反且大小相近
        if left_speed * right_speed < 0:  # 符号相反
            similarity = 1.0 - speed_diff / (speed_sum + 1e-6)
            if similarity > 0.8:
                # 明显的原地旋转
                spinning_penalty = similarity
                return max(0.0,1.0 - spinning_penalty * 0.8)

        # 检测振荡行为
        if angular_velocity_history and len(angular_velocity_history) > 5:
            recent_history = angular_velocity_history[-10:]

            # 计算方向改变次数
            direction_changes = 0
            for i in range(1, len(recent_history)):
                if recent_history[i] * recent_history[i - 1] < 0:
                    direction_changes += 1

            # 频繁改变方向表示振荡
            if direction_changes > 5:
                oscillation_penalty = min(direction_changes / 10.0, 0.6)
                return max(0,1.0 - oscillation_penalty)


        # if self.is_on_edge:
        #     return 0.0
        # print(max(abs(self.velocity_left), abs(self.velocity_right)))
        # 轻微转向不惩罚
        turn_ratio = abs(angular_velocity) / (speed_sum + 1e-6)

        if turn_ratio < 0.3:
            return 1.0



        # 中等转向轻微惩罚
        return max(0.0,1.0 - turn_ratio * 0.2)

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
            return min_max
        elif (value < -min_max):
            return -min_max
        return value

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

    def get_adaptive_weights(self, generation, max_generations):
        progress = generation / max_generations

        # 使用sigmoid平滑过渡
        def smooth_transition(x, center, steepness=10):
            return 1 / (1 + np.exp(-steepness * (x - center)))

        # 前进权重：从0.5平滑降到0.2
        forward_weight = 0.5 - 0.3 * smooth_transition(progress, 0.5)

        # 循线权重：从0.2平滑升到0.4再降
        followline_weight = 0.2 + 0.3 * np.sin(progress * np.pi)

        # 避障权重：从0.25平滑升到0.35
        avoid_weight = 0.25 + 0.1 * smooth_transition(progress, 0.7)

        # 归一化
        total = forward_weight + followline_weight + avoid_weight + 0.05
        return {
            'forward': forward_weight / total,
            'followLine': followline_weight / total,
            'avoidCollision': avoid_weight / total,
            'spinning': 0.05 / total
        }

    def calculate_fitness(self):
        # pos = self.robot.getSelf().getPosition()
        # print("Robot Position: x {:.3f} y {:.3f} z {:.3f}".format(pos[0],pos[1],pos[2]))
        #
        ### Define the fitness function to increase the speed of the robot and
        ### to encourage the robot to move forward only
        forwardFitness = self.forwardFitness()

        ### Define the fitness function to encourage the robot to follow the line
        followLineFitness = self.followLineFitness()

        ### Define the fitness function to avoid collision
        avoidCollisionFitness = self.avoidCollisionFitness()

        ### Define the fitness function to avoid spining behaviour
        spinningFitness = self.spinningFitness()
        # if self.action_number%100 == 0:
        #     print("Fitness Components - Forward: {:.3f}, Line: {:.3f}, Avoid Collision: {:.3f}, Spinning Penalty: {:.3f}".format(forwardFitness, followLineFitness, avoidCollisionFitness, spinningFitness))
        # if avoidCollisionFitness<0.0:
        #     print("avoidCollisionFitness negative:", avoidCollisionFitness)
        # if spinningFitness<0.0:
        #     print("spinningFitness negative:", spinningFitness)
        # if forwardFitness<0.0:
        #     print("forwardFitness negative:", forwardFitness)
        # if followLineFitness<0.0:
        #     print("followLineFitness negative:", followLineFitness)
        # if self.current_generation <= 0.3 * self.num_generations:
        #     fitnessWeightsMapping = {"forwardFitness": 0.50, "followLineFitness": 0.2, "avoidCollisionFitness": 0.25,
        #                              "spinningFitness": 0.05}
        # elif self.current_generation > 0.3 * self.num_generations and self.current_generation <= 0.7 * self.num_generations:
        #     fitnessWeightsMapping = {"forwardFitness": 0.25, "followLineFitness": 0.5, "avoidCollisionFitness": 0.2,
        #                              "spinningFitness": 0.05}
        # elif self.current_generation > 0.7 * self.num_generations and self.current_generation <= self.num_generations:
        #     fitnessWeightsMapping = {"forwardFitness": 0.2, "followLineFitness": 0.4, "avoidCollisionFitness": 0.35,
        #                              "spinningFitness": 0.05}
        # if self.action_number % 100 == 0:
        #     print("num_generatetions:", self.num_generations, " current_generation:", self.current_generation)
        #     print("Fitness Weights Mapping:", fitnessWeightsMapping)
        adaptive_weights = self.get_adaptive_weights(self.current_generation,self.num_generations)


        ### Define the fitness function of this iteration which should be a combination of the previous functions
        combinedFitness = forwardFitness * adaptive_weights['forward'] + followLineFitness * \
                          adaptive_weights['followLine'] + avoidCollisionFitness * adaptive_weights['avoidCollision'] + spinningFitness * adaptive_weights['spinning']

        # if self.action_number % 100 == 0:
        #     # print("avoidCollisionFitness:", avoidCollisionFitness)
        #     print("Fitness Components - Forward: {:.3f}, Line: {:.3f}, Avoid Collision: {:.3f}, Spinning Penalty: {:.3f}".format(forwardFitness, followLineFitness, avoidCollisionFitness, spinningFitness))
        #     # print("real speed:",self.real_speed)
        #     print("velocity_left, velocity_right:",self.velocity_left,self.velocity_right)
        self.fitness_values.append(combinedFitness)
        self.fitness = np.mean(self.fitness_values)
        # if self.action_number % 200 == 0:
        #     print("Adaptive Weights:", adaptive_weights)
        #     print("forwardFitness:", forwardFitness, " followLineFitness:", followLineFitness,"avoidCollisionFitness:", avoidCollisionFitness, " spinningFitness:", spinningFitness)
        #     print("Combined Fitness:", combinedFitness)
        #     print("fitness:", self.fitness)

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
                data_from_supervisor = self.receiver.getString()
                # print("robot received:", self.receivedData)
                # print(type(self.receivedData))
                if data_from_supervisor.startswith("genotype: "):
                    # print(data_from_supervisor)
                    self.receivedData = data_from_supervisor[11:-1]
                    self.receivedData = self.receivedData.split()
                    x = np.array(self.receivedData)
                    self.receivedData = x.astype(float)
                    # print("Controller handle receiver data:", self.receivedData)
                elif data_from_supervisor.startswith("current_generation: "):
                    generation_data = data_from_supervisor[20:len(data_from_supervisor)]
                    # print("Received generation data:", data_from_supervisor)
                    self.current_generation = int(generation_data)
                    # print("Controller handle receiver generation:", self.current_generation)
                elif data_from_supervisor.startswith("num_generations: "):
                    num_generations = data_from_supervisor[17:len(data_from_supervisor)]
                    self.num_generations = int(num_generations)
                    # print("Controller handle receiver population:", self.num_generations)
                elif data_from_supervisor.startswith("real_speed: "):
                    speed_data = data_from_supervisor[12:len(data_from_supervisor)]
                    self.real_speed = float(speed_data)
                    # print("Controller handle receiver real speed:", self.real_speed)
                elif data_from_supervisor.startswith("position: "):
                    position_data = data_from_supervisor[10:len(data_from_supervisor)]
                    # print("Received position data:", position_data)
                    # Convert string representation of list to actual list
                    pos = eval(position_data)
                    self.position = pos[0],pos[1]
                    x, y, z = pos
                    if abs(x) > 0.69 or abs(y) > 0.69:
                        self.is_on_edge = True
                        # if self.is_on_edge:
                        #     if self.action_number % 100 == 0:
                        #         print("x,y:{},{}".format(x,y))
                    # print("Controller handle receiver position:", position_list)
                    else:
                        self.is_on_edge = False
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
            max_gs = 100

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
                    max_ds = 100

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

def run(robot):
    controller = Controller(robot)
    controller.run_robot()

if __name__ == "__main__":
    # Call Robot function to initialize the robot
    my_robot = Robot()
    # Initialize the parameters of the controller by sending my_robot
    controller = Controller(my_robot)
    # Run the controller
    controller.run_robot()
