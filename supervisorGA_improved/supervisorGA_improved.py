from controller import Supervisor
from controller import Keyboard
from controller import Display

import numpy as np
import struct, math
import ga_improved as ga
import os, sys


class ImprovedSupervisorGA:
    def __init__(self):
        # Simulation Parameters
        self.time_step = 32  # ms
        self.time_experiment = 150  # s
        
        # Initiate Supervisor Module
        self.supervisor = Supervisor()
        
        # Check if the robot node exists
        self.robot_node = self.supervisor.getFromDef("Controller")
        if self.robot_node is None:
            sys.stderr.write("No DEF Controller node found in the current world file\n")
            sys.exit(1)
        
        # Get robot translation and rotation fields
        self.trans_field = self.robot_node.getField("translation")
        self.rot_field = self.robot_node.getField("rotation")
        
        # Communication
        self.emitter = self.supervisor.getDevice("emitter")
        self.receiver = self.supervisor.getDevice("receiver")
        self.receiver.enable(self.time_step)
        
        # Data
        self.receivedData = ""
        self.receivedWeights = ""
        self.receivedFitness = ""
        self.emitterData = ""
        self.current_generation = 0
        
        ### 改进的GA参数
        self.num_generations = 60  # 增加到200代
        self.num_population = 60    # 增加到80个个体
        self.num_elite = 6         # 精英数量增加到8
        
        # 初始化改进的GA
        self.ga = ga.ImprovedGA(
            population_size=self.num_population,
            elite_size=self.num_elite,
            initial_crossover_rate=0.85,
            initial_mutation_rate=0.12,
            tournament_size=7,
            adaptive=True
        )
        
        self.num_weights = 0
        self.population = []
        self.genotypes = []
        self.real_speed = 0.0
        self.position_history = []
        
        # 性能追踪
        self.best_fitness_history = []
        self.avg_fitness_history = []
        self.diversity_history = []
        self.stagnation_counter = 0
        self.last_best_fitness = 0.0
        
        # Display
        self.display = self.supervisor.getDevice("display")
        self.width = self.display.getWidth()
        self.height = self.display.getHeight()
        self.prev_best_fitness = 0.0
        self.prev_average_fitness = 0.0
        self.display.drawText("Fitness (Best - Red)", 0, 0)
        self.display.drawText("Fitness (Average - Green)", 0, 10)
        self.display.drawText("Diversity (Blue)", 0, 20)
        self.position =0,0
        self.reach_corner = False
        self.up_point =0,0.57
        self.right_point = -0.49,0
        self.down_poin = 0, -0.575
        self.up_reached = False
        self.down_reached = False
        self.right_reached = False
    
    def detect_circles(self, close_threshold=0.75, min_circle_len=2):
        """检测机器人轨迹中的圆圈"""
        points = self.position_history
        circles = []
        start_idx = 0
        accumulated_len = 0.0
        
        for i in range(1, len(points)):
            p0 = np.array(points[i - 1])
            p1 = np.array(points[i])
            step_dist = np.linalg.norm(p1 - p0)
            accumulated_len += step_dist
            
            if np.linalg.norm(p1 - np.array(points[start_idx])) < close_threshold and accumulated_len > min_circle_len:
                circle_points = points[start_idx:i + 1]
                circle_len = np.sum(np.linalg.norm(np.diff(circle_points, axis=0), axis=1))
                circles.append((circle_len, (start_idx, i)))
                start_idx = i
                accumulated_len = 0.0
        
        return circles
    
    def createRandomPopulation(self):
        """创建初始随机种群"""
        if self.num_weights > 0:
            pop_size = (self.num_population, self.num_weights)
            # 使用Xavier初始化，更适合神经网络
            limit = np.sqrt(6.0 / (self.num_weights + 2))  # 假设输出层2个神经元
            self.population = np.random.uniform(low=-limit, high=limit, size=pop_size)
            print(f"Created initial population with Xavier initialization (±{limit:.3f})")
    
    def handle_receiver(self):
        """处理从机器人接收的消息"""
        while self.receiver.getQueueLength() > 0:
            self.receivedData = self.receiver.getString()
            typeMessage = self.receivedData[0:7]
            
            if typeMessage == "weights":
                self.receivedWeights = self.receivedData[9:len(self.receivedData)]
                self.num_weights = int(self.receivedWeights)
            elif typeMessage == "fitness":
                self.receivedFitness = float(self.receivedData[9:len(self.receivedData)])
            
            self.receiver.nextPacket()
    
    def handle_emitter(self):
        """发送消息给机器人"""
        if self.num_weights > 0:
            string_message = "genotype: " + str(self.emitterData)
            string_message = string_message.encode("utf-8")
            self.emitter.send(string_message)
        
        self.emitter.send("current_generation: {}".format(self.current_generation).encode("utf-8"))
        self.emitter.send("num_generations: {}".format(self.num_generations).encode("utf-8"))
        
        v = self.robot_node.getVelocity()
        self.real_speed = (v[0]**2 + v[1]**2 + v[2]**2)**0.5
        self.emitter.send("real_speed: {}".format(self.real_speed).encode("utf-8"))
        
        pos = self.robot_node.getPosition()
        self.position = pos[0], pos[1]
        self.position_history.append([pos[0], pos[1]])
        # up_distance = math.sqrt((self.position[0]-self.up_point[0])**2 + (self.position[1]-self.up_point[1])**2)
        # right_distance = math.sqrt((self.position[0]-self.right_point[0])**2 + (self.position[1]-self.right_point[1])**2)
        # down_distance = math.sqrt((self.position[0]-self.down_poin[0])**2 + (self.position[1]-self.down_poin[1])**2)
        # if up_distance < 0.1:
        #     self.up_reached = True
        #     print("Up distance:", up_distance)
        # if down_distance < 0.1:
        #     self.down_reached = True
        #     print("Down distance:", down_distance)
        # if right_distance < 0.1:
        #     self.right_reached = True
        #     print("Right distance:", right_distance)
        # if self.up_reached and self.down_reached and self.right_reached:
        #     self.reach_corner = True

        self.emitter.send("position: {}".format([pos[0], pos[1], pos[2]]).encode("utf-8"))
    
    def run_seconds(self, seconds):
        """运行指定秒数的仿真"""
        stop = int((seconds * 1000) / self.time_step)
        iterations = 0
        while self.supervisor.step(self.time_step) != -1:
            self.handle_emitter()
            self.handle_receiver()
            if stop == iterations:
                break
            iterations += 1
    
    def evaluate_genotype(self, genotype, generation):
        """评估单个基因型"""
        self.emitterData = str(genotype)
        
        # 重置机器人位置
        INITIAL_TRANS = [0.47, 0.16, 0]
        self.trans_field.setSFVec3f(INITIAL_TRANS)
        INITIAL_ROT = [0, 0, 1, 1.57]
        self.rot_field.setSFRotation(INITIAL_ROT)
        self.robot_node.resetPhysics()
        
        # 运行评估
        self.run_seconds(self.time_experiment)
        
        # 获取适应度
        fitness = self.receivedFitness
        current = (generation, genotype, fitness)
        self.genotypes.append(current)
        
        return fitness
    
    def check_stagnation(self, best_fitness):
        """
        检测进化停滞
        如果连续多代没有改进，触发多样性注入
        """
        improvement = best_fitness - self.last_best_fitness
        
        if improvement < 0.001:  # 改进很小
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = 0
        
        self.last_best_fitness = best_fitness
        
        # 如果停滞超过10代，注入新个体
        if self.stagnation_counter >= 10:
            print("  [Warning] Stagnation detected! Injecting diversity...")
            return True
        
        return False
    
    def inject_diversity(self, current_population):
        """
        注入多样性：替换部分较差个体为随机个体
        """
        num_inject = self.num_population // 5  # 替换20%
        ranked = self.ga.rank_population(current_population)
        
        # 保留前80%，替换后20%
        keep_size = self.num_population - num_inject
        new_population = [ind[0] for ind in ranked[-keep_size:]]
        
        # 生成新的随机个体
        limit = np.sqrt(6.0 / (self.num_weights + 2))
        for _ in range(num_inject):
            random_individual = np.random.uniform(low=-limit, high=limit, size=self.num_weights)
            new_population.append(random_individual)
        
        self.stagnation_counter = 0
        return new_population
    
    def run_optimization(self):
        """运行改进的遗传算法优化"""
        # 等待接收权重数量
        while self.num_weights == 0:
            self.handle_receiver()
            self.createRandomPopulation()
        
        print("=" * 60)
        print("Starting IMPROVED GA Optimization")
        print("=" * 60)
        print(f"Population Size: {self.num_population}")
        print(f"Generations: {self.num_generations}")
        print(f"Elite Size: {self.num_elite}")
        print(f"Initial Crossover Rate: {self.ga.crossover_rate:.2f}")
        print(f"Initial Mutation Rate: {self.ga.mutation_rate:.2f}")
        print(f"Adaptive Parameters: {self.ga.adaptive}")
        print("=" * 60)
        
        for generation in range(self.num_generations):
            print(f"\n{'='*60}")
            print(f"Generation {generation + 1}/{self.num_generations}")
            print(f"{'='*60}")
            
            current_population = []
            self.current_generation = generation
            
            # 评估每个个体
            for population_idx in range(self.num_population):
                self.up_reached = False
                self.right_reached = False
                self.down_reached = False
                self.reach_corner = False
                self.position_history = []
                genotype = self.population[population_idx]
                
                # 评估基础适应度
                fitness = self.evaluate_genotype(genotype, generation)
                
                # # # 圆圈检测奖励
                # circles = self.detect_circles()
                # have_big_circle = False
                # have_middle_circle = False
                # for (length, (start_idx, end_idx)) in circles:
                #     if length >=2.8:
                #         have_big_circle = True
                #         break
                #     elif 2.0 <= length <2.8:
                #         have_middle_circle = True
                #         break
                # if have_big_circle:
                #     fitness += 0.1
                # elif have_middle_circle:
                #     fitness += 0.05
                # else:
                #     fitness -= 0.05
                # print(circles)
                # for (length, (start_idx, end_idx)) in circles:
                #     print(length)
                # for (length, (start_idx, end_idx)) in circles:
                #     circle_quality = length / 4.0
                #     if 0.85 <= circle_quality <= 1.1 and self.reach_corner:
                #         fitness += 0.2
                #         break
                #     elif 0.8 < circle_quality < 1.2 and self.reach_corner:
                #         fitness += 0.1
                #         break
                
                print(f"  Individual {population_idx + 1:2d}: Fitness = {fitness:.4f}")
                current_population.append((genotype, float(fitness)))
            
            # 统计信息
            best = self.ga.get_best_genotype(current_population)
            average = self.ga.get_average_fitness(current_population)
            diversity = self.ga.calculate_diversity(current_population)
            
            self.best_fitness_history.append(best[1])
            self.avg_fitness_history.append(average)
            self.diversity_history.append(diversity)
            
            print(f"\n{'='*60}")
            print(f"Generation {generation + 1} Summary:")
            print(f"  Best Fitness:    {best[1]:.4f}")
            print(f"  Average Fitness: {average:.4f}")
            print(f"  Diversity:       {diversity:.4f}")
            print(f"  Crossover Rate:  {self.ga.crossover_rate:.2f}")
            print(f"  Mutation Rate:   {self.ga.mutation_rate:.2f}")
            print(f"{'='*60}")
            
            # 保存最优个体
            np.save("../supervisorGA_starter/Best.npy", best[0])
            
            # 绘制适应度曲线
            self.plot_fitness(generation, best[1], average, diversity)
            
            # 检测停滞并注入多样性
            if self.check_stagnation(best[1]):
                self.population = self.inject_diversity(current_population)
            elif generation < self.num_generations - 1:
                # 生成新一代
                self.population = self.ga.population_reproduce(
                    current_population, 
                    generation, 
                    self.num_generations
                )
        
        # 最终统计
        print("\n" + "=" * 60)
        print("GA Optimization Complete!")
        print("=" * 60)
        print(f"Final Best Fitness: {self.best_fitness_history[-1]:.4f}")
        print(f"Final Avg Fitness:  {self.avg_fitness_history[-1]:.4f}")
        print(f"Improvement:        {self.best_fitness_history[-1] - self.best_fitness_history[0]:.4f}")
        print("=" * 60)
        
        # 保存训练历史
        np.save("training_history.npy", {
            'best_fitness': self.best_fitness_history,
            'avg_fitness': self.avg_fitness_history,
            'diversity': self.diversity_history
        })
    
    def run_demo(self):
        """运行最优个体演示"""
        genotype = np.load("../supervisorGA_starter/Best.npy")
        self.emitterData = str(genotype)
        
        INITIAL_TRANS = [0.47, 0.16, 0]
        self.trans_field.setSFVec3f(INITIAL_TRANS)
        INITIAL_ROT = [0, 0, 1, 1.57]
        self.rot_field.setSFRotation(INITIAL_ROT)
        self.robot_node.resetPhysics()
        
        self.run_seconds(self.time_experiment)
    
    def draw_scaled_line(self, generation, y1, y2, color=0xff0000):
        """绘制缩放后的线条"""
        XSCALE = int(self.width / self.num_generations)
        YSCALE = 100
        self.display.setColor(color)
        self.display.drawLine(
            (generation - 1) * XSCALE, self.height - int(y1 * YSCALE),
            generation * XSCALE, self.height - int(y2 * YSCALE)
        )
    
    def plot_fitness(self, generation, best_fitness, average_fitness, diversity):
        """绘制适应度和多样性曲线"""
        if generation > 0:
            # 最优适应度 - 红色
            self.draw_scaled_line(generation, self.prev_best_fitness, best_fitness, 0xff0000)
            
            # 平均适应度 - 绿色
            self.draw_scaled_line(generation, self.prev_average_fitness, average_fitness, 0x00ff00)
            
            # 多样性 - 蓝色（缩放到可见范围）
            scaled_diversity = diversity * 10  # 放大10倍以便可视化
            self.draw_scaled_line(generation, self.prev_diversity * 10, scaled_diversity, 0x0000ff)
        
        self.prev_best_fitness = best_fitness
        self.prev_average_fitness = average_fitness
        self.prev_diversity = diversity


if __name__ == "__main__":
    gaModel = ImprovedSupervisorGA()
    
    keyboard = Keyboard()
    keyboard.enable(50)
    
    print("\n" + "=" * 60)
    print("IMPROVED GA Controller for E-puck Robot")
    print("=" * 60)
    print("Commands:")
    print("  [S] - Start optimization with improved GA")
    print("  [R] - Run best individual demo")
    print("=" * 60)
    
    while gaModel.supervisor.step(gaModel.time_step) != -1:
        resp = keyboard.getKey()
        if resp == 83 or resp == 115:  # S or s
            gaModel.run_optimization()
            print("\n[S] Start optimization | [R] Run demo")
        elif resp == 82 or resp == 114:  # R or r
            gaModel.run_demo()
            print("\n[S] Start optimization | [R] Run demo")
