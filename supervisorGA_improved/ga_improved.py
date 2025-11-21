import numpy as np
import random

class ImprovedGA:
    """
    改进的遗传算法实现
    主要改进：
    1. 自适应交叉率和变异率
    2. 多种交叉策略
    3. 改进的选择机制
    4. 多样性保护
    5. 自适应变异幅度
    """
    
    def __init__(self, 
                 population_size=60,
                 elite_size=6,
                 initial_crossover_rate=0.8,
                 initial_mutation_rate=0.15,
                 tournament_size=7,
                 adaptive=True):
        """
        初始化GA参数
        
        参数:
            population_size: 种群大小
            elite_size: 精英个体数量
            initial_crossover_rate: 初始交叉率 (0.8 = 80%)
            initial_mutation_rate: 初始变异率 (0.15 = 15%)
            tournament_size: 锦标赛选择的个体数
            adaptive: 是否使用自适应参数
        """
        self.population_size = population_size
        self.elite_size = elite_size
        self.crossover_rate = initial_crossover_rate
        self.mutation_rate = initial_mutation_rate
        self.tournament_size = tournament_size
        self.adaptive = adaptive
        
        # 记录历史最优适应度，用于自适应调整
        self.best_fitness_history = []
        self.diversity_history = []
        
    def adapt_parameters(self, generation, max_generations, current_diversity):
        """
        自适应调整交叉率和变异率
        
        策略：
        - 早期：高交叉率(0.8-0.9)，低变异率(0.1-0.15) - 快速探索
        - 中期：中等交叉率(0.7-0.8)，中等变异率(0.15-0.2) - 平衡探索和利用
        - 后期：低交叉率(0.6-0.7)，高变异率(0.2-0.3) - 精细搜索
        
        同时根据种群多样性动态调整
        """
        if not self.adaptive:
            return
        
        progress = generation / max_generations
        
        # 基于进化进度的基础调整
        if progress < 0.3:  # 早期
            base_crossover = 0.85
            base_mutation = 0.12
        elif progress < 0.7:  # 中期
            base_crossover = 0.75
            base_mutation = 0.18
        else:  # 后期
            base_crossover = 0.65
            base_mutation = 0.25
        
        # 基于多样性的微调
        if current_diversity < 0.1:  # 多样性过低
            self.mutation_rate = min(base_mutation + 0.1, 0.4)
            self.crossover_rate = max(base_crossover - 0.1, 0.5)
        elif current_diversity > 0.5:  # 多样性过高
            self.mutation_rate = max(base_mutation - 0.05, 0.05)
            self.crossover_rate = min(base_crossover + 0.1, 0.95)
        else:
            self.mutation_rate = base_mutation
            self.crossover_rate = base_crossover
        
        print(f"  [Adaptive] Crossover Rate: {self.crossover_rate:.2f}, Mutation Rate: {self.mutation_rate:.2f}")
    
    def calculate_diversity(self, population):
        """
        计算种群多样性
        使用基因型之间的平均欧氏距离
        """
        if len(population) < 2:
            return 0.0
        
        genotypes = [ind[0] for ind in population]
        distances = []
        
        for i in range(len(genotypes)):
            for j in range(i+1, len(genotypes)):
                dist = np.linalg.norm(genotypes[i] - genotypes[j])
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def rank_population(self, genotypes):
        """按适应度排序（从低到高）"""
        genotypes.sort(key=lambda item: item[1])
        return genotypes
    
    def get_best_genotype(self, genotypes):
        """获取最优个体"""
        return self.rank_population(genotypes)[-1]
    
    def get_average_fitness(self, genotypes):
        """计算平均适应度"""
        return np.mean([g[1] for g in genotypes])
    
    def tournament_selection(self, population):
        """
        改进的锦标赛选择
        
        改进点：
        1. 增加锦标赛规模到7个
        2. 基于适应度概率选择，而非完全随机
        3. 引入选择压力参数
        """
        # 根据适应度计算选择概率
        fitnesses = np.array([ind[1] for ind in population])
        # 避免负适应度问题
        fitnesses = fitnesses - fitnesses.min() + 1e-6
        probabilities = fitnesses / fitnesses.sum()
        
        # 按概率选择锦标赛参与者
        indices = np.random.choice(len(population), 
                                   size=self.tournament_size, 
                                   replace=False,
                                   p=probabilities)
        
        tournament = [population[i] for i in indices]
        
        # 选择锦标赛中的最优个体
        winner = max(tournament, key=lambda x: x[1])
        return winner
    
    def roulette_wheel_selection(self, population):
        """
        轮盘赌选择（作为备选方案）
        """
        fitnesses = np.array([ind[1] for ind in population])
        # 处理负适应度
        fitnesses = fitnesses - fitnesses.min() + 1e-6
        probabilities = fitnesses / fitnesses.sum()
        
        idx = np.random.choice(len(population), p=probabilities)
        return population[idx]
    
    def uniform_crossover(self, parent1, parent2):
        """
        均匀交叉
        每个基因独立地从两个父代中选择
        """
        child = []
        for i in range(len(parent1[0])):
            if random.random() < 0.5:
                child.append(parent1[0][i])
            else:
                child.append(parent2[0][i])
        return child
    
    def two_point_crossover(self, parent1, parent2):
        """
        两点交叉
        随机选择两个交叉点
        """
        length = len(parent1[0])
        point1 = random.randint(1, length - 2)
        point2 = random.randint(point1 + 1, length - 1)
        
        child = []
        for i in range(length):
            if point1 <= i < point2:
                child.append(parent2[0][i])
            else:
                child.append(parent1[0][i])
        return child
    
    def arithmetic_crossover(self, parent1, parent2, alpha=0.5):
        """
        算术交叉（适合连续值优化）
        child = alpha * parent1 + (1 - alpha) * parent2
        """
        alpha = random.uniform(0.3, 0.7)  # 随机权重
        child = []
        for i in range(len(parent1[0])):
            gene = alpha * parent1[0][i] + (1 - alpha) * parent2[0][i]
            child.append(gene)
        return child
    
    def blx_alpha_crossover(self, parent1, parent2, alpha=0.5):
        """
        BLX-α交叉（Blend Crossover）
        在父代基因值的扩展范围内随机选择
        """
        child = []
        for i in range(len(parent1[0])):
            gene1, gene2 = parent1[0][i], parent2[0][i]
            min_val, max_val = min(gene1, gene2), max(gene1, gene2)
            range_val = max_val - min_val
            
            # 扩展范围
            lower = min_val - alpha * range_val
            upper = max_val + alpha * range_val
            
            # 限制在[-1, 1]范围内
            lower = max(lower, -1.0)
            upper = min(upper, 1.0)
            
            gene = random.uniform(lower, upper)
            child.append(gene)
        return child
    
    def adaptive_crossover(self, parent1, parent2, generation, max_generations):
        """
        自适应交叉策略
        根据进化阶段选择不同的交叉方法
        """
        progress = generation / max_generations
        rand = random.random()
        
        if progress < 0.3:  # 早期：使用BLX交叉，探索更广
            return self.blx_alpha_crossover(parent1, parent2, alpha=0.5)
        elif progress < 0.7:  # 中期：混合使用
            if rand < 0.5:
                return self.arithmetic_crossover(parent1, parent2)
            else:
                return self.two_point_crossover(parent1, parent2)
        else:  # 后期：使用算术交叉，精细调整
            return self.arithmetic_crossover(parent1, parent2)
    
    def gaussian_mutation(self, child, generation, max_generations):
        """
        高斯变异（改进版）
        
        改进点：
        1. 自适应变异幅度（随代数递减）
        2. 使用高斯分布而非均匀分布
        3. 变异幅度与基因当前值相关
        """
        after_mutation = []
        progress = generation / max_generations
        
        # 自适应变异幅度：从0.5递减到0.1
        mutation_strength = 0.5 * (1 - progress) + 0.1
        
        for gene in child:
            if random.random() < self.mutation_rate:
                # 高斯变异：均值为0，标准差为mutation_strength
                noise = np.random.normal(0, mutation_strength)
                new_gene = gene + noise
                
                # 限制在[-1, 1]范围内
                new_gene = np.clip(new_gene, -1.0, 1.0)
                after_mutation.append(new_gene)
            else:
                after_mutation.append(gene)
        
        return after_mutation
    
    def non_uniform_mutation(self, child, generation, max_generations):
        """
        非均匀变异
        变异幅度随代数非线性递减
        """
        after_mutation = []
        b = 5  # 形状参数
        
        for gene in child:
            if random.random() < self.mutation_rate:
                r = random.random()
                if random.random() < 0.5:
                    delta = (1.0 - gene) * (1 - r ** ((1 - generation/max_generations) ** b))
                else:
                    delta = (gene + 1.0) * (1 - r ** ((1 - generation/max_generations) ** b))
                    delta = -delta
                
                new_gene = gene + delta
                new_gene = np.clip(new_gene, -1.0, 1.0)
                after_mutation.append(new_gene)
            else:
                after_mutation.append(gene)
        
        return after_mutation
    
    def population_reproduce(self, genotypes, generation, max_generations):
        """
        改进的种群繁殖函数
        
        改进点：
        1. 自适应参数调整
        2. 多样性保护
        3. 改进的选择和交叉策略
        """
        # 计算当前多样性
        diversity = self.calculate_diversity(genotypes)
        self.diversity_history.append(diversity)
        
        # 自适应调整参数
        self.adapt_parameters(generation, max_generations, diversity)
        
        # 排序种群
        ranked_genotypes = self.rank_population(genotypes)
        
        new_population = []
        
        # 1. 精英保留
        for i in range(self.elite_size):
            new_population.append(ranked_genotypes[-(i+1)][0])
        
        # 2. 生成剩余个体
        while len(new_population) < self.population_size:
            # 选择父代
            parent1 = self.tournament_selection(genotypes)
            parent2 = self.tournament_selection(genotypes)
            
            # 交叉
            if random.random() < self.crossover_rate:
                child = self.adaptive_crossover(parent1, parent2, generation, max_generations)
            else:
                # 不交叉则随机选择一个父代
                child = list(parent1[0]) if random.random() < 0.5 else list(parent2[0])
            
            # 变异
            offspring = self.gaussian_mutation(child, generation, max_generations)
            
            new_population.append(np.array(offspring))
        
        return new_population


# 兼容旧接口的包装函数
def population_reproduce(genotypes, elite):
    """
    兼容原有接口的包装函数
    使用改进的GA算法
    """
    ga = ImprovedGA(
        population_size=len(genotypes),
        elite_size=elite,
        initial_crossover_rate=0.8,
        initial_mutation_rate=0.15,
        adaptive=True
    )
    
    # 假设在中期阶段
    generation = 50
    max_generations = 120
    
    return ga.population_reproduce(genotypes, generation, max_generations)


def rankPopulation(genotypes):
    """兼容原有接口"""
    genotypes.sort(key=lambda item: item[1])
    return genotypes


def getBestGenotype(genotypes):
    """兼容原有接口"""
    return rankPopulation(genotypes)[-1]


def getAverageGenotype(genotypes):
    """兼容原有接口"""
    return np.mean([g[1] for g in genotypes])
