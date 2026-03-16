import random
from utils.model_zoo import ModelZoo


class GeneticAlgorithmOptimizer:
    def __init__(self, model_zoo: ModelZoo, target_model: str, rename_fn, pop_size=10, max_generations=10):
        self.model_zoo = model_zoo
        self.target_model = target_model
        self.rename_fn = rename_fn
        self.pop_size = pop_size
        self.max_generations = max_generations

    def run(self, code, original_pred, target_vars, subs_pool):
        population = [{var: var for var in target_vars}]
        for _ in range(self.pop_size - 1):
            population.append({v: random.choice(subs_pool.get(v, [v]) + [v]) for v in target_vars})

        best_code = code
        best_fitness = -1

        for gen in range(self.max_generations):
            evaluated = []
            for ind in population:
                rename_map = {k: v for k, v in ind.items() if k != v}
                mutated_code = self.rename_fn(code, rename_map)
                probs, pred = self.model_zoo.predict(mutated_code, self.target_model)

                fitness = 1.0 - probs[original_pred]
                evaluated.append((ind, fitness, pred, mutated_code, probs))

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_code = mutated_code

                if pred != original_pred:
                    print(f"    [+] Target Model ({self.target_model}) Attack Succeeded at Gen {gen}!")
                    return True, mutated_code, probs, pred

            evaluated.sort(key=lambda x: x[1], reverse=True)
            elites = [x[0] for x in evaluated[:max(2, self.pop_size // 2)]]

            new_pop = elites.copy()
            while len(new_pop) < self.pop_size:
                p1, p2 = random.sample(elites, 2)
                child = {v: p1[v] if random.random() > 0.5 else p2[v] for v in target_vars}
                for v in child:
                    if random.random() < 0.2:
                        child[v] = random.choice(subs_pool.get(v, [v]) + [v])
                new_pop.append(child)
            population = new_pop

        final_probs, final_pred = self.model_zoo.predict(best_code, self.target_model)
        return False, best_code, final_probs, final_pred
