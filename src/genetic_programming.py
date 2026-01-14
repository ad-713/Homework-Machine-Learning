import operator
import numpy as np
from deap import base, creator, tools, gp
from sklearn.metrics import accuracy_score
from modAL.uncertainty import uncertainty_sampling

class GPClassifierWrapper:
    """
    Wrapper to make a DEAP GP individual compatible with modAL.
    """
    def __init__(self, individual, toolbox):
        self.individual = individual
        self.toolbox = toolbox
        self.func = toolbox.compile(expr=individual)
        self.classes_ = [0, 1]
    
    def predict(self, X):
        predictions = []
        for p in X:
            try:
                val = self.func(*p)
                predictions.append(1 if val > 0 else 0)
            except (OverflowError, ZeroDivisionError, ValueError):
                predictions.append(0)
        return np.array(predictions)
    
    def predict_proba(self, X):
        """
        Calculates probabilities using a sigmoid function on the GP tree output.
        """
        def sigmoid(x):
            # Robust sigmoid
            try:
                if x >= 0:
                    z = np.exp(-x)
                    return 1 / (1 + z)
                else:
                    z = np.exp(x)
                    return z / (1 + z)
            except OverflowError:
                return 1.0 if x > 0 else 0.0
        
        probs = []
        for p in X:
            try:
                val = self.func(*p)
                p1 = sigmoid(val)
                probs.append([1 - p1, p1])
            except (OverflowError, ZeroDivisionError, ValueError):
                probs.append([0.5, 0.5]) # Maximum uncertainty on error
        return np.array(probs)

def setup_gp(n_features):
    """
    Sets up the DEAP GP environment.
    """
    # Define primitives: addition, subtraction, multiplication
    pset = gp.PrimitiveSet("MAIN", n_features)
    pset.addPrimitive(operator.add, 2)
    pset.addPrimitive(operator.sub, 2)
    pset.addPrimitive(operator.mul, 2)
    
    # Define fitness and individual
    if not hasattr(creator, "FitnessMax"):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "Individual"):
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    
    return pset, toolbox

def eval_classifier(individual, points, labels, toolbox):
    """
    Evaluates an individual (GP tree) on a set of points.
    Using accuracy as the fitness metric.
    """
    func = toolbox.compile(expr=individual)
    try:
        # points might be a numpy array or list
        predictions = [1 if func(*p) > 0 else 0 for p in points]
        acc = accuracy_score(labels, predictions)
        return (acc,)
    except (OverflowError, ZeroDivisionError, ValueError):
        return (0.0,)

def run_active_learning_ga(X_train, y_train, X_pool, y_pool, pset, toolbox, n_gen=10, pop_size=50, k=3, n_instances=10):
    """
    Runs the Genetic Algorithm loop with Active Learning integration.
    """
    # Register evaluation function with initial data
    toolbox.register("evaluate", eval_classifier, points=X_train, labels=y_train, toolbox=toolbox)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    
    # Bloat control
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    
    # Initialize population
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + stats.fields

    curr_X_train = np.copy(X_train)
    curr_y_train = np.copy(y_train)
    curr_X_pool = np.copy(X_pool)
    curr_y_pool = np.copy(y_pool)

    # Initial evaluation
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit
    
    hof.update(pop)
    record = stats.compile(pop)
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    print(logbook.stream)

    for gen in range(1, n_gen + 1):
        # AL check: Every k generations
        if gen % k == 0 and len(curr_X_pool) >= n_instances:
            print(f"\n--- Active Learning Step at Generation {gen} ---")
            best_ind = hof[0]
            classifier = GPClassifierWrapper(best_ind, toolbox)
            
            # Query instances
            query_idx, _ = uncertainty_sampling(classifier, curr_X_pool, n_instances=n_instances)
            
            # Update data
            curr_X_train = np.vstack([curr_X_train, curr_X_pool[query_idx]])
            curr_y_train = np.concatenate([curr_y_train, curr_y_pool[query_idx]])
            
            curr_X_pool = np.delete(curr_X_pool, query_idx, axis=0)
            curr_y_pool = np.delete(curr_y_pool, query_idx, axis=0)
            
            print(f"Added {n_instances} samples. New training set size: {len(curr_X_train)}")
            
            # Update evaluate function
            toolbox.register("evaluate", eval_classifier, points=curr_X_train, labels=curr_y_train, toolbox=toolbox)
            
            # Re-evaluate all individuals
            for ind in pop:
                del ind.fitness.values
            invalid_ind = pop
        else:
            # Traditional GA step
            offspring = tools.selTournament(pop, len(pop), tournsize=3)
            offspring = [toolbox.clone(ind) for ind in offspring]
            
            from deap import algorithms
            offspring = algorithms.varAnd(offspring, toolbox, cxpb=0.7, mutpb=0.2)
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        if not (gen % k == 0 and len(curr_X_pool) >= n_instances):
            pop[:] = offspring

        hof.update(pop)
        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        print(logbook.stream)

    return pop, logbook, hof

def run_ga(X_train, y_train, pset, toolbox, n_gen=10, pop_size=50):
    """
    Backward compatibility wrapper.
    """
    return run_active_learning_ga(X_train, y_train, np.empty((0, X_train.shape[1])), np.empty((0,)), 
                                  pset, toolbox, n_gen, pop_size, k=n_gen+1)
