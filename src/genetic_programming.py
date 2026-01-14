import operator
import numpy as np
from deap import base, creator, tools, gp
from sklearn.metrics import accuracy_score, f1_score

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
    # We want to maximize accuracy/f1, so weight is positive
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
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
    # Transform the tree expression into a callable function
    func = toolbox.compile(expr=individual)
    
    # Predict for each point
    # GP tree outputs a float; we threshold at 0 to get 0 or 1
    # points is expected to be a 2D numpy array or list of lists
    try:
        # We use a threshold of 0.0: positive -> 1, non-positive -> 0
        predictions = [1 if func(*p) > 0 else 0 for p in points]
        acc = accuracy_score(labels, predictions)
        return (acc,)
    except (OverflowError, ZeroDivisionError, ValueError):
        # Return 0 fitness for failed individuals (e.g. overflow in multiplication)
        return (0.0,)

def run_ga(X_train, y_train, pset, toolbox, n_gen=10, pop_size=50):
    """
    Runs the Genetic Algorithm loop.
    """
    # Register evaluation function with data
    toolbox.register("evaluate", eval_classifier, points=X_train, labels=y_train, toolbox=toolbox)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
    
    # Decoration to limit the height of trees (bloat control)
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
    
    # Initialize population
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    
    # Statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    
    # Run GA
    from deap import algorithms
    pop, log = algorithms.eaSimple(pop, toolbox, 0.7, 0.2, n_gen, stats=stats, halloffame=hof, verbose=True)
    
    return pop, log, hof
