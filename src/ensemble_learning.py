import numpy as np
from src.genetic_programming import GPClassifierWrapper

class GPEnsembleClassifier:
    """
    An ensemble of Genetic Programming individuals.
    Supports hard, soft, and weighted voting.
    """
    def __init__(self, individuals, toolbox):
        """
        Initialize the ensemble.
        
        Args:
            individuals (list): A list of DEAP GP individuals.
            toolbox: The DEAP toolbox for compiling individuals.
        """
        self.individuals = individuals
        self.toolbox = toolbox
        self.classifiers = [GPClassifierWrapper(ind, toolbox) for ind in individuals]
        self.weights = [ind.fitness.values[0] for ind in individuals]
        
    def predict_proba(self, X):
        """
        Computes the average probability of class 1 across all individuals (Soft Voting).
        
        Args:
            X: Input data features.
            
        Returns:
            np.array: Averaged class probabilities.
        """
        all_probs = np.array([clf.predict_proba(X) for clf in self.classifiers])
        # all_probs shape: (n_classifiers, n_samples, 2)
        avg_probs = np.mean(all_probs, axis=0)
        return avg_probs

    def predict(self, X, voting='hard'):
        """
        Predict class labels for X.
        
        Args:
            X: Input data features.
            voting (str): 'hard' for majority vote, 'soft' for probability average,
                         'weighted' for fitness-weighted majority vote.
                         
        Returns:
            np.array: Predicted class labels.
        """
        if voting == 'soft':
            probs = self.predict_proba(X)
            return np.argmax(probs, axis=1)
        
        elif voting == 'hard':
            all_preds = np.array([clf.predict(X) for clf in self.classifiers])
            # all_preds shape: (n_classifiers, n_samples)
            # Majority vote
            summed_votes = np.sum(all_preds, axis=0)
            return (summed_votes > (len(self.classifiers) / 2)).astype(int)
            
        elif voting == 'weighted':
            all_preds = np.array([clf.predict(X) for clf in self.classifiers])
            # all_preds shape: (n_classifiers, n_samples)
            # Weighted sum of votes
            # Normalize weights to sum to 1
            w = np.array(self.weights)
            if np.sum(w) == 0:
                # Fallback to equal weights if all fitnesses are 0
                w = np.ones(len(w)) / len(w)
            else:
                w = w / np.sum(w)
            
            # Reshape w for broadcasting: (n_classifiers, 1)
            w = w.reshape(-1, 1)
            weighted_votes = np.sum(all_preds * w, axis=0)
            return (weighted_votes > 0.5).astype(int)
            
        else:
            raise ValueError(f"Unknown voting method: {voting}")

    def score(self, X, y, voting='hard'):
        """
        Calculate accuracy on the given dataset.
        """
        from sklearn.metrics import accuracy_score
        preds = self.predict(X, voting=voting)
        return accuracy_score(y, preds)
