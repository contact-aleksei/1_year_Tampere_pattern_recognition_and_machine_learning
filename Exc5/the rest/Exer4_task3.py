import numpy as np
import random
import matplotlib.pyplot as plt
import pickle

def log_loss(w, X, y):
    """ 
    Computes the log-loss function at w. The 
    computation uses the data in X with
    corresponding labels in y. 
    """
    
    L = 0 # Accumulate loss terms here.
        
    # Process each sample in X:
    for n in range(X.shape[0]):
        L += np.log(1 + np.exp(y[n] * np.dot(w, X[n])))
    
    return L
    
def grad(w, X, y):
    """ 
    Computes the gradient of the log-loss function
    at w. The computation uses the data in X with
    corresponding labels in y. 
    """
        
    G = 0 # Accumulate gradient here.
    
    # Process each sample in X:
    for n in range(X.shape[0]):
    
        numerator = np.exp(-y[n] * np.dot(w, X[n])) * (-y[n]) * X[n]
        denominator = 1 + np.exp(-y[n] * np.dot(w, X[n]))
        
        G += numerator / denominator
    
    return G
    
if __name__ == "__main__":
        
    # Add your code here:
        
    # 1) Load X and y from file.


    X=np.loadtxt('X.csv', delimiter=',')
    y=np.loadtxt('y.csv')
    
    # 2) Initialize w at random: w = np.random.randn(2)
    
    # w = np.random.randn(2)
    w=np.array([1,-1]) 
   
    # 3) Set step_size to a small positive value.
    
    step_size=0.001    
    
    # 4) Initialize empty lists for storing the path and
    # accuracies: W = []; accuracies = []
    
    W = []
    accuracies = []
    
    for iteration in range(100):

        # 5) Apply the gradient descent rule.
        
        G=grad(w, X, y)
        w=w-step_size*G  
        # 6) Print the current state.
        print ("Iteration %d: w = %s (log-loss = %.2f)" % \
              (iteration, str(w), log_loss(w, X, y)))
        
        # 7) Compute the accuracy:
        scores = np.dot(X, w)
		# Predict class 1 probability
        y_prob = 1 / (1 + np.exp(-np.dot(X, w)))
		# Threshold at 0.5
            
        #y_pred=( 1 if y_prob [index] > 0.5 else -1)
        y_pred=( y_prob > 0.5).astype(int)
        y_pred=2*y_pred-1
        accuracy = np.mean(y_pred == y)
        accuracies.append(accuracy)
        
        W.append(w)
    
    # 8) Below is a template for plotting. Feel free to 
    # rewrite if you prefer different style.
    
    W = np.array(W)
    
    plt.figure(figsize = [5,5])
    plt.subplot(211)
    plt.plot(W[:,0], W[:,1], 'ro-')
    plt.xlabel('w$_0$')
    plt.ylabel('w$_1$')
    plt.title('Optimization path')
    
    plt.subplot(212)
    plt.plot(100.0 * np.array(accuracies), linewidth = 2)
    plt.ylabel('Accuracy / %')
    plt.xlabel('Iteration')
    plt.tight_layout()
    plt.savefig("log_loss_minimization.pdf", bbox_inches = "tight")
    