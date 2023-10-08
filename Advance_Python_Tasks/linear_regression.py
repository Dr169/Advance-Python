import numpy as np
import random
import functools
import matplotlib.pyplot as plt
import tqdm


# Generate a numpy array as instances.
instances = np.array(random.sample(range(1, 1000001), 1000000))
# Multiply the instances by the weight to generate the Y's.
y = instances * 2

# Primery weight
weight = 0.1
# Number of iterations.
epochs = 35
# Coefficient for updating the weights.
learning_rate = 0.000000000001


def loss_function(predicted_y, y):
    """
    Compute the cost of function with MSE approach and return mean of them.
    
    Parameters:
    y (np array): Array of acctual valus.
    predicted_y (np array): Array of predicted valus.
    
    Returns:
    mean_of_loss (float): Avrage of loss Array.
    
    """
    # Sum of array loss's
    loss = functools.reduce(lambda x, y: x + y, ((predicted_y - y) ** 2))
    # Avrage of loss Array
    mean_of_loss = np.mean(loss)
    
    return mean_of_loss


def gradient_descent(weight, learning_rate, predicted_y, y, instances):
    """
    This function performs the term weight.
    
    Parameters:
    weight (float): Is the coefficient of instances.
    lernin_rate (float): Coefficient for updating the weights.
    y (np array): Array of acctual valus.
    predicted_y (np array): Array of predicted valus.
    instances (np array): Array of instances.
    
    Returns:
    weight (float): Updated weight.
    
    """
    
    weight = weight - (learning_rate * (np.mean((2 * (predicted_y - y) * instances))))
    
    return weight


def plot_loss_prediction(epoch, weight, instances, y, learning_rate):
    """
    This function updating the term weight.
    
    Parameters:
    epoch (int): Number of iterations.
    weight (float): Coefficient of instances.
    instances (np array): Array of instances.
    y (np array): Array of acctual valus.
    lerning_rate (float): Coefficient for updating the weights.
    
    """
    
    # List of avrage loss arrays
    mean_of_loss_func = []
    
    for epoch in tqdm.tqdm(range(epochs)):
        # Array of predicted valus.
        predicted_y = weight * instances
        
        mean_of_loss_func.append(loss_function(predicted_y,y))
        
        weight = gradient_descent(weight,learning_rate,predicted_y,y,instances)
        

    print("\nweight : {}".format(weight))
    print("last loss : {}".format(mean_of_loss_func[-1]))
    plt.plot(range(1,epochs+1), mean_of_loss_func, marker = 'o')
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.title("")
    plt.show()
    
    
plot_loss_prediction(epochs, weight, instances, y, learning_rate)