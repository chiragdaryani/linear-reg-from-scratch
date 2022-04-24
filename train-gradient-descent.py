#!/usr/bin/env python3

import sklearn.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt


def predict(X, w, y=None, normalize=True):
    # X_new: Nsample x (d+1)
    # w: (d+1) x 1
    # y_new: Nsample


    # Get prediction
    y_hat = X.dot(w)
    #print(y_hat)

    if normalize==False:
        # Perform de-normalization on true label and predicted label
        y = y * std_y + mean_y
        y_hat = y_hat *  std_y + mean_y



    #Calculate mean squared error
    loss = (np.square(np.subtract(y_hat, y)).mean())*0.5
    #print(loss)

    
    #Calculate mean absolute error
    risk = np.absolute(np.subtract(y_hat, y)).mean()
    #print(risk)
    
    
    return y_hat, loss, risk


def train(X_train, y_train, X_val, y_val):
    N_train = X_train.shape[0]
    N_val = X_val.shape[0]

    # initialization
    w = np.zeros([X_train.shape[1], 1])
    # w: (d+1)x1

    losses_train = []
    risks_val = []

    w_best = None
    risk_best = 10000
    epoch_best = 0

    for epoch in range(MaxIter):

        loss_this_epoch = 0

        for b in range(int(np.ceil(N_train/batch_size))):


            X_batch = X_train[b*batch_size: (b+1)*batch_size]
            y_batch = y_train[b*batch_size: (b+1)*batch_size]

            y_hat_batch, loss_batch, _ = predict(X_batch, w, y_batch)
            loss_this_epoch += loss_batch
            
            
            # Mini-batch gradient descent
            
            #Update value of weights using learning rate and derivative of loss

            #derivative of Loss over batch = (1/M)*[(X.T)(pred-actual)]
            gradient = ((1/batch_size)*((X_batch.T).dot(y_hat_batch-y_batch)))
    
            #update weight
            w = w - alpha*gradient

        # monitor model behavior after each epoch
        # 1. Compute the training loss by averaging loss_this_epoch
        avg_loss_for_epoch =  loss_this_epoch / (N_train/batch_size) #divide by no of batches
        #print(avg_loss_for_epoch)
        # append to list of training losses over all epochs
        losses_train.append(avg_loss_for_epoch)
        
        # 2. Perform validation on the validation set by the risk
        y_hat_val, loss_val, risk_for_val = predict(X_val, w, y_val, normalize=False)
        #print(risk_for_val)
        risks_val.append(risk_for_val)  

        # 3. Keep track of the best validation epoch, risk, and the weights

        # We have to find whether this epoch score has improved the performance or not
        # If current epoch risk is LESS THAN MINIMUM of previous iterations, we update the minimum to current epoch values
        if(risk_for_val<=np.min(risks_val)):
            #print("True")
            w_best = w
            risk_best = risk_for_val
            epoch_best = epoch   
        
    # Best epoch, risk and w after all epochs completed
    #print(epoch_best)
    #print(risk_best)
    #print(w_best)

    # Return some variables as needed
    return epoch_best, risk_best, w_best, losses_train, risks_val


############################
# Main code starts here
############################
# Load data. This is the only allowed API call from sklearn
X, y = datasets.load_boston(return_X_y=True)
y = y.reshape([-1, 1])
# X: sample x dimension
# y: sample x 1

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)







# Augment feature
X_ = np.concatenate((np.ones([X.shape[0], 1]), X), axis=1)
# X_: Nsample x (d+1)





# normalize features:
mean_y = np.mean(y)
std_y = np.std(y)

y = (y - np.mean(y)) / np.std(y)

# print(X.shape, y.shape) # It's always helpful to print the shape of a variable


# Randomly shuffle the data
np.random.seed(314)
np.random.shuffle(X_)
np.random.seed(314)
np.random.shuffle(y)

X_train = X_[:300]
y_train = y[:300]

X_val = X_[300:400]
y_val = y[300:400]

X_test = X_[400:]
y_test = y[400:]

#####################
# setting

alpha = 0.001      # learning rate
batch_size = 10    # batch size
MaxIter = 100        # Maximum iteration
decay = 0.0          # weight decay



# Get best validation performance parameters
epoch_best, risk_best, w_best, losses_train, risks_val = train(X_train, y_train, X_val, y_val)

#print(epoch_best)
#print(risk_best)
#print(w_best)



# Perform test by the weights yielding the best validation performance

y_hat_test, loss_test, risk_test = predict(X_test, w_best, y_test, normalize=False)
#print(y_hat_test)
#print(loss_test)
#print(risk_test)




# Report numbers and draw plots as required.

print("\nThe number of epoch that yields the best validation performance: ", epoch_best) 
print("The validation performance (risk) in that epoch: ", risk_best )
print("The test performance (risk) in that epoch: ",risk_test) 

# Visualize loss history
epochs = list(range(MaxIter))



plt.figure()

plt.plot(epochs, losses_train, 'b-')
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.savefig('Q2.a) Training Loss.png')


plt.figure()

plt.plot(epochs, risks_val, 'r--')
plt.xlabel('Epoch')
plt.ylabel('Validation Risk')
plt.savefig('Q2.a) Validation Risk.png')



