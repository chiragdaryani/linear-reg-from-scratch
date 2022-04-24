# linear-reg-from-scratch

In this project, we will implement gradient descent optimization for linear regression on the Boston house price prediction dataset.

We use the mean square error as our loss to train our model and the measure of success (risk/error) ,is the mean of the absolute difference between the predicted price and true price. This reflects how much money we would lose for a bad prediction. The lower, the better.

We implement the train-validation-test framework, where we train the model by mini-batch gradient descent, and validate model performance after each epoch. After reaching the maximum number of iterations, we pick the epoch that yields the best validation performance (the lowest risk), and test the model on the test set.
