import numpy as np

class LogisticRegressionGD:
    """
     This is the gradient descent for the logistic regression classification.
     input:
          learning_rate=0.01, epochs=1000 ;
           
     output:
          fit the and decide weight and Bias.  
    
    """
    def __init__(self, learning_rate=0.01, epochs=1000,threshold = None):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.threshold = threshold if threshold is not None else 0.5
        self.loss_history = []

    def _sigmoid(self, z):
        # Clip z to prevent overflow errors in exp
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        m, n = X.shape
        # 1. Initialize weights and bias
        self.weights = np.zeros((n, 1))
        self.bias = 0
        
        for i in range(self.epochs):
            # 2. Forward Pass (Linear Sum -> Sigmoid)
            z = np.dot(X, self.weights) + self.bias
            a = self._sigmoid(z)
            
            # 3. Calculate Loss (Binary Cross Entropy)
            # Add epsilon to prevent log(0)
            epsilon = 1e-15
            loss = -(1/m) * np.sum(y * np.log(a + epsilon) + (1-y) * np.log(1 - a + epsilon))
            self.loss_history.append(loss)
            
            # 4. Backward Pass (Calculate Gradients)
            dz = a - y
            dw = (1/m) * np.dot(X.T, dz)
            db = (1/m) * np.sum(dz)
            
            # 5. Update Parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
            if i % (self.epochs // 10) == 0:
                print(f"Epoch {i}: Loss {loss:.4f}")

    def predict_proba(self, X):
        z = np.dot(X, self.weights) + self.bias
        return self._sigmoid(z)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return (probabilities >= self.threshold).astype(int)