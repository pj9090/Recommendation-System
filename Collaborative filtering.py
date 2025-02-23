import numpy as np 
import pandas as pd
import scipy.io
import matplotlib.pyplot as plt

def cost_func(R,Y,theta,X,Lambda):
    predictions=np.dot(X,theta.T)
    error=predictions-Y
    j=1/2*np.sum((error**2)*R) # x by R because we just want were user has rated
    j_reg=j+Lambda/2*np.sum(theta**2)+Lambda/2*np.sum(X**2)
    x_grad=np.dot(error*R,theta)
    theta_grad=np.dot((error*R).T,X)
    reg_x_grad=x_grad + Lambda*X
    reg_theta_grad=theta_grad + Lambda*theta
    return j,x_grad,theta_grad,j_reg,reg_x_grad,reg_theta_grad

def normalization(Y, R):
    m,n = Y.shape[0], Y.shape[1]
    Ymean = np.zeros((m,1))
    Ynorm = np.zeros((m,n))
    for i in range(m):
        Ymean[i] = np.sum(Y[i,:])/np.count_nonzero(R[i,:])
        Ynorm[i,R[i,:]==1] = Y[i,R[i,:]==1] - Ymean[i]
    return Ynorm, Ymean
    
def gradient_descent(X,theta,Y,R,iterations,alpha,Lambda):
    j_history=[]
    for i in range(iterations):
        print ('iteration '+str(i))
        cost,x_grad,theta_grad=cost_func(R,Y,theta,X,Lambda)[3:]
        X=X-alpha*x_grad
        theta=theta-alpha*theta_grad
        j_history.append(cost)
    return X,theta,j_history

# Inputs
dataset=scipy.io.loadmat('collaborative fitering-movies.mat')
R=dataset['R'] # indicators if i user has given rating to j movie
Y=dataset['Y'] # movie ratings
#param=scipy.io.loadmat('C:/Users/prasun.j/Desktop/ng/collab filter param.mat')
num_features=10 # no of features
iterations=400 # 100 will also works
movieList = open("collab_movie_ids.txt", "r", encoding="ISO-8859-1").read().split("\n")[:-1]
my_ratings = np.zeros((1682,1)) # Create own ratings
my_ratings[0] = 4 
my_ratings[97] = 2
my_ratings[6] = 3
my_ratings[11]= 5
my_ratings[53] = 4
my_ratings[63]= 5
my_ratings[65]= 3
my_ratings[68] = 5
my_ratings[82]= 4
my_ratings[225] = 5
my_ratings[354]= 5
Y = np.hstack((my_ratings,Y)) # adding myself in users list as 1st column
R =np.hstack((my_ratings!=0,R)) # adding myself in indicators list
num_users = Y.shape[1]
num_movies = Y.shape[0]
X = np.random.randn(num_movies, num_features)
theta = np.random.randn(num_users, num_features)
Lambda = 10
alpha=0.001

# Normalisation
Ynorm, Ymean = normalization(Y, R) # normalising entire data(including my data)

# Learning
X,theta,j_history=gradient_descent(X,theta,Y,R,iterations,alpha,Lambda) # input Y only

# Evaluation
plt.plot(j_history)
plt.xlabel("Iteration")
plt.ylabel("$J(\Theta)$")

# Prediction
p = np.dot(X,theta.T)
my_predictions = p[:,0][:,np.newaxis] + Ymean # my predicted ratings

# Recommendation
df = pd.DataFrame(np.hstack((my_predictions,np.array(movieList)[:,np.newaxis])))
df.sort_values(by=[0],ascending=False,inplace=True) # sorting predicted ratings given by me 
df.reset_index(drop=True,inplace=True)
print("Top recommendations for you:\n")
for i in range(10):
    print("Predicting rating",round(float(df[0][i]),1)," for index",df[1][i])
