[![Wisdomic Panda](https://github.com/robagwe/wisdomic-panda/blob/master/imgs/panda.png)](http://www.rohanbagwe.com/)  **Wisdomic Panda**
> *Hold the Vision, Trust the Process.*


# Essentials of Machine Learning Algorithms, Top 10 ML Algos you need to code! 
*... Building systems (or models) that can learn from experience (or datasets).*
###### python | scikit-learn | numpy | pandas | matplotlib

### What is Machine Learning?

*A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P if its performance at tasks in T, as measured by P, improves with experience E. But to the best of my knowledge, I would say...*


> #### "Machine Learning creates an illusion of Intelligence, but at it’s core it is just a Mathematical Optimization. Even though it posses an ability to make decisions and classify datasets, it is very narrow in the way it works. Currently, most AI systems are based on layers of mathematics that are only loosely inspired by the way the human brain works. But the brains of mammals do not use an activation function at synapses. The Neurons have an electric activations that result in a 'wave of propagation' through the axon, and undeniably Machine Learning algorithms are not architected in the same way. The reason for this is we cannot understand the 'SourceCode' of the human brain that maintains around 100 billion neurons 24\*7*365 days, nor can we simply feed it values and debug results. And thus, *Machine Learning is about making decision based on trial and error and is a more sophisticated and application oriented version of statistics.*"

*"ML is an example of artificial narrow intelligence (ANI). Meanwhile, we’re continuing to make foundational advances towards human-level artificial general intelligence (AGI), also known as strong AI. The definition of an AGI is an artificial intelligence that can successfully perform any intellectual task that a human being can, including learning, planning and decision-making under uncertainty, communicating in natural language, making jokes, manipulating people, trading stocks, or reprogramming itself, which is a big deal! Once we create an AI that can improve itself, it will unlock a cycle of recursive self-improvement that could lead to an intelligence explosion over some unknown time period, ranging from many decades to a single day." - Vishal Maini \[Research comms @DeepMindAI.]*

## :clipboard: Table of Contents:
   
 1. <a href="https://github.com/robagwe/wisdomic-panda/blob/master/cookbook.txt">Linear Regression</a>
 2. <a href="https://github.com/robagwe/wisdomic-panda/blob/master/cookbook.txt">Logistic Regression</a>
 3. <a href="https://github.com/robagwe/wisdomic-panda/blob/master/cookbook.txt">KNN Classifier</a>
 4. <a href="https://github.com/robagwe/wisdomic-panda/blob/master/cookbook.txt">NB Classifier</a>
 5. <a href="https://github.com/robagwe/wisdomic-panda/blob/master/cookbook.txt">SVM</a>
 6. <a href="https://github.com/robagwe/wisdomic-panda/blob/master/cookbook.txt">Decision Trees</a>
 7. <a href="https://github.com/robagwe/wisdomic-panda/blob/master/cookbook.txt">RF Classifier</a>
 8. <a href="https://github.com/robagwe/wisdomic-panda/blob/master/cookbook.txt">Boosting</a>
 9. <a href="https://github.com/robagwe/wisdomic-panda/blob/master/cookbook.txt">Clustering</a>
 10. <a href="https://github.com/robagwe/wisdomic-panda/blob/master/cookbook.txt">ANN</a>
 
 ###### [Datasets](https://github.com/robagwe/wisdomic-panda/blob/master/cookbook.txt)
 
#### :heavy_exclamation_mark: I run on Mac OS/Ubuntu so you might have to slightly modify the code to make it work in your env.

 ### :coffee: Ingredients:

- python
- scikit-learn
- pandas
- numpy
- matplotlib
- Spyder IDE
- Ubuntu 16.4 LTS


> ###### A quick run-through of some key pointers in machine learning.

### ML Cheat-sheet

| **Algorithm** | **Definition** |
| :---         |  :---    |
| **Linear Regression** | A supervised learning algorithm for finding the correlation between two continuous variables. After finding the correlation, it predicts a real value corresponding to the un-trained continuous variables. |
| **Logistic Regression**  | Majorly used in binary classification problems. It predicts the occurrence of an event depending on the values of an independent variable,that could be categorical or numerical. For e.g. It can predict the chance that a patient has a particular disease given certain symptoms. |
| **Naive Bayes** | Classification technique based on Bayesian Theorem. Parameter estimation for naive bayes models uses the method of maximum likelihood. Assume that the value of a particular feature is independent of the value of any other feature, given the class variable. Particularly suited when the dimensionality of inputs is high. |
| **SVM (Support Vector Machine)** |   The fundamental idea behind SVM is to separate the classes with a straight line (hyper plane). Help to maximize the margin i.e. the distance between the separating hyperplane and the training classes that are closest to the hyperplane. Most commonly used in text classification problems.  |
| **Decision Tree** |  Supervised learning algorithm, used for classification tasks. Here dataset / features split into smaller sub-groups (decision nodes where sub-nodes exists) to best describe the groups and input variables.  |
| **KNN**  |Here ‘K’ is the number that decides how many neighbors are defined based on distance metric that influence the classification. Classification is done by majority vote of its neighbors, and assigned to the class most common amongst its K nearest neighbors measured by a distance function. If K=1, then assign to the class to nearest neighbor.| 
| **Neural Network**  |Similar to linear/logistic regression, but with the addition of an activation function which makes it possible to predict outputs that are not linear combinations of inputs.|


### Key Concepts in ML

| Concept | Explanation |
| :---         |  :---    |
| **Gradient Descent / Stochastic Gradient Descent (SGD)** | Parameter optimization algorithm to obtain the best value of parameters which will give the highest accuracy on validation data. SGD or stochastic gradient descent is a form of gradient descent in which parameters are updated for each training data. |
| **Overfitting** | When the model performs well on the input data but poorly on the test data, it generally happens when the model is complex, means too many parameters. |
| **Underfitting** | When the model is unable to learn from the training data and performs poor on test data.|
| **Hyperparameter** | Parameter of learning algorithms, fixed before start of training process. Regularization is generally applied or controlled using hyperparameters. It can be decided through choosing different values, and training models.|
| **Activation Function** | The activation function is mostly used to make a non-linear transformation which allows us to fit nonlinear hypotheses or to estimate the complex functions. There are multiple activation functions, like: “Sigmoid”, “Tanh”, "ReLu" and many other.|
| **Bias** | Measures the gap in prediction and the correct value if the model is rebuilt multiple times on different training datasets. A high bias model is most likely to underfit the training data. |
| **Variance** | Measures the variability of the model prediction if the model is retrained multiple times, means model is sensitive to small variations in the training data. High variance is likely to overfit the training data. |
| **Bias/Variance Tradeoff** | Tradeoff means increase variance, reduce its bias and vice versa. Generally increasing model complexity increases the variance and reduces its bias. Similarly reducing model complexity reduces the variance, and increases its bias. One way of finding a good bias-variance tradeoff is to tune the complexity of model via regularization.|
| **Weight** | A type of parameter used in neural network, which determines the relevance of feature for the next layer of neural network. | 
| **Cost Function** | Cost function or error function is used to calculate the error obtained when data is being trained using a machine learning algorithm. Cost function is used for optimizing the parameters in GD algorithm. |
| **Confusion Matrix** | Used to evaluate the performance of a classifier. It counts the number of times a class is wrong and classified to any other class. Prediction is compared with actual targets. It defines matrix on: 1. True Positives (TP) 2. True Negatives (TN) 3. False Positives (FP) 4. False Negatives (FN) |

 
 # :construction: Kick-off-ML-Algorithms-from-scratch

<img src="https://github.com/robagwe/wisdomic-panda/blob/master/imgs/uc.gif" width="700">

##### :pushpin: Well, thanks to one of my lunch mates who has encouraged me to draft kick-off write-ups for each ML algorithm and I'm planning to start the activity soon. I'll update this repo once its done. Please stay tuned.  
   
   
## <img src="https://github.com/robagwe/wisdomic-panda/blob/master/imgs/acr.png" width="50">   Hey Buddy!</img>

> This write-up is targeting people who already know the basics of ML algorithms or could implement them from scratch. 
 Here, all algorithms are implemented in Python, using scikit-learn, matplotlib and numpy, have a dekko at it!. I'll keep updating this repo myself for each ML Algorithm, but what I really hope is you collaborate on this work. Your contributions are always welcome! Feel free to improve existing code, documentation or implement new algorithm.
Also, please follow if you'd be interested in reading it. Keep yourself updated with the latest science and technology affairs          which will help you with your AI learning initiatives. Please follow if you find it handy and hit :star: to get more kick-off repo updates.

:email: [Drop In!!](https://www.rohanbagwe.com) Seriously, it'd be great to discuss Technology.

### Happy Learning!

> *"Man cannot discover new oceans unless he has the courage to lose sight of the shore" - Andre Gide*



