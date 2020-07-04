                                        Assignment-4
                                        
| Activation Function                          | Sigmoid                                     | ReLu                                         | Tanh                                       |
|----------------------------------------------|---------------------------------------------|----------------------------------------------|--------------------------------------------|
| L2 norm regularization (0.01) + epoch = 10   | 75.4% (val accuracy) 53.5% (test accuracy)  | 75.44% (val accuracy) 53.4% (test accuracy)  | 75.8% (val accuracy) 53.5% (test accuracy) |
| L2 norm regularization (0.001) + epoch = 10  | 75.6% (val accuracy) 53.09% (test accuracy) | 75% (val accuracy) 53.7% (test accuracy)     | 76.1% (val accuracy) 53.5% (test accuracy) |
| L2 norm regularization (0.01) + epoch = 20   | 76.4% (val accuracy) 53% (test accuracy)    | 75.33% (val accuracy) 53.02% (test accuracy) | 75.4% (val accuracy) 53.8% (test accuracy  |
| L2 norm regularization (0.001) + epoch = 20  | 75.9% (val accuracy) 53.6% (test accuracy)  | 75.3% (val accuracy) 53% (test accuracy)     | 76.2% (val accuracy) 54.2% (test accuracy) |


In this we have considered data with stopwords beause without sopwords as we have seen that in assignment-3 the accyracy was less because the meaning of the sentence changes because of removeal of stopwords. From the above table we can see that the accuracy with tanh is highest in comparision to other activation function. Even though the difference is very less. Tanh is showing consistently showing better result. The test accuracy is less because there are some words whch are not present in the vocabulary.
