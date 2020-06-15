Ques-1 Accuracy Table
| Stopword Removed | Text Features    | Accuracy (Test Set)  |
|------------------|------------------|----------------------|
| Yes              | Unigrams         | 80.63                |
| Yes              | Bigrams          | 82.53                |
| Yes              | Unigrams+Bigrams | 83.27                |
| No               | Unigrams         | 80.41                |
| No               | Bigrams          | 75.51                |
| No               | Unigrams+Bigrams | 82.56                |
Ques-2 Answer the following questions:
a)	Which condition performed better: with or without stop words? Write a brief
             paragraph (5-6 sentences) discussing why you think there is a difference in
             performance.
Ans:     As, from the above table it can be observed that the classifier performed better with                 stop words. Even though there us no much difference in the performance od the classifier with or without stop words. But classifier performs better with stop words consistently. The main reason behind this is because in case of sentimental analysis there are cases where the stop words plays an important role and if they are removed then the meaning of the sentence entirely change. Maybe the comment which is negative may become positive with removal of stop words. 

b)	Which condition performed better: unigrams, bigrams or unigrams+bigrams? Briefly (in 5-6 sentences) discuss why you think there is a difference?
Ans:     From the above table it can be observed that unigram+ bigram performs better than other two. It can be observed that only unigrams perform better that bigrams which is not at all common. Here, the reason might be that the longer n-grams would be less, leafing to higher idf value that is why bigrams performs the worst. On contrary, some meaningful phrases may consist of more than one word and have opposite meaning to the content. Thus, by considering the bigrams, we can make use of context to help with semantic analysis. That may be the reason why the mix of unigrams and bigrams perform better than only unigrams and bigrams.
