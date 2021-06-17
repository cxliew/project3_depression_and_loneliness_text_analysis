# Project 3: Depression and Loneliness Text Analysis

### Problem Statement
With the increasing comparison behaviour due to the constant flow of information from the internet and social media, coupled with the lack of physical social interactions, there is an increasing trend in both depression and loneliness among younger millennials [1]. As a data scientist working in a mental healthcare institutition, we are interested to first explore the top 10 differences in word expression between depression and loneliness behavioural through the social media forum. We will be using 4 types of models namely the null model, multinomial naive bayes, k-nearest neighbours and logistic regression to predict the identification of depression and loneliness of the social media forum. Of these models, we will select the model that are able to provide the highest accuracy with more than 80% that takes into the account of the similarity between depression and loneliness. This will allows us to gain an insight to understand the early behavioural symptoms through their words usage and instill this as part of the assessment and identification when communicating with person in need and provide better assistance through other communication means including the chatbox and the helpline with the model built with more than 80% accuracy.

---
### Background

Depression and loneliness are terms that are often associated together due to the common belief that loneliness is a subset of depression and they have high similarity such as the symptoms of being in pain and helplessness [2]. However, the main distinction between depression and loneliness is loneliness is a transient emotional state that corresponds to a person needs for belonging and connection while depression is a mental health condition that can significantly affect a person's ability to function normally with its persistent negative feelings of emptiness, worthlessness, hopelessness and sadness [3] [4]. Loneliness condition can be improved should those needs be met, while depression on the other hand, requires treatment and can lead to suicidal on severe cases [5]. Nevertheless, a person experiencing loneliness should not be left unattended as it is an indicator of social well-being and prolonged loneliness can also lead to other mental health conditions including depression [3]. Hence, it is important to distinguish the differences of loneliness and depression to allow for early detection and offer the appropriate treatment to combat this before it prolonged to undesire consequences.

Globally, depression is ranked as the single largest contributor to global disability by World Health Organization (WHO) whereby, it is estimated to exceed 300 million people (4.4% of the world's population) having depression in 2015 and it is a major contributor to suicide deaths (up to 800 thosands per year) [6]. The high prevalence rate of depression and suicidal is due to barrier to effective care including stigma surrounding mental disorders, lack of awareness and inaccurate assessment [7]. With the increase usage of internet and social media, there is an increasing trend in both depression and loneliness among younger millenials in which the usage of these platforms encourage unintentional comparison behaviour with their peers and instill a sense of disconnect from real-life social interaction[8]. This leave them feeling insecure and unaccomplished, thus resulting in lower self-esteem, anxiety and depression. Thus, it is important to create awareness and reachout to them by being available and accessible for support or assistance should they experience depression or loneliness, which is a factor leading to depression.

Hence, as a data scientist working in a mental healthcare institutition, I will be handling on this project and our goals are the following:

1. To identify the top 10 differences between depression and loneliness behavioural through the word expression in social media forum

2. To build models such as null model, multinomial naive bayes, k-nearest neighbours and logistic regression to predict the identification of depression and loneliness behavioural with using the social media forum as a platform to measure the accuracy of the model prediction in this identification.

3. To select a model among the models built with the highest accuracy and having more than 80% accuracy to predict the identification of depression and loneliness behavioural in the social media forum, with taking an account of the similarity of depression and loneliness. A recent study suggests that 18% of the loneliness is responsible for depression in older adults. [9]. As the quantitative measurement of loneliness responsible for depression in young adults are not commonly studied, we will use 20% as a benchmark with reference to the study.

By knowing top 10 words that differences that distinguish the depression and loneliness behaviourial, the mental institution will be able to use this insight to instill these words as part of their assessment or identification and offer help when communicating with person in need. In addition, the model built with more than 80% accuracy will enable to provide a better assistance through other communication means including chatbox and the helpline.


---
### Data Dictionary
The dataset contains the posts that were published under the social news website, [reddit](https://www.reddit.com) between September 2020 to March 2021. 

The dataset used for this analysis are as followed:--

* depression_data (5000 posts)
* lonely_data (5000 posts)
* depression_lonely_data (compilation of depression_data and lonely_data)
* depression_lonely_data_clean (compilation of depression_data and lonely_data that have undergone text preprocessing) - labelled as depression_lonely_data


|Feature|Type|Dataset|Description|
|:---|:---|:---|:---|
|**author**|*object*| depression_lonely_data|The author who write the post in subreddit|
|**subreddit**|*object*| depression_lonely_data|The forum in the website reddit|
|**selftext**|*object*| depression_lonely_data|The post in the subreddit|
|**title**|*object*| depression_lonely_data|The title of the post in the subreddit|
|**created_utc**|*int64*| depression_lonely_data|The utc time that the post is created|
|**selftext_word_count**|*int64*| depression_lonely_data|The word count of a post|
|**title_word_count**|*int64*| depression_lonely_data|The word count of a title post|
|**total_posts**|*int64*|depression_lonely_data|The number of posts an author posted during the period the data was collected|
|**selftext_tokenize**|*object*|depression_lonely_data|The posts that undergo tokenization|
|**selftext_stopremoval**|*object*| depression_lonely_data|The posts that undergo tokenization and removal of stop words|
|**title_tokenize**|*object*| depression_lonely_data|The title of the posts that undergo tokenization|
|**title_stopremoval**|*object*| depression_lonely_data|The title of the posts that undergo tokenization and removal of stop words|
|**selftext_lem**|*object*| depression_lonely_data|The posts that undergo tokenization, removal of stop words and lemmatization|
|**title_lem**|*object*| depression_lonely_data|The title posts that undergo tokenization, removal of stop words and lemmatization|
|**selftext_pstem**|*object*| depression_lonely_data|The posts that undergo tokenization, removal of stop words and stemming|
|**title_pstem**|*object*| depression_lonely_data|The title posts that undergo tokenization, removal of stop words and stemming|

The data source below are obtained from database [reddit](https://www.reddit.com):
* [depression](https://www.reddit.com/search/?q=depression) 
* [foreveralone](https://www.reddit.com/search?q=foreveralone&type=link)


---
### Data Cleaning

For both depression_data and lonely_data:

* Selected columns only for further analysis (drop other columns)
* Drop missing values
* Compile the two data into one dataframe labeled depression_lonely_data

For depression_lonely_data: Data cleaning were performed. The text are cleaned as followed:-

* Removal of newline (\n) and return (\r)
* Removal of url, short links
* Removal of users (@username)
* Removal of special symbols (&amp, &agt, &lt, #x200b)
* Removal of hashtag
* Removal of punctuations
* Removal of numbers
* Removal of additional spaces to a single spacing

---
### Exploratory Data Analysis

To understand more on the selftext, title and authors, the following exploration has been performed:-

1. The length of the posts (selftext) and title by users
    * There is a non-symmetrical distribution in both selftext_word_count and title_word_count, which are highly positvely skewed.
    * Majority of the users have less than 500 words in selftext (post).
    * Majority of the users have less than 15 words in the title post.
    * Majority of the title posts in foreveralone have a longer range of words than in depression.


2. Frequency of total posts by authors
    * There is a non-symmetrical distribution in total_posts of depression and foreveralone and foreveralone has more posts per user in comparison with depression.
    * The are unique authors present in both depression and foreveralone, which is less than 1% of the compilation dataset.
    * it is to note that there are authors that are lonely and depressed but as there is 1% authors of the both forums, the dataset is representative for our further analysis to distinguish and classify between the two forums.


3. The usage of emojis in depression_data and lonely_data
    * Of the total selftext, 2% of the selftext contains emoji and the emoji usage are balanced between the two forums
    * Of the total title, 0.4% of the selftext contains emoji and 22 of these, the emoji usage are slightly higher in depression forum compared to foreveralone with a ratio of 60:40.
    * Due to the title and selftext having emoji is very few, the emoji usage in title would not be able to use to distiguish the forum in which the posts belong to.


4. Preprocessing and word counts
    * The selftext and title has been subjected to stop removal using NLTK library as a reference and stemming or lemming.
    * The english stop words in countvectorizer further reduces the number of common words on top of the removal of stop words using NLTK library.
    * Stemming helps to revert the words back to their root form, which reduces the number of words with the expense of some truncated words. This results in the changes in counts of certain words and enhance the detection of words that are not present in lemming results.
    * We have discovered words that occurred in top 30 words specific to depression forum only and foreveralone forum only. The words found in both selftext and title can be used as words to distinguish the differences between loneliness and depression in social media forum and will provide an insight to facilitate the mental health institution in the assessment or identification process of people experiencing loneliness and depression. 

----
### Data Preprocessing and Modeling & Analysis Summary

Part of the preprocessing has been performed in the exploratory data analysis section
* Drop missing values
* Binarize the subreddit

For modeling, we will only be focusing on selftext.

We will be performing a train/test split on our data set to have a training set and a holdout set. In total, we will be using a null model as the baseline model and 3 models to fit the dataset and evaluate the models in predicting the y-values using the validation set.

As this project aims to develop a model that are able to predict the identification of depression and loneliness of the social media forum. As the prediction is either a depression or a loneliness forum, it is a categorical classification, the models selected are as followed:-
* Null Model
* Multinomial Naive Bayes
* K-nearest neighbours
* Logistic Regression

**Model Evaluation**
* In our modeling, all the models has a score of more than 0.8 in comparison with the null model which is 0.5115, thus, all models here have some predictors value.
* In our modeling:

    * The multinomial naive bayes has a training and testing score of 0.87 and 0.85 respectively. 
    * The k-nearest-neighbours have a training and testing score of 0.9993 and 0.8217 respectively. It can be seen that there is an overfitting of the model due to a very high training score in comparison with testing score. This might indicate a very high variance. As preliminary screening was performed and a similar results were observed. Thus, this may indicate that the choice of trained model may not be suitable or there is a very high complexity of data making it hard to predict.
    * The logistic regression have a training and testing score of 0.8879 and 0.8461 respectively.This might indicate that there is slight overfitting although ridge regularization was used. 
    * Nevertheless, all three model fits our study criteria with an accuracy of more than 80%. 
    
* With the evaluation of the scoring of each model, all three models have a high accuracy with more than 80%. 
* For all three models, they have a lesser misclassification rate than the null model of about 30%. Out of the three models, Multinomial Naive Bayes have the lowest misclassification rate, which signifies the lowest inaccurate predictions of the observations. 
* Other parameters have been compared such as sensitivity, specificity, precision, f1score and ROCC AUC score with Multinomial Naive Bayes having the highest score compared to the other models.
* With these findings, Multinomial Naive Bayes fits our criteria the best, as it has the highest score with an accuracy of more than 80%, the highest sensitivity, specificty, precision, f1score and ROCC AUC score compared to the other models. Thus, this selected model will help to predict the people having depression or loneliness through other communication means including chatbox and the helpline and thus the mental health institution is able to provide a better assistance and reachout to the person in need.


---

### Conclusions

In conclusion, we have collected about 5000 posts from each forum. We then performed data cleaning of the posts and title of the datasets followed by exploring and visualizing the data using exploratory data analysis. We then processed the text and title, and we subjected these into countvectorizer to transform the text and title into word counts which we then can analyze the top common words in each forum. From our findings, we have discovered words of the top 30 frequeny of the most common words that are specific to the forum. These word allows us to distinguish the differences between loneliness and depression in social media forum and will provide an insight to facilitate the mental health institution in the assessment or identification process of people experiencing loneliness and depression. 

Using the processed text, we built four models: null model, Multinomial Naive Bayes, k-Nearest Neighbours and Logistic Regression. All three models 
have a score of 0.8, which is higher than the null model, 0.5. Both Logistic Regression and Multinomial Naive Bayes have similar training and testing score while Multinomial Naive Bayes score better then the other models in other parameters including sensitivity, specificty, precision, F1 score and ROCC AUC score. Using this selected model, we are able to facilitate the identification of people potentially having depression or loneliness through communication means including chatbox and the helpline and thus the mental health institution is able to provide a better assistance and reachout to the person in need.


---

### Recommendations

Overall, the top 10 differences between depression and loneliness behavioural through the word expression is alone, girl, date, relationship, guy, depress, help, live, thought and need. These words found will help to distinguish the differences between loneliness and depression and will provide an insight to facilitate the mental health institution in the assessment or identification process of people experiencing loneliness and depression. 

Of the models built, Multinomial Naive Bayes model are able to achieve an accuracy rate of more than 80% and this model score better then the other models in other parameters including sensitivity, specificty, precision, F1 score and ROCC AUC score. Using this selected model, we are able to facilitate the identification of people potentially having depression or loneliness through communication means including chatbox and the helpline and thus the mental health institution is able to provide a better assistance and reachout to the person in need.

---
### Future Development

With these findings, the project can be further explored to classify the identification of people who are undergoing different severity of depression and loneliness such as from mild, moderate to severe condition.
This can be done using the data collection which we have with severe condition of depression [(r/depressed)](https://www.reddit.com/r/depressed/) and loneliness [(r/lonely)](https://www.reddit.com/r/lonely/). As suicidal and depression are associated, the project can also be explored in classifiying the identification of people who are depressed with people who are having severe depression leading to suicidal thoughts [(r/SuicideWatch)](https://www.reddit.com/r/SuicideWatch/). The identification of these three group pairs will help to provide insights to facilitate the mental health institution in the assessment of identification process of people experiencing these conditions and to offer reachout to the person in need on a timely manner.

---
### References
[1] C. Curley, "Why Millennial Depression Is on the Rise," *Healthline Media*, March 11, 2019. [Online]. Available: https://www.healthline.com/health-news/millennial-depression-on-the-rise#Is-social-media-to-blame? [Accessed: Apr. 17, 2021].

[2] R. Mushtaq, S. Shoib, T. Shah, and S. Mushtaq, "Relationship Between Loneliness, Psychiatric Disorders and Physical Health ? A Review on the Psychological Aspects of Loneliness," *Journal of Clinical and Diagnostic Research,* vol.8, no.9, WE01-04, September 20, 2014. doi: 10.7860/JCDR/2014/10077.4828 [Online]. Available: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4225959/ [Accessed: Apr. 17, 2021].

[3] C. Raypole, "Loneliness and Depression: What's the Connection?," *Healthline Media*, July 2, 2020. [Online]. Available: https://www.healthline.com/health/loneliness-and-depression [Accessed: Apr. 17, 2021].

[4] "Depression Hotline Number," *MentalHelp.net*. [Online]. Available: https://www.mentalhelp.net/depression/hotline/ [Accessed: Apr. 17, 2021].

[5] "The Long-term Effects of Not Seeking Treatment for Depression," *Salience TMS Neuro Solutions*, March 23, 2020. [Online]. Available: https://salienceneuro.com/long-term-effects-of-depression-2/ [Accessed: Apr. 17, 2021].

[6] "Depression and Other Common Mental Disorders," *World Health Organization*, January 3, 2017. [Online]. Available: https://www.who.int/publications/i/item/depression-global-health-estimates [Accessed: Apr. 17, 2021].

[7] "Depression," *World Health Organization*, January 30, 2020. [Online]. Available: https://www.who.int/news-room/fact-sheets/detail/depression [Accessed: Apr. 17, 2021].

[8] J. E. Johnson, "Social Media Use, Social Comparison, and Loneliness," *Dissertations and
Theses*, Paper 5571, 2020, pp. 1-35. doi: 10.15760/etd.7445 [Online]. Available: https://pdxscholar.library.pdx.edu/cgi/viewcontent.cgi?article=6646&context=open_access_etds [Accessed: Apr. 17, 2021].

[9] "Loneliness a leading cause of depression in older adults," *ScienceDaily*, 2021. [Online]. Available: https://www.sciencedaily.com/releases/2020/11/201109184947.htm [Accessed: Apr. 22, 2021].

---
