# Braindump

### Data
- language - we tried to get toefl but instead we found italki (scraped from language learning app) and lang8
- lang8 is good because it's big - we're thinking to join some languages on the same language branch to have less classes to predict
- also repo of guy who did the same thing
- age and gender - blog authorship corpus, short texts, noisy data, not all ages -> regression
- additionally for gender we have the email dataset for which we used an AI to pull author name from email metadata
- then we scraped a website with a list of names and their respective gender -> map those onto our dataset
- removing forwarded
- losing some data in the process, but that's ok
- mbti - was easy to find, a lot of kaggle datasets and some scraped reddit datasets
- we had to preprocess by eliminating links, etc.
- political - long articles - proved to be unusable, not enough personalized writing styles
- instead short tweets and synthetic data, much better for real life inference
- all data has been analyzed by data length (in tokens) and cut and combined so that token distribution is similar (and around 250)
- cutting long articles might have affected their usefullness (less context in a part of an article)

### Ideas
- idea 1: shap -> can help debug the model, check if it pays attention to meaningful things
- can help observe natural language tendencies of people in certain groups

### Observations (about the model)
- observation 1: predictions are better when writer is pretending to speak to someone
- (could be because of the nature of the data: tweets, emails)
- observation 2: age predictions tend to be higher (~25 for us) - could be because no labels for 21?

### Observations (about the results)
- liberals care more about interpunction? (find example and present on report?)
- women are nicer
- etc.