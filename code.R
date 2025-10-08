library(tidyverse)
library(dplyr)
library(ggplot2)
library(zoo)
library(VIM)
library(tseries)
library(forecast)
library(TTR)
library(tm)
library(wordcloud)
library(textstem)
library(textcat)
library(RColorBrewer)
library(topicmodels)
library(tidytext)
library(ldatuning)


##### PART 1 #####

# DATA UNDERSTANDING AND EXPLORATION
initial_data <- read.csv("PCE.csv", stringsAsFactors = FALSE, encoding='UTF-8')
dim(initial_data)
str(initial_data)
colnames(initial_data)
names(initial_data)[1] <- "Date" 
head(initial_data)
summary(initial_data)
initial_data$PCE <- as.numeric(initial_data$PCE)
clean_data <- initial_data %>%
  filter(!is.na(Date), !is.na(PCE))
ggplot(clean_data, aes(x = Date, y = PCE, group = 1)) +
  geom_line(color = "steelblue", linewidth = 1) +
  labs(
    title = "US Personal Consumption Expenditures Over Time",
    x = "Date",
    y = "PCE (Billions USD)"
  ) +
  theme_minimal()
adf.test(clean_data$PCE)
ggplot(clean_data, aes(y = PCE)) +
  geom_boxplot(fill = "skyblue") +
  labs(
    title = "Outlier Detection for Personal Consumption Expenditures",
    y = "PCE (Billions USD)"
  ) +
  theme_minimal()

# DATA CLEANING AND PREPARATION
pce_data <- initial_data
## handling dates
pce_data$Date <- as.Date(pce_data$Date, format = "%Y-%m-%d")
pce_data <- pce_data %>% arrange(Date)
## handling missing values
aggr(pce_data, col = c("navy", "yellow"), numbers = TRUE, sortVars = TRUE, 
     labels = names(initial_data), cex.axis = 0.7, gap = 3, 
     ylab = c("Missing data", "Patterns"))
pce_data$PCE <- na.approx(pce_data$PCE)
sum(is.na(pce_data))

# SIMPLE FORECASTING
start_year <- as.numeric(format(min(as.Date(pce_data$Date)), "%Y"))
start_month <- as.numeric(format(min(as.Date(pce_data$Date)), "%m"))
pce_ts <- ts(pce_data$PCE, start = c(start_year, start_month), frequency = 12)
plot(pce_ts)
pce_ma <- ma(pce_ts, order = 12)
plot(pce_ts)
lines(pce_ma, col="red", lwd=3)
train <- window(pce_ts, end=c(2022,12)) 
test <- window(pce_ts, start=c(2023,1))
naive_model <- naive(train, h = 12)
autoplot(naive_model) +
  ggtitle("Naive Forecast - Next 12 Months") +
  ylab("PCE (Billions USD)") + xlab("Year")
accuracy(naive_model, test)

# EXPONENTIAL SMOOTHING
hw_model <- hw(train, seasonal = "additive")
hw_forecast <- forecast(hw_model, h = 12)
autoplot(hw_forecast) + ggtitle("Holt-Winters Forecast (Additive Seasonality)")
accuracy(hw_forecast, test)

# ARIMA MODELLING
arima_model <- auto.arima(train)
arima_forecast <- forecast(arima_model, h=12)
autoplot(arima_forecast) + ggtitle("ARIMA Forecast - Next 12 Months")
accuracy(arima_forecast, test)
final_data <- window(pce_ts, end = c(2024,12))
final_arima_model <- auto.arima(final_data)

# FORECAST
final_forecast <- forecast(final_arima_model, h=12)
print(final_forecast)

##### PART 2 #####

# DATA UNDERSTANDING AND EXPLORATION
initial_data_2 <- read.csv("HotelsData.csv", stringsAsFactors = FALSE, encoding='UTF-8')
dim(initial_data_2)
str(initial_data_2)
colnames(initial_data_2)
names(initial_data_2)[1] <- "Review Score" 
names(initial_data_2)[2] <- "Text" 
head(initial_data_2)
summary(initial_data_2)

# DATA CLEANING AND PREPARATION
hotel_data <- initial_data_2
## checking missing values
sum(is.na(hotel_data))
## cleaning text
set.seed(504) 
sampled_reviews <- sample_n(hotel_data, 2000)
sampled_reviews$language <- textcat(sampled_reviews$Text)
table(sampled_reviews$language)
sampled_reviews_english <- sampled_reviews %>%
  filter(language == "english")
docs <- Corpus(VectorSource(sampled_reviews_english$Text))
clean_docs <- function(text_corpus) {
  text_corpus <- tm_map(text_corpus, content_transformer(tolower))
  text_corpus <- tm_map(text_corpus, removePunctuation)
  text_corpus <- tm_map(text_corpus, removeNumbers)
  text_corpus <- tm_map(text_corpus, removeWords, stopwords("english"))
  text_corpus <- tm_map(text_corpus, content_transformer(lemmatize_words))
  text_corpus <- tm_map(text_corpus, stripWhitespace)
  return(text_corpus)
}
hotel_data_clean <- clean_docs(docs)

# CREATING DOCUMENT TERM MATRIX
dtm <- DocumentTermMatrix(hotel_data_clean)
dtms <- removeSparseTerms(dtm, 0.99)
findFreqTerms(dtms, 200)

# WORDCLOUD
freq = data.frame(sort(colSums(as.matrix(dtms)), decreasing=TRUE))
wordcloud(rownames(freq), freq[,1], 
          max.words = 50, 
          colors = brewer.pal(3, "Dark2"),  
          scale = c(2, 0.5))  

# CATEGORISING REVIEWS
hotel_data$Sentiment <- ifelse(hotel_data$`Review Score` >= 4, "Positive", 
                               ifelse(hotel_data$`Review Score` <= 2, "Negative", 
                                      "Neutral"))
table(hotel_data$Sentiment)

# WORD FREQUENCY ANALYSIS BY SENTIMENT
## positive reviews
positive_reviews <- hotel_data %>% filter(Sentiment == "Positive")
positive_docs <- clean_docs(Corpus(VectorSource(positive_reviews$Text)))
positive_dtm <- DocumentTermMatrix(positive_docs)
positive_freq <- sort(colSums(as.matrix(positive_dtm)), decreasing = TRUE)
head(positive_freq, 10)
positive_top_words <- data.frame(Word = names(positive_freq)[1:20], 
                                 Frequency = positive_freq[1:20])
print(positive_top_words)
## negative reviews
negative_reviews <- hotel_data %>% filter(Sentiment == "Negative")
negative_docs <- clean_docs(Corpus(VectorSource(negative_reviews$Text)))
negative_dtm <- DocumentTermMatrix(negative_docs)
negative_freq <- sort(colSums(as.matrix(negative_dtm)), decreasing = TRUE)
head(negative_freq, 10)
negative_top_words <- data.frame(Word = names(negative_freq)[1:20], 
                                 Frequency = negative_freq[1:20])
print(negative_top_words)

# VISUALISATION
positive_top10 <- head(positive_top_words, 10)
negative_top10 <- head(negative_top_words, 10)
ggplot(positive_top10, aes(x = reorder(Word, Frequency), y = Frequency)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Top Positive Review Words", x = "Word", y = "Frequency")
ggplot(negative_top10, aes(x = reorder(Word, Frequency), y = Frequency)) +
  geom_bar(stat = "identity", fill = "firebrick") +
  coord_flip() +
  labs(title = "Top Negative Review Words", x = "Word", y = "Frequency")
wordcloud(words = positive_top_words$Word, 
          freq = positive_top_words$Frequency, 
          max.words = 20, 
          random.order = FALSE, 
          colors = brewer.pal(8, "Dark2"), 
          scale = c(3, 0.5))
wordcloud(words = negative_top_words$Word, 
          freq = negative_top_words$Frequency, 
          max.words = 20, 
          random.order = FALSE, 
          colors = brewer.pal(8, "Set1"), 
          scale = c(3, 0.5))


##### PART 2 #####

# DATA UNDERSTANDING AND EXPLORATION
initial_data_2 <- read.csv("HotelsData.csv", stringsAsFactors = FALSE, encoding='UTF-8')
dim(initial_data_2)
str(initial_data_2)
colnames(initial_data_2)
names(initial_data_2)[1] <- "Review Score" 
names(initial_data_2)[2] <- "Text" 
head(initial_data_2)
summary(initial_data_2)
unique(initial_data_2$`Review Score`)

# DATA CLEANING AND PREPARATION
hotel_data <- initial_data_2
## checking missing values
sum(is.na(hotel_data))
## cleaning text
set.seed(504) 
sampled_reviews <- sample_n(hotel_data, 2000)
sampled_reviews$language <- textcat(sampled_reviews$Text)
table(sampled_reviews$language)
sampled_reviews_english <- sampled_reviews %>%
  filter(language == "english")
docs <- Corpus(VectorSource(sampled_reviews_english$Text))
clean_docs <- function(docs) {
  docs <- tm_map(docs, content_transformer(tolower))
  docs <- tm_map(docs, removePunctuation)
  docs <- tm_map(docs, removeNumbers)
  docs <- tm_map(docs, removeWords, stopwords("english"))
  docs <- tm_map(docs, content_transformer(lemmatize_words))
  docs <- tm_map(docs, stripWhitespace)
  return(docs)
}
docs_2 <- Corpus(VectorSource(sampled_reviews_english$Text))
hotel_data_clean <- clean_docs(docs_2)

# CREATING DOCUMENT TERM MATRIX
dtm <- DocumentTermMatrix(hotel_data_clean)
dtms <- removeSparseTerms(dtm, 0.99)
findFreqTerms(dtms, 200)
freq = data.frame(sort(colSums(as.matrix(dtms)), decreasing=TRUE))
wordcloud(rownames(freq), freq[,1], rot.per=0.15, 
          random.order = FALSE, scale=c(4,0.5),
          random.color = FALSE, colors=brewer.pal(8,"Dark2"))

# WORD FREQUENCY ANALYSIS
sampled_reviews_english$Sentiment <- ifelse(sampled_reviews_english$`Review Score` >= 4, "Positive", 
                                            ifelse(sampled_reviews_english$`Review Score` <= 2, "Negative", 
                                                   "Neutral"))
table(sampled_reviews_english$Sentiment )
## positive reviews
positive_reviews <- sampled_reviews_english %>% filter(Sentiment == "Positive")
positive_docs <- clean_docs(Corpus(VectorSource(positive_reviews$Text)))
positive_dtm <- DocumentTermMatrix(positive_docs)
positive_dtm_sparse <- removeSparseTerms(positive_dtm, 0.99)
positive_freq <- sort(colSums(as.matrix(positive_dtm)), decreasing = TRUE)
positive_top15 <- head(positive_freq, 15)
positive_top15_df <- data.frame(Word = names(positive_top15), Frequency = positive_top15)
ggplot(positive_top15_df, aes(x = reorder(Word, Frequency), y = Frequency)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Top 15 Words in Positive Reviews", x = "Word", y = "Frequency")
## negative reviews
negative_reviews <- sampled_reviews_english %>% filter(Sentiment == "Negative")
negative_docs <- clean_docs(Corpus(VectorSource(negative_reviews$Text)))
negative_dtm <- DocumentTermMatrix(negative_docs)
negative_dtm_sparse <- removeSparseTerms(negative_dtm, 0.99)
negative_freq <- sort(colSums(as.matrix(negative_dtm)), decreasing = TRUE)
negative_top15 <- head(negative_freq, 15)
negative_top15_df <- data.frame(Word = names(negative_top15), Frequency = negative_top15)
ggplot(negative_top15_df, aes(x = reorder(Word, Frequency), y = Frequency)) +
  geom_bar(stat = "identity", fill = "firebrick") +
  coord_flip() +
  labs(title = "Top 15 Words in Negative Reviews", x = "Word", y = "Frequency")

# TOPIC MODELLING WITH LDA
## positive reviews
positive_result <- FindTopicsNumber(
  dtm = positive_dtm_sparse,
  topics = seq(2, 10, by = 1),
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010"),
  method = "Gibbs",
  control = list(seed = 1234),
  mc.cores = 2L,
  verbose = TRUE
)
FindTopicsNumber_plot(positive_result)
positive_lda_model <- LDA(positive_dtm_sparse, k = 9, method = "Gibbs", 
                          control = list(seed = 1234, burnin = 1000, thin = 100, 
                                         iter = 2000))
positive_topic_terms <- terms(positive_lda_model, 10)
print(positive_topic_terms)
positive_topic_labels <- c(
  "Room and Amenities", "Hotel and Dining", "Cleanliness and Convenience", 
  "Recommendation and Overall Experience", "Hospitality and Experience", 
  "Staff and Service", "Booking and Arrival", "Location and Accessibility", 
  "Overall Value and Feedback"
)
positive_topic_probabilities <- posterior(positive_lda_model)$topics
poisitve_topic_headings <- apply(positive_topic_probabilities, 1, which.max)
positive_reviews$Topic <- poisitve_topic_headings
positive_reviews$Topic_Label <- factor(poisitve_topic_headings, 
                                       labels = positive_topic_labels)
positive_topics <- as.data.frame(table(positive_reviews$Topic_Label))
colnames(positive_topics) <- c("Topic_Label", "Document_Count")
ggplot(positive_topics, aes(x = reorder(Topic_Label, Document_Count), 
                                  y = Document_Count)) +
  geom_bar(stat = "identity", fill = "darkgreen") +
  labs(title = "Number of Positive Reviews per Topic",
       x = "Topic",
       y = "Document Count") +
  coord_flip()
## negative reviews
negative_result <- FindTopicsNumber(
  dtm = negative_dtm_sparse,
  topics = seq(2, 10, by = 1),
  metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010"),
  method = "Gibbs",
  control = list(seed = 1234),
  mc.cores = 2L,
  verbose = TRUE
)
FindTopicsNumber_plot(negative_result)
negative_lda_model <- LDA(negative_dtm_sparse, k = 9, method = "Gibbs", 
                          control = list(seed = 1234, burnin = 1000, thin = 100, 
                                         iter = 2000))
negative_topic_terms <- terms(negative_lda_model, 10)
print(negative_topic_terms)
negative_topic_labels <- c(
  "Room Cleanliness and Maintenance", "Service Timing and Experience", 
  "Maintenance and Amenities", "Meeting Expectations", "Booking and Arrival", 
  "Accomodation and Food",
  "Staff and Service", "Sleep Quality and Room", 
  "Overall Value and Management"
)
negative_topic_probabilities <- posterior(negative_lda_model)$topics
negative_topic_headings <- apply(negative_topic_probabilities, 1, which.max)
negative_reviews$Topic <- negative_topic_headings
negative_reviews$Topic_Label <- factor(negative_topic_headings, 
                                       labels = negative_topic_labels)
negative_topics <- as.data.frame(table(negative_reviews$Topic_Label))
colnames(negative_topics) <- c("Topic_Label", "Document_Count")
ggplot(negative_topics, aes(x = reorder(Topic_Label, Document_Count), 
                                  y = Document_Count)) +
  geom_bar(stat = "identity", fill = "firebrick") +
  labs(title = "Number of Negative Reviews per Topic",
       x = "Topic",
       y = "Document Count") +
  coord_flip()












