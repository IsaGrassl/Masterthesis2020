from nltk.sentiment import SentimentIntensityAnalyzer
from langdetect import detect
import pandas as pd


def make_sentiment_df_for_comments(comments):
    analyser = SentimentIntensityAnalyzer()
    summary = {'number_of_comments': len(comments), 'number_of_words': 0, 'number_of_eng': 0, 'negative': 0,
               'neutral': 0, 'positive': 0,
               'neg_score_mean': 0, 'pos_score_mean': 0, 'neu_score_mean': 0}
    data = {'comment_string': [],
            'neg': [],
            'neu': [],
            'pos': [],
            'compound': [],
            'is_english': []}
    for comment in comments:
        data['comment_string'].append(comment)
        score = analyser.polarity_scores(comment)
        __add_score_to_summary(score, summary)
        data['neg'].append(score['neg'])
        data['neu'].append(score['neu'])
        data['pos'].append(score['pos'])
        data['compound'].append(score['compound'])
        lang = __get_lang(comment)
        is_english = lang == 'en'
        data['is_english'].append(is_english)
        summary['number_of_words'] += len(comment.split())
    if summary['number_of_comments'] > 0:
        summary['neg_score_mean'] = sum(data['neg']) / len(data['neg'])
        summary['pos_score_mean'] = sum(data['pos']) / len(data['pos'])
        summary['neu_score_mean'] = sum(data['neu']) / len(data['neu'])
    summary['number_of_eng'] = sum(data['is_english'])
    df = pd.DataFrame(data)
    return summary, df


def __add_score_to_summary(score, summary):
    if score['compound'] >= 0.05:
        summary['positive'] += 1
    elif score['compound'] <= -0.05:
        summary['negative'] += 1
    else:
        summary['neutral'] += 1


def __get_lang(comment):
    try:
        return detect(comment)
    except:
        pass
    return ''
