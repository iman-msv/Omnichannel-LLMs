import pandas as pd
from scrapy import Selector

# Number of Pages
num_pages = 10

# Loading Downloaded HTML Files
def scraping_review_pages(html_file_path):
    with open(html_file_path, 'r') as file:
        html_string = file.read()
    
    sel = Selector(text=html_string)
    reviews_xpath = '//div[contains(@id, "customer_review")]/div[4]/span/span/text()'
    reviews = sel.xpath(reviews_xpath).extract()
    reviews = [review.strip() for review in reviews]

    short_reviews_xpath = '//div[contains(@id, "customer_review")]/div[2]/a/span[2]/text()'
    short_reviews = sel.xpath(short_reviews_xpath).extract()
    short_reviews = [review.strip() for review in short_reviews]

    rating_xpath = '//div[contains(@id, "customer_review")]/div[2]/a/i/span/text()'
    ratings = sel.xpath(rating_xpath).extract()

    date_xpath = '//div[contains(@id, "customer_review")]/span/text()'
    dates = sel.xpath(date_xpath).extract()

    return reviews, short_reviews, ratings, dates

reviews = []
short_reviews = []
ratings = []
dates = []

for i in range(1, num_pages + 1):
    path = f'Amazon Customer Review P{i}.html'

    review_res, short_review_res, rating_res, dates_res = scraping_review_pages(path)
    
    reviews.extend(review_res)

    short_reviews.extend(short_review_res)

    ratings.extend(rating_res)

    dates.extend(dates_res)

# Creating DataFrame
amazon_comments = pd.DataFrame({
    'review': reviews,
    'short_review': short_reviews,
    'rating': ratings,
    'date': dates
})

# Rating Must Be Float
amazon_comments['rating'] = amazon_comments['rating'].apply(lambda x: x[:3]).astype('float16')

# Dates Must Be in Date Type
amazon_comments['date'] = amazon_comments['date'].apply(lambda x: x.replace('Reviewed in the United States on', '')).astype('datetime64[ns]')
