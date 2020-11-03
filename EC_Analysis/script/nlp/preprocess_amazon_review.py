import pandas as pd


def main():

    df = pd.read_csv('../../data/scraped_text/amazon_product_review.csv')
    df = df[1::2]
    save_to = '../../data/scraped_text/amazon_product_review_preprocessed.csv'
    df.to_csv(save_to, index=False)


if __name__ == '__main__':
    main()
