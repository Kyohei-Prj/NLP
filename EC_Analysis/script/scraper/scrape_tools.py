from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from yaml_loader import load_yaml
from bs4 import BeautifulSoup
import requests
import pandas as pd
import time
import sys
import os


config = load_yaml(sys.argv[1])
browser_head = config['browser_head']
save_path = config['save_path']
browser_driver = config['driver']


class ScrapeStore:

    def __init__(self,
                 name,
                 main_url,
                 keywords,
                 button_id,
                 button_press,
                 sleep,
                 product_css,
                 flip_size,
                 url_head,
                 review_css=0):

        self.name = name
        self.main_url = main_url
        self.keywords = keywords
        self.button_id = button_id
        self.button_press = button_press
        self.sleep = sleep
        self.product_css = product_css
        self.flip_size = flip_size
        self.url_head = url_head
        self.review_css = review_css

    @staticmethod
    def load_scraper(url):

        headers = browser_head
        html = requests.get(url, headers=headers).text
        soup = BeautifulSoup(html, features='lxml')

        return soup

    def load_search_page(self):

        option = webdriver.ChromeOptions()
        option.add_argument('headless')
        driver = webdriver.Chrome(browser_driver, options=option)
        driver.get(self.main_url)

        return driver

    def get_tag(self, url_list, func_list, tag_atb_list, css):

        tag_dict = {}
        for i, url in enumerate(url_list):
            print('progress: {}/{}'.format(i+1, len(url_list)))
            soup = ScrapeStore.load_scraper(url)

            atb_dict = {}
            for func, atb in zip(func_list, tag_atb_list):
                atb_dict[atb] = [func(tag) for tag in soup.select(css)]

            time.sleep(self.sleep)

            for key in atb_dict.keys():
                tag_dict[(url, key)] = atb_dict[key]

        return tag_dict

    def get_tag_text(self, tag):

        return tag.text

    def get_tag_url(self, tag):

        return self.url_head + tag.get('href').replace('dp', 'review')

    def get_product_info(self, url_dict, func_list, tag_atb_list, css):

        product_dict = {}
        for keyword in url_dict.keys():
            print(self.name + ': ' + keyword)
            product_dict[keyword] = ScrapeStore.get_tag(
                self, url_dict[keyword], func_list, tag_atb_list, css)

        return product_dict

    def get_product_review(self, product_dict, func_list, tag_atb_list, css):

        review_dict = {}
        for key in product_dict.keys():
            for keyword in product_dict[key].keys():
                if keyword[1] == 'url':
                    review_dict[key] = ScrapeStore.get_tag(
                        self, product_dict[key][keyword], func_list,
                        tag_atb_list, css)

        return review_dict

    def save_to_csv(self, target_dict, content):

        save_dict = {}
        for key in target_dict.keys():
            save_list = []
            for keyword in target_dict[key].keys():
                if keyword[1] != 'url':
                    for item in target_dict[key][keyword]:
                        save_list.append(item)
            save_dict[key] = save_list

        df = pd.DataFrame.from_dict(save_dict, orient='index').T
        save_to = save_path + self.name + '_' + content + '.csv'
        if (os.path.exists(save_to)):
            df_arch = pd.read_csv(save_to)
            df_concat = pd.concat([df_arch, df], axis=1)
            df_concat.to_csv(save_to,
                             index=False,
                             encoding='utf_8_sig',
                             errors='ignore')
        else:
            df.to_csv(save_to,
                      index=False,
                      encoding='utf_8_sig',
                      errors='ignore')


class PageStore(ScrapeStore):
    def __init__(self, name, main_url, keywords, button_id, button_press,
                 sleep, product_css, flip_size, url_head, review_css,
                 flip_method):
        ScrapeStore.__init__(self, name, main_url, keywords, button_id,
                             button_press, sleep, product_css, flip_size,
                             url_head, review_css)
        self.flip_page = flip_method

    def get_keyword_url(self, driver):

        url_dict = {}
        for keyword in self.keywords:
            url_list = []
            search_box = driver.find_element_by_id(self.button_id)
            search_box.send_keys(keyword)
            click_box = driver.find_element_by_class_name(self.button_press)
            click_box.send_keys(Keys.ENTER)

            time.sleep(self.sleep)
            url_list.append(driver.current_url)

            search_box = driver.find_element_by_id(self.button_id)
            search_box.clear()

            for page in range(2, self.flip_size + 1):
                url_list.append(
                    driver.current_url.split('&')[0] + self.flip_page +
                    str(page))

            url_dict[keyword] = url_list

        return url_dict


class ScrollStore(ScrapeStore):
    def __init__(self, name, main_url, keywords, button_id, button_press,
                 sleep, product_css, flip_size, url_head, review_css,
                 flip_method):
        ScrapeStore.__init__(self,
                             name,
                             main_url,
                             keywords,
                             button_id,
                             button_press,
                             sleep,
                             product_css,
                             flip_size,
                             url_head,
                             review_css=None)
        self.scroll = flip_method

    def get_keyword_url(self, driver):

        url_list = []
        for keyword in self.keywords:
            search_box = driver.find_element_by_id(self.button_id)
            search_box.send_keys(keyword)
            click_box = driver.find_element_by_class_name(self.button_press)
            click_box.click()

            url_list.append(driver.current_url)
            time.sleep(self.sleep)

            search_box = driver.find_element_by_id(self.button_id)
            search_box.clear()

            for page in range(2, self.flip_size + 1):
                url_list.append(
                    driver.current_url.split('&')[0] + self.scroll + str(page))

        return url_list


def main():

    store = PageStore('store_name', 'url', ['word_1', 'word_2'], 'buttion_id',
                      'button_press', 15, 'product_css', 5, 'review_css')

    print(store.main_url)
    print(store.sleep)


if __name__ == '__main__':
    main()
