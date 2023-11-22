import time
import pandas as pd
from tqdm import tqdm
from trafilatura.sitemaps import sitemap_search
from trafilatura import fetch_url, extract


def get_urls_from_sitemap(resource_url: str) -> list:
    """
    Funzione che crea un DataFrame Pandas di URL e articoli.
    """
    urls = sitemap_search(resource_url)
    return urls


def extract_article(url: str) -> dict:
    """
    Estrae un articolo da una URL con Trafilatura
    """
    downloaded = fetch_url(url)
    article = extract(downloaded, favor_precision=True)

    return article


def create_dataset(list_of_websites: list) -> pd.DataFrame:
    """
    Funzione che crea un DataFrame Pandas di URL e articoli.
    """
    data = []
    for website in tqdm(list_of_websites, desc="Websites"):
        urls = get_urls_from_sitemap(website)
        for url in tqdm(urls, desc="URLs"):
            d = {"url": url, "article": extract_article(url)}
            data.append(d)
            time.sleep(0.5)

    df = pd.DataFrame(data)
    df = df.drop_duplicates()
    df = df.dropna()

    return df


if __name__ == "__main__":
    list_of_websites = [
        "https://www.diariodiunanalista.it/",
    ]

    df = create_dataset(list_of_websites)

    df.to_csv("./data/articles.csv", index=False)
