import requests
from bs4 import BeautifulSoup

list_of_new_ad = []

for page in range(1,10):
    if page%10==0:
        print(page)
        
    page = requests.get(f"https://divar.ir/s/iran/electronic-devices?page={page}")
    content = BeautifulSoup(page.content, "html.parser")
    list_of_all_ad = content.find_all("article", class_="kt-post-card kt-post-card--outlined kt-post-card--padded kt-post-card--has-action")
    
    for ad in list_of_all_ad:
        if ad.find('div', class_='kt-post-card__description').get_text() == 'نو':
            list_of_new_ad.append(ad.get_text("-"))
        else: 
            continue

print(list_of_new_ad)