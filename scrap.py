import requests
import torch
import os
import shutil
from bs4 import BeautifulSoup
from transformers import pipeline
from tqdm import tqdm
from datetime import datetime
from duckduckgo_search import DDGS
from urllib.parse import urlparse

date = datetime.now().strftime('%Y-%m-%d')

pref_url = 'https://huggingface.co'
url = pref_url + '/models?sort=likes'

model_class = 'overview-card-wrapper group/repo'
type_class = 'mr-1 flex items-center overflow-hidden whitespace-nowrap text-sm leading-tight text-gray-400'
desc_class = 'model-card-content'

gpu_av = torch.cuda.is_available()
print("Cuda ", 'un' if not gpu_av else '', 'available', sep='')
device = 'cuda' if gpu_av else 'cpu'

print("Loading summarizer...")

summarizer = pipeline(
    "summarization", 
    model="facebook/bart-large-cnn",
    torch_dtype=torch.float16,
    device=device
)
print("Summarizer loaded")

def clear_make_path(pth):
    if os.path.exists(pth):
        shutil.rmtree(pth)
    os.makedirs(pth)

post_pth = '_posts/' 
images_pth = 'assets/images'

clear_make_path(post_pth)
clear_make_path(images_pth)

post_pth += date + '-'
post_pref = '/models/' 

def make_md(title, layout='page', background=""):
    return \
f'''---
layout: {layout}
title: {title}
date: {date}
background: {background}
---
'''

def md_h(text: str, l=1):
    return '\n\n' + ('#' * l) + ' ' + text

def md_text(text: str):
    return '\n\n' + text

def md_link(text: str, link: str):
    return '\n\n' + '[' + text + '](' + link + ')'

def md_list(lst: list):
    return '\n\n' + '\n'.join(['- ' + i for i in lst])

def get_info(model):
    type = model.find('div', class_=type_class).text
    model_page_url = pref_url + model.find('a').get('href')

    page = requests.get(model_page_url).text
    page_soup = BeautifulSoup(page, 'html.parser')

    tags = page_soup.find_all('div', class_='tag tag-white')

    desc = page_soup.find_all('div', class_='model-card-content')
    desc = ' '.join([i.text for i in desc])
    desc = desc[:2400] if len(desc) > 2400 else desc

    license = page_soup.find('span', string='License:')
    if license is not None:
        license = license.next_sibling.next_sibling.text.strip()
    else:
        license = "ND"

    s_summary = summarizer(desc, max_length=50, min_length=20, do_sample=False)
    summary = summarizer(desc, max_length=400, min_length=100, do_sample=False)

    search_phraze = model.find('h4').get_text(strip=True).split('/')[1]
    ddgs_links = DDGS().text(search_phraze, max_results=5, safesearch='on', region='pl-pl')
    ddgs_links = [i['href'] for i in ddgs_links]

    img_url = DDGS().images(search_phraze, max_results=1, safesearch='on', region='pl-pl')[0]['image']
    img_res = requests.get(img_url)
    img_pth = images_pth + '/' + str(search_phraze).replace('.', '-').replace('/', '-').lower() + '.jpg'

    with open(img_pth, 'wb') as f:
        f.write(img_res.content)    

    data = {}

    data['name'] = model.find('h4').get_text(strip=True)
    data['type'] = type.split('•')[0].strip()
    data['url'] = model_page_url
    data['transformers'] = any(tag.find('span', string='Transformers') is not None for tag in tags)
    data['license'] = license
    data['likes'] = page_soup.find('button', title='See users who liked this repository').text.strip()    
    data['summary'] = summary[0]['summary_text']
    data['short-summary'] = s_summary[0]['summary_text']
    data['img'] = '/' + img_pth
    data['links'] = filter(lambda h: 'huggingface.co' not in h, ddgs_links)

    return data




list_md = make_md('Lista modeli z 🤗')

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

limit = 25
models = soup.find_all('article', class_=model_class)
models_classes = models[:limit]

print("Generating list and posts...")
for model in tqdm(models_classes):
    limit -= 1
    if limit == 0:
        break

    data = get_info(model)

    sub_md = make_md(title=data['name'], layout='post', background=data['img'])
    sub_md += md_text('**Type**: ' + data['type'])
    sub_md += md_text('**License**: ' + data['license'])
    sub_md += md_text('**Likes**: ' + data['likes'])
    sub_md += md_text('**Transformers**: ' + ('✅' if data['transformers'] else '❌'))
    sub_md += md_link('🤗 link', data['url'])
    sub_md += md_text('**Summary**:\n' + data['summary'])
    sub_md += md_text('**Useful links**:\n')

    for link in data['links']:
        domain = urlparse(link).netloc
        sub_md += f'\n - [{domain}/...]({link})'
    
    sub_name = post_pth + data['name'].replace('/', '_') + '.md'
    post_url = post_pref + data['name'].replace('/', '_')

    with open(sub_name, 'w') as f:
        print(sub_md, file=f)
    
    list_md += md_h(data['name'], 2)
    list_md += md_text('**Type**: ' + data['type'])
    list_md += md_text('**License**: ' + data['license'])
    list_md += md_text('**Likes**: ' + data['likes'])
    list_md += md_text('**Transformers**: ' + ('✅' if data['transformers'] else '❌'))
    list_md += md_link('🤗 link', data['url'])
    list_md += md_link('Post', post_url)
    list_md += md_text('**Short summary**:\n' + data['short-summary'])

with open('posts.md', 'w') as f:
    print(list_md, file=f)

print("Generation completed!")