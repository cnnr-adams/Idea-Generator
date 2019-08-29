from bs4 import BeautifulSoup
import requests
import json
# (name, description, tags)
project_list = []
try:
    def scrape_website(url):
        response = requests.get(url)
        for project in json.loads(response.content)['software']:
            project_list.append((project['name'], project['tagline']))
        print(len(project_list))

    for i in range(1, 100000):
        print("https://devpost.com/software/search?page={}".format(i))
        scrape_website(
            "https://devpost.com/software/search?page={}".format(i))

except:
    with open("projects.json", "w") as f:
        json.dump(project_list, f)
