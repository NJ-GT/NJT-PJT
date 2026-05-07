import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options

opts = Options()
opts.add_argument('--headless')
opts.add_argument('--no-sandbox')
opts.add_argument('--window-size=1600,1000')

driver = webdriver.Chrome(options=opts)
html_path = r'c:\Users\USER\Documents\GitHub\기말공모전\NJT-PJT\data\Map_Seoul10_Firestation.html'
out_path  = r'c:\Users\USER\Documents\GitHub\기말공모전\NJT-PJT\data\Map_Seoul10_Firestation.png'

driver.get('file:///' + html_path.replace('\\', '/'))
time.sleep(20)
driver.save_screenshot(out_path)
driver.quit()
print('done:', out_path)
