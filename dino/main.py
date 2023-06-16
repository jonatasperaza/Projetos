import random
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import time

def jump():
    driver.find_element_by_tag_name('body').send_keys(' ')
    print("Jumped!")

def duck():
    driver.find_element_by_tag_name('body').send_keys(webdriver.common.keys.Keys.ARROW_DOWN)
    print("Ducked!")

# Defina o caminho para o executável do WebDriver
webdriver_path = "C:/Chrome_driver/chromedriver.exe"

# Crie um objeto Service usando o caminho do WebDriver
service = Service(webdriver_path)

# Crie uma instância do driver passando o objeto Service
driver = webdriver.Chrome(service=service)

driver.get("chrome://dino:9515")
time.sleep(1200)

driver.find_element_by_xpath("""//*[@id="trex-controller"]/div[2]/button""").click()
time.sleep(1200)

while True:
    obstacle_height = driver.execute_script("return Runner.instance_.horizon.obstacles[0].yPos")
    if obstacle_height == 40:
        duck()
    else:
        jump()

    time.sleep(0.5)
