import pyautogui
import time



def scroll_down():
    pyautogui.scroll(-500)
    print("Scroll Down")

def scroll_up():
    pyautogui.scroll(500)
    print("Scroll Up")



if __name__ == "__main__":
    time.sleep(3)
    scroll_down()
    print("Scrolled down")    