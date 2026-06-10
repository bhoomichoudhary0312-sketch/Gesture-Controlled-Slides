import pyautogui

def next_slide():
    pyautogui.press("right")
    

def previous_slide():
    pyautogui.press("left")
    


if __name__ == "__main__":
    import time

    time.sleep(3)
    next_slide()    