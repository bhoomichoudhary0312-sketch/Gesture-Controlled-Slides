import pyautogui

def zoom_in():

    pyautogui.keyDown("ctrl")
    pyautogui.scroll(500)
    pyautogui.keyUp("ctrl")

    print("Zoom In")


def zoom_out():

    pyautogui.keyDown("ctrl")
    pyautogui.scroll(-500)
    pyautogui.keyUp("ctrl")

    print("Zoom Out")