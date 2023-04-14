import pyautogui as auto
import time
import keyboard
import random
import threading

auto.FAILSAFE = True
# auto.PAUSE = 2.5


class Moniter:
    def __init__(self):
        self.work = True
        keyboard.add_hotkey('alt+q', self.stop)

        self.con_click = False
        self.click_interval = 0.25
        keyboard.hook(self.listen)

    def listen(self, event):
        x, y = event.name, event.event_type
        if x == 'space' and y == 'down' and self.con_click == False:
            self.con_click = True
            # so it begins clicking
            threading.Thread(target=self.clicking).start()
        if x == 'space' and y == 'up' and self.con_click == True:
            self.con_click = False

    def stop(self):
        self.work = False
        auto.alert("🍉 see you again 🍉🍉🍉🍉🍉🍉🍉🍉🍉🍉🍉🍉")
        # auto.alert("give u a 🍉, now get some rest :)")
        # self.work = True

    def ok(self):
        while self.work:
            # find ok and press enter*2
            if auto.locateOnScreen("assets/1ok.png", confidence=0.8):
                # print("find ok" + time.ctime())
                # auto.press("enter")
                # auto.press("enter")
                try:
                    x, y = auto.locateCenterOnScreen(
                        "assets/ok.png", confidence=0.5)
                    x0, y0 = auto.position()
                    auto.click(x, y)
                    auto.moveTo(x0, y0)
                except:
                    pass

    def dot(self):
        while self.work:
            try:
                x, y = auto.locateCenterOnScreen(
                    "assets/bdot.jpg", confidence=0.6)
                auto.click(x, y)
            except:
                pass

    def clicking(self):
        while self.con_click:
            auto.click()
            time.sleep(self.click_interval)

    def start_service(self):
        # threading.Thread(target=self.dot).start()
        threading.Thread(target=self.ok).start()


if __name__ == "__main__":
    m = Moniter()
    m.start_service()
