import pyautogui
import numpy as np

def bound_by_screen(x, y, w_screen, h_screen):
    if x < 0:
        x, y = (20, y)  # Left boundary
    elif x > w_screen:
        x, y = (w_screen - 20, y)  # Right boundary

    if y < 0:
        x, y = (x, 20)  # Top boundary
    elif y > h_screen:
        x, y = (x, h_screen - 20)  # Bottom boundary

    return x, y


def control_mouse(x, y, w_screen, h_screen, click_signal=False):
    xx, yy = bound_by_screen(x, y, w_screen, h_screen)
    # Move the mouse to the specified location
    pyautogui.moveTo(xx, yy)  # The duration makes the movement smooth

    if click_signal:
        # Perform a click at the current position
        pyautogui.click()
        # pyautogui.click()

class Stack():
    def __init__(self):
        self.stack = np.array([])

    def push(self, item):
        item = np.array(item)
        self.stack = np.vstack((self.stack, item))

    def clean(self):
        self.stack = np.array([])

    def is_empty(self):
        return len(self.stack) == 0
    
    def size(self):
        return len(self.stack)
    
def control_click(x, y, w_screen, h_screen, stack, wait_counter, no_focus_counter):    
    # 如果栈中有100个坐标，实现点击，并清空栈
    if stack.size() >= 25:
        xx, yy = np.mean(stack.stack, axis=0)
        print("Click at ({}, {})".format(xx, yy))
        control_mouse(xx, yy, w_screen, h_screen)
        stack.clean()
        wait_counter = 25
        return wait_counter, stack, no_focus_counter
    
    # 如果焦点丢失超过10帧，则清空栈
    if no_focus_counter > 5:
        stack.clean()
        no_focus_counter = 0
        return wait_counter, stack, no_focus_counter

    # 如果等待计数器大于0，则等待
    if wait_counter > 0:
        print("Wait for {} frames".format(wait_counter))
        wait_counter -= 1
        return wait_counter, stack, no_focus_counter
    
    else:
        # 如果栈为空，则将当前坐标压入栈中
        if stack.is_empty():
            print("Stack is empty, push ({}, {}) into stack".format(x, y))
            stack.stack = np.array([[x, y]])
            return wait_counter, stack, no_focus_counter
        
        # 如果栈中有坐标，则计算当前坐标与栈中平均坐标的距离，如果距离小于150，则将当前坐标压入栈中
        else:
            mean_x, mean_y = np.mean(stack.stack, axis=0)
            distance = np.sqrt((x - mean_x) ** 2 + (y - mean_y) ** 2)
            if distance < 150:
                print("Push ({}, {}) into stack".format(x, y))
                stack.push([x, y])
                return wait_counter, stack, no_focus_counter
            
            else:
                no_focus_counter += 1
                return wait_counter, stack, no_focus_counter