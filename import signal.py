import keyboard


def on_key_press(event):
    if event.name == 't':
        print("按下了 t 键")


# 监听键盘事件
keyboard.on_press(on_key_press)

# 进入监听状态，等待键盘事件
keyboard.wait()
