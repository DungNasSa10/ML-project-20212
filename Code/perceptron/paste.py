import pyperclip

s = pyperclip.paste()
s = s.split()
t = str()
for i in s:
    t += " " + i + " &"
pyperclip.copy(t[:-1])