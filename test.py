# -*- coding:utf-8 -*-
def exchange(value):
    if value == 2:
        value = 0
    elif value == 4:
        value = 1
    return value

print(exchange(2))