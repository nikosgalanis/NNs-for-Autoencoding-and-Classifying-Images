def list_difference(l1, l2):
    diff = 0
    index = -1
    for i, (x, y) in enumerate(zip(l1, l2)):
        if x != y:
            diff += 1
            index = i

    return (diff, index)


l1 = (1, 2, 3)
l2 = (1, 2, 4)

print(list_difference(l1, l2))

print(l1[1:])
