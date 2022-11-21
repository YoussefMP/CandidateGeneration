t = []

for i in range(23):
    t.append(i)


for j in range(5, len(t), 5):
    print(t[j-5: j])

    if j+5 > len(t):
        print(t[j:])

print(t)