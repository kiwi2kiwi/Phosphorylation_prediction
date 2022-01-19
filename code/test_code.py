from pathlib import Path
WD = Path(__file__).resolve().parents[1]
lst = []
with open((WD/"pdb_collection"/"4fbn.txt"),"r") as r:
    for ln in r.readlines():
        lst.append(float(ln.split()[5]))

print(list(set(lst)).__len__())

lst = list(set(lst))
#lst = [1,2,3,4,5,7,8]
counter = 1
for i in lst[1:]:
    if i - lst[counter-1] != 1:
        print(i)
    counter += 1

[1,2,3].__contains__([1,2])

print("end")