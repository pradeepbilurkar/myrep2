import time
timestamplist=[]

def checkFatigue(timestamplist):
    for i in len(timestamplist):
        if timestamplist[-1]-timestamplist[i] <5:
            if i>=5:
                fatigue=True
                return fatigue

for i in range(15):
    time.sleep(5)
    timestamplist.append(time.time())

checkFatigue(timestamplist)