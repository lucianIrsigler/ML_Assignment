with open("traindata.txt","r") as f:
    data =[i.split(',') for i in f.readlines()]
    j = 2
    """with open(f"temp{j}.txt","w") as f1:
        for i in data[j]:
            f1.write(i+"\n")"""
    
    
    constIndexes = []
    constThing = data[0]
    for i in range(len(constThing)):
        valid = True
        for j in data:
            if j[i]!=constThing[i]:
                valid= False
        
        if valid==True:
            constIndexes.append(i)

    print(constIndexes)


    