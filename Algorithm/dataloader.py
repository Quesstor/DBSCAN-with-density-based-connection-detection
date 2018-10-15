def loadcsv():
    import csv
    data = list()
    with open('accidents_2012_to_2014.csv', encoding='utf8') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        headers = next(reader)
        for row in reader:
            lng = float(row[3])
            lat = float(row[4])
            
            #All
            #data.append([lng,lat]) 

            #London
            #if lng>-1.2 and lng<.75 and lat>51 and lat<52: data.append([lng,lat]) 
            
            #Liverpool and Manchester
            if lng>-3.1 and lng<-1.9 and lat>53.2 and lat<53.7: data.append([lng,lat])
    return data
    