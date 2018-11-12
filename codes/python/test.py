pathDB = os.getcwd()+'/database/'
DB_name = 'mitdb'
fs = 360
jump_lines = 1

    # Read files: signal (.csv )  annotations (.txt)    
fRecords = list()
fAnnotations = list()

lst = os.listdir(pathDB + DB_name + "/csv")
lst.sort()
fRecords = list()
fAnnotations = list()
for file in lst:
    if file.endswith(".csv"):
       
        fRecords.append(file)
    elif file.endswith(".text"):
      
        fAnnotations.append(file)        

MITBIH_classes = ['N', 'L', 'R', 'e', 'j', 'A', 'a', 'J', 'S', 'V', 'E', 'F']#, 'P', '/', 'f', 'u']
AAMI_classes = []
AAMI_classes.append(['N', 'L', 'R'])                    # N
AAMI_classes.append(['A', 'a', 'J', 'S', 'e', 'j'])     # SVEB 
AAMI_classes.append(['V', 'E'])                         # VEB
AAMI_classes.append(['F'])                              # F

RAW_signals = []
r_index = 0
