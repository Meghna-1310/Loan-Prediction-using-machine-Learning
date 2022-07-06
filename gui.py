import tkinter as tk
from tkinter import *
from tkinter.filedialog import askopenfile
import pandas as pd
import pickle


data = None
path = None
root = tk.Tk()

root.geometry("1200x600")

root.resizable(0,0)
'''
frame = tk.Frame(master=root, width=100,
                 height=500, bg)
frame.pack()
'''
wrapper1 = tk.LabelFrame(root)

heading = tk.Label(wrapper1, text="Loan Prediction")
heading.config(font=("Arial", 44, 'bold'))
heading.place(x=370, y=10)

label = tk.Label(wrapper1,text='Upload csv file',anchor='w',bg="white",fg="gray",width=130,height=1)

label.place(x=35,y=105)

upload = tk.Button(
    wrapper1,
    text = 'Upload',
    width=5,
    height=1,
    )

def open_file(event):
    global data
    global label
    global path
    file = askopenfile(mode ='r', filetypes =[('CSV', '*.csv')])
    if file is not None:
        path = file.buffer.name
        data = pd.read_csv(path)
    label.config(text=path, fg='black')
    

upload.bind('<Button-1>',open_file)


upload.place(x=1100, y=100)

predict = tk.Button(
    wrapper1,
    text = 'Predict',
    width=5,
    height=1,
    )

def predict_loan(event):
    global data
    global path
    data = data.dropna()
    data1 = data.copy()
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    data['Gender'] = encoder.fit_transform(data['Gender'])
    data['Married'] = encoder.fit_transform(data['Married'])
    data['Dependents'] = encoder.fit_transform(data['Dependents'])
    data['Education'] = encoder.fit_transform(data['Education'])
    data['Self_Employed'] = encoder.fit_transform(data['Self_Employed'])
    data['Property_Area'] = encoder.fit_transform(data['Property_Area'])
    X = data.iloc[:, 1:]
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X = sc.fit_transform(X)
    classifier = pickle.load(open("predictive_model.sav", "rb"))
    y_pred = classifier.predict(X)
    y_pred = list(y_pred)

    for i in range(len(y_pred)):
        if y_pred[i] == 1:
            y_pred[i] = 'Approved'
        else:
            y_pred[i] = 'Not Approved'
    
    data1['Loan_Status'] = list(y_pred)
    lst1 = data1.values.tolist()
    lst1.insert(0,list(data1.columns))

    print(lst1)

    for i in range(261):
        t.insert(END,"-")
    t.insert(END,"\n")
    for i in range(len(lst1)):
        t.insert(END,"|")
        for j in range(13):
            s = str(lst1[i][j])+str(" "*(19-len(str(lst1[i][j]))))
            t.insert(END,s)
            t.insert(END,"|")
        t.insert(END,"\n")
        for k in range(261):
            t.insert(END,"-")
        t.insert(END,"\n")

    data1.to_csv(path[:-4]+"_predictive.csv", index = False, header=True)



predict.bind('<Button-1>',predict_loan)

predict.place(x=550, y=150)
wrapper2 = tk.LabelFrame(root)

h = tk.Scrollbar(wrapper2, orient = 'horizontal') 
h.pack(side = tk.BOTTOM, fill = tk.X)  
v = tk.Scrollbar(wrapper2)  
v.pack(side = tk.RIGHT, fill = tk.Y)

t = Text(wrapper2, width = 15, height = 10, wrap = NONE, 
                 xscrollcommand = h.set,  
                 yscrollcommand = v.set)


data1 = pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
lst = list(data1.columns)




t.pack(side=tk.BOTTOM, fill='both', expand='yes')

h.config(command=t.xview)
v.config(command=t.yview)

wrapper1.pack(fill='both', expand='yes')

wrapper2.pack(fill='both', expand='yes')

root.mainloop()
