from tkinter import *
import numpy as np
import pandas as pd
# from gui_stuff import *

# --------------------------- Original symptom & disease lists ---------------------------
l1=['back_pain','constipation','abdominal_pain','diarrhoea','mild_fever','yellow_urine',
'yellowing_of_eyes','acute_liver_failure','fluid_overload','swelling_of_stomach',
'swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm','throat_irritation',
'redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain','weakness_in_limbs',
'fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails',
'swollen_extremeties','excessive_hunger','extra_marital_contacts','drying_and_tingling_lips',
'slurred_speech','knee_pain','hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints',
'movement_stiffness','spinning_movements','loss_of_balance','unsteadiness',
'weakness_of_one_body_side','loss_of_smell','bladder_discomfort','foul_smell_of urine',
'continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria','family_history','mucoid_sputum',
'rusty_sputum','lack_of_concentration','visual_disturbances','receiving_blood_transfusion',
'receiving_unsterile_injections','coma','stomach_bleeding','distention_of_abdomen',
'history_of_alcohol_consumption','fluid_overload','blood_in_sputum','prominent_veins_on_calf',
'palpitations','painful_walking','pus_filled_pimples','blackheads','scurring','skin_peeling',
'silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister','red_sore_around_nose',
'yellow_crust_ooze']

disease=['Fungal infection','Allergy','GERD','Chronic cholestasis','Drug Reaction',
'Peptic ulcer diseae','AIDS','Diabetes','Gastroenteritis','Bronchial Asthma','Hypertension',
' Migraine','Cervical spondylosis',
'Paralysis (brain hemorrhage)','Jaundice','Malaria','Chicken pox','Dengue','Typhoid','hepatitis A',
'Hepatitis B','Hepatitis C','Hepatitis D','Hepatitis E','Alcoholic hepatitis','Tuberculosis',
'Common Cold','Pneumonia','Dimorphic hemmorhoids(piles)',
'Heartattack','Varicoseveins','Hypothyroidism','Hyperthyroidism','Hypoglycemia','Osteoarthristis',
'Arthritis','(vertigo) Paroymsal  Positional Vertigo','Acne','Urinary tract infection','Psoriasis',
'Impetigo']

# original global l2 as in your code (keeps same behaviour)
l2=[]
for x in range(0,len(l1)):
    l2.append(0)

# --------------------------- TESTING DATA df (original replacements preserved) ---------------------------
df=pd.read_csv("Training.csv")

df.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

X= df[l1]

y = df[["prognosis"]]
np.ravel(y)

# --------------------------- TRAINING DATA tr (original replacements preserved) ---------------------------
tr=pd.read_csv("Testing.csv")
tr.replace({'prognosis':{'Fungal infection':0,'Allergy':1,'GERD':2,'Chronic cholestasis':3,'Drug Reaction':4,
'Peptic ulcer diseae':5,'AIDS':6,'Diabetes ':7,'Gastroenteritis':8,'Bronchial Asthma':9,'Hypertension ':10,
'Migraine':11,'Cervical spondylosis':12,
'Paralysis (brain hemorrhage)':13,'Jaundice':14,'Malaria':15,'Chicken pox':16,'Dengue':17,'Typhoid':18,'hepatitis A':19,
'Hepatitis B':20,'Hepatitis C':21,'Hepatitis D':22,'Hepatitis E':23,'Alcoholic hepatitis':24,'Tuberculosis':25,
'Common Cold':26,'Pneumonia':27,'Dimorphic hemmorhoids(piles)':28,'Heart attack':29,'Varicose veins':30,'Hypothyroidism':31,
'Hyperthyroidism':32,'Hypoglycemia':33,'Osteoarthristis':34,'Arthritis':35,
'(vertigo) Paroymsal  Positional Vertigo':36,'Acne':37,'Urinary tract infection':38,'Psoriasis':39,
'Impetigo':40}},inplace=True)

X_test= tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)

# --------------------------- Prediction functions (kept unchanged logically) ---------------------------
def DecisionTree():

    from sklearn import tree

    clf3 = tree.DecisionTreeClassifier()   # empty model of the decision tree
    clf3 = clf3.fit(X,y)

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=clf3.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

    for k in range(0,len(l1)):
        # print (k,)
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf3.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break


    if (h=='yes'):
        t1.delete("1.0", END)
        t1.insert(END, disease[a])
    else:
        t1.delete("1.0", END)
        t1.insert(END, "Not Found")


def randomforest():
    from sklearn.ensemble import RandomForestClassifier
    clf4 = RandomForestClassifier()
    clf4 = clf4.fit(X,np.ravel(y))

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=clf4.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]

    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = clf4.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        t2.delete("1.0", END)
        t2.insert(END, disease[a])
    else:
        t2.delete("1.0", END)
        t2.insert(END, "Not Found")


def NaiveBayes():
    from sklearn.naive_bayes import GaussianNB
    gnb = GaussianNB()
    gnb=gnb.fit(X,np.ravel(y))

    # calculating accuracy-------------------------------------------------------------------
    from sklearn.metrics import accuracy_score
    y_pred=gnb.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(accuracy_score(y_test, y_pred,normalize=False))
    # -----------------------------------------------------

    psymptoms = [Symptom1.get(),Symptom2.get(),Symptom3.get(),Symptom4.get(),Symptom5.get()]
    for k in range(0,len(l1)):
        for z in psymptoms:
            if(z==l1[k]):
                l2[k]=1

    inputtest = [l2]
    predict = gnb.predict(inputtest)
    predicted=predict[0]

    h='no'
    for a in range(0,len(disease)):
        if(predicted == a):
            h='yes'
            break

    if (h=='yes'):
        t3.delete("1.0", END)
        t3.insert(END, disease[a])
    else:
        t3.delete("1.0", END)
        t3.insert(END, "Not Found")

# --------------------------- GUI (redesigned frontend; backend untouched) ---------------------------
root = Tk()
root.title("Disease Predictor - Shalini M S")
root.geometry("1050x720")
root.resizable(False, False)

# Professional / formal color palette
BG = "#F4F7FB"      # page background
HEADER_BG = "#1F3A93"  # deep navy
SIDEBAR_BG = "#FFFFFF"
PANEL_BG = "#F8FAFF"
ACCENT = "#2E86DE"
TEXT = "#222222"
BUTTON_BG = "#1F3A93"
BUTTON_FG = "white"
RESULT_BG = "#EEF2FF"

root.configure(bg=BG)

# --- Header ---
header = Frame(root, bg=HEADER_BG, height=90)
header.pack(fill='x')

title = Label(header, text="Disease Predictor using Machine Learning", bg=HEADER_BG, fg="white",
              font=("Segoe UI", 22, "bold"))
title.place(x=20, y=15)

author = Label(header, text="A Project by Shalini M S", bg=HEADER_BG, fg="white",
               font=("Segoe UI", 12, "italic"))
author.place(x=22, y=52)

# --- Left Sidebar (portfolio-like) ---
sidebar = Frame(root, bg=SIDEBAR_BG, width=300, height=600, highlightbackground="#E1E8F6",
                highlightthickness=1)
sidebar.place(x=20, y=110)

# Placeholder 'photo' box
photo_frame = Frame(sidebar, bg="#E6EEF9", width=260, height=160)
photo_frame.place(x=20, y=18)
photo_label = Label(photo_frame, text="Photo\n(placeholder)", bg="#E6EEF9", fg=TEXT, font=("Segoe UI", 11))
photo_label.place(relx=0.5, rely=0.5, anchor="center")

# About / bio
about_title = Label(sidebar, text="About the Project", bg=SIDEBAR_BG, fg=TEXT, font=("Segoe UI", 12, "bold"))
about_title.place(x=20, y=200)

about_text = ("This GUI wraps three ML classifiers (Decision Tree, Random Forest, GaussianNB)\n"
              "to predict probable diseases based on selected symptoms.\n\n"
              "UI improved for clarity and presentation — core logic unchanged.")
about_label = Label(sidebar, text=about_text, justify=LEFT, bg=SIDEBAR_BG, fg="#333333",
                    font=("Segoe UI", 10), wraplength=260)
about_label.place(x=20, y=230)

# Contact / quick info
contact_title = Label(sidebar, text="Quick Info", bg=SIDEBAR_BG, fg=TEXT, font=("Segoe UI", 12, "bold"))
contact_title.place(x=20, y=370)

contact_text = "Author: Shalini M S\nEmail: (add if desired)\nProject: Disease Predictor"
contact_label = Label(sidebar, text=contact_text, justify=LEFT, bg=SIDEBAR_BG, fg="#333333",
                      font=("Segoe UI", 10))
contact_label.place(x=20, y=400)

# --- Main panel (inputs + results) ---
main_panel = Frame(root, bg=PANEL_BG, width=680, height=600, highlightbackground="#E1E8F6",
                   highlightthickness=1)
main_panel.place(x=340, y=110)

# Instruction
instr = Label(main_panel, text="Enter patient details and choose up to 5 symptoms from the list.",
              bg=PANEL_BG, fg=TEXT, font=("Segoe UI", 11))
instr.place(x=20, y=12)

# Entry variables (keep original variable names)
Symptom1 = StringVar()
Symptom1.set(None)
Symptom2 = StringVar()
Symptom2.set(None)
Symptom3 = StringVar()
Symptom3.set(None)
Symptom4 = StringVar()
Symptom4.set(None)
Symptom5 = StringVar()
Symptom5.set(None)
Name = StringVar()

# Input fields layout (kept variable names consistent)
label_name = Label(main_panel, text="Name of Patient:", bg=PANEL_BG, fg=TEXT, font=("Segoe UI", 11))
label_name.place(x=20, y=50)
NameEn = Entry(main_panel, textvariable=Name, font=("Segoe UI", 11), width=28)
NameEn.place(x=160, y=50)

# Symptom option lists
OPTIONS = sorted(l1)

s1_lbl = Label(main_panel, text="Symptom 1:", bg=PANEL_BG, fg=TEXT, font=("Segoe UI", 11))
s1_lbl.place(x=20, y=95)
S1En = OptionMenu(main_panel, Symptom1, *OPTIONS)
S1En.config(font=("Segoe UI", 10))
S1En.place(x=160, y=90, width=360)

s2_lbl = Label(main_panel, text="Symptom 2:", bg=PANEL_BG, fg=TEXT, font=("Segoe UI", 11))
s2_lbl.place(x=20, y=135)
S2En = OptionMenu(main_panel, Symptom2, *OPTIONS)
S2En.config(font=("Segoe UI", 10))
S2En.place(x=160, y=130, width=360)

s3_lbl = Label(main_panel, text="Symptom 3:", bg=PANEL_BG, fg=TEXT, font=("Segoe UI", 11))
s3_lbl.place(x=20, y=175)
S3En = OptionMenu(main_panel, Symptom3, *OPTIONS)
S3En.config(font=("Segoe UI", 10))
S3En.place(x=160, y=170, width=360)

s4_lbl = Label(main_panel, text="Symptom 4:", bg=PANEL_BG, fg=TEXT, font=("Segoe UI", 11))
s4_lbl.place(x=20, y=215)
S4En = OptionMenu(main_panel, Symptom4, *OPTIONS)
S4En.config(font=("Segoe UI", 10))
S4En.place(x=160, y=210, width=360)

s5_lbl = Label(main_panel, text="Symptom 5:", bg=PANEL_BG, fg=TEXT, font=("Segoe UI", 11))
s5_lbl.place(x=20, y=255)
S5En = OptionMenu(main_panel, Symptom5, *OPTIONS)
S5En.config(font=("Segoe UI", 10))
S5En.place(x=160, y=250, width=360)

# Buttons (styled)
dst = Button(main_panel, text="Decision Tree", command=DecisionTree, bg=BUTTON_BG, fg=BUTTON_FG,
             font=("Segoe UI", 11, "bold"), width=15)
dst.place(x=60, y=305)

rnf = Button(main_panel, text="Random Forest", command=randomforest, bg=BUTTON_BG, fg=BUTTON_FG,
             font=("Segoe UI", 11, "bold"), width=15)
rnf.place(x=260, y=305)

lr = Button(main_panel, text="Naive Bayes", command=NaiveBayes, bg=BUTTON_BG, fg=BUTTON_FG,
            font=("Segoe UI", 11, "bold"), width=15)
lr.place(x=460, y=305)

# Divider
divider = Frame(main_panel, bg="#E1E6F6", height=2, width=620)
divider.place(x=20, y=350)

# Results area (kept t1, t2, t3 variable names to avoid changing logic)
res_title = Label(main_panel, text="Prediction Results:", bg=PANEL_BG, fg=TEXT, font=("Segoe UI", 13, "bold"))
res_title.place(x=20, y=370)

# Decision Tree result
dt_label = Label(main_panel, text="DecisionTree:", bg=PANEL_BG, fg=TEXT, font=("Segoe UI", 11))
dt_label.place(x=20, y=410)
t1 = Text(main_panel, height=1, width=60, bg=RESULT_BG, fg="black", font=("Segoe UI", 11))
t1.place(x=140, y=408)

# Random Forest result
rf_label = Label(main_panel, text="RandomForest:", bg=PANEL_BG, fg=TEXT, font=("Segoe UI", 11))
rf_label.place(x=20, y=450)
t2 = Text(main_panel, height=1, width=60, bg=RESULT_BG, fg="black", font=("Segoe UI", 11))
t2.place(x=140, y=448)

# Naive Bayes result
nb_label = Label(main_panel, text="NaiveBayes:", bg=PANEL_BG, fg=TEXT, font=("Segoe UI", 11))
nb_label.place(x=20, y=490)
t3 = Text(main_panel, height=1, width=60, bg=RESULT_BG, fg="black", font=("Segoe UI", 11))
t3.place(x=140, y=488)

# Footer note
footer = Label(root, text="Note: Models are trained using provided Training.csv and tested with Testing.csv. "
                          "Accuracy metrics are printed to console when predicting.", bg=BG, fg="#555555",
               font=("Segoe UI", 9))
footer.place(x=20, y=690)

root.mainloop()
