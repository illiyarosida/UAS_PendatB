import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

st.title(" Prediksi Penyakit Gagal Jantung")
st.write("##### Nama  : ILLIYA ROSIDA ")
st.write("##### Nim   : 210411100051 ")

#Navbar
data, preprocessing, modeling, implementation = st.tabs(["Data", "Preprocessing", "Modeling", "Implementation"])

df = pd.read_csv('https://raw.githubusercontent.com/illiyarosida/data/main/heart.csv')

#data_set_description
with data:
    st.write("###### Data Set Ini Adalah : Prediksi Gagal Jantung ")
    st.write('link datasets : https://raw.githubusercontent.com/illiyarosida/data/main/heart.csv')
    st.write('link Asal data dari Kaggle : https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction')
    st.write("""###### Isi kolom : """)
    st.write("""1.  :Age (Usia)

    Usia pada pasien penyakit Gagal Jantung
    """)
    st.write("""2. Sex (Jenis Kelamin) :
    Seks adalah pembagian dua jenis kelamin, yakni laki-laki dan perempuan, yang ditentukan secara biologis. Seks juga berkaitan dengan karakter dasar fisik dan fungsi manusia, mulai dari kromosom, kadar hormon, dan bentuk organ reproduksi.
    """)

    st.write("""3. ChestPainType :
     kondisi ketika dada terasa seperti tertusuk, perih, atau tertekan. Nyeri ini bisa terjadi di dada sebelah kanan, sebelah kiri, atau dada tengah. 

    """)
    st.write("""4. RestingBP :
    Pemeriksaan tekanan darah bertujuan untuk memantau sirkulasi darah dalam tubuh.

    """)

    st.write("""5. Cholestrol:
    Kolesterol adalah lemak yang diproduksi secara alami oleh organ hati.

    """)
    st.write("""6. FastingBs:
    Puasa gula darah, uji darah untuk memeriksa kadar gula dalam darah setelah berpuasa.

    """)
    st.write("""7. RostingECG:
    resting EKG adalah pemeriksaan EKG yang dilakukan saat pasien dalam kondisi istirahat (dalam posisi berbaring).

    """)
    st.write("""8. MaxHR:
    perhitungan detak jantung maksimal

    """)
    st.write("""9. ExercireseAngina:
    tes untuk menunjukkan bagaimana jantung bekerja selama aktivitas fisik/nyeri dada.

    """)
    st.write("""10. OLDPEAK:
    tingkat  diagnosis depresi

    """)
    st.write("""11. ST_Slope:
    Segmen ST adalah garis lurus yang menghubungkan ujung akhir kompleks WRS dengan bagian awal gelombang T. segmen ini mengukur waktu antara akhir depolarisasi ventrikel sampai pada mulainya repolarisasi ventrikel

    """)
    st.write("""Menggunakan Kolom (input) :

    
    """)
    st.write("""Memprediksi Pasien penyakit Gagal Jantung (output) :

    1. 1 yang berarti Diagnosa penyakit Gagal Jantung
    2. 0 yang berarti non diagnosa penyakit Gagal Jantung
    """)
    st.write("")

#Preprocessing
with preprocessing:
    st.subheader("""Normalisasi Data""")
    st.write(""" Rumus Normalisasi data :""")
    st.write("""

 Min-Max Normalization:
   Rumus Min-Max Normalization digunakan untuk mengubah nilai data ke dalam rentang tertentu, biasanya antara 0 dan 1. 
   Rumusnya adalah:
   
   X_norm = (X - X_min) / (X_max - X_min)
   
   di mana:
   - `X` adalah nilai data yang akan dinormalisasi.
   - `X_norm` adalah nilai data yang telah dinormalisasi.
   - `X_min` adalah nilai minimum dari data.
   - `X_max` adalah nilai maksimum dari data.

""")
    
    df = df.drop(columns=["Sex","ChestPainType","RestingECG","ExerciseAngina","ST_Slope"])
    #Mendefinisikan Varible X dan Y
    X = df.drop(columns=['HeartDisease'])
    y = df['HeartDisease'].values
    df_min = X.min()
    df_max = X.max()
    
    #NORMALISASI NILAI X
    scaler = MinMaxScaler()
    #scaler.fit(features)
    #scaler.transform(features)
    scaled = scaler.fit_transform(X)
    features_names = X.columns.copy()
    #features_names.remove('label')
    scaled_features = pd.DataFrame(scaled, columns=features_names)

    st.subheader('Hasil Normalisasi Data')
    st.write(scaled_features)


#Modelling
with modeling:
    training, test = train_test_split(scaled_features, test_size=0.2, random_state=1)
    training_label, test_label = train_test_split(y, test_size=0.2, random_state=1)

    # Naive Bayes
    gaussian = GaussianNB()
    gaussian = gaussian.fit(training, training_label)
    y_pred_gaussian = gaussian.predict(test)
    y_compare = np.vstack((test_label, y_pred_gaussian)).T
    gaussian.predict_proba(test)
    gaussian_accuracy = round(100 * accuracy_score(test_label, y_pred_gaussian))

    # K-NN
    K = 10
    knn = KNeighborsClassifier(n_neighbors=K)
    knn.fit(training, training_label)
    knn_predict = knn.predict(test)
    knn_accuracy = round(100 * accuracy_score(test_label, knn_predict))

    # Decision Tree
    dt = DecisionTreeClassifier()
    dt.fit(training, training_label)
    dt_pred = dt.predict(test)
    dt_accuracy = round(100 * accuracy_score(test_label, dt_pred))

    # Model ANN
    mlp = MLPClassifier(hidden_layer_sizes=(4,), max_iter=443, random_state=42)
    mlp.fit(training, training_label)
    mlp_pred = mlp.predict(test)
    mlp_accuracy = round(100 * accuracy_score(test_label, mlp_pred))

    with st.form("modeling"):
        st.subheader('Modeling')
        st.write("Pilihlah model yang akan dilakukan pengecekkan akurasi:")
        naive = st.checkbox('Gaussian Naive Bayes')
        k_nn = st.checkbox('K-Nearest Neighbor')
        destree = st.checkbox('Decision Tree')
        mlp = st.checkbox('ANN')
        submitted = st.form_submit_button("Submit")

        if submitted:
            if naive:
                st.write('Model Naive Bayes accuracy score: {0:0.2f}'.format(gaussian_accuracy))
            if k_nn:
                st.write("Model K-NN accuracy score: {0:0.2f}".format(knn_accuracy))
            if destree:
                st.write("Model Decision Tree accuracy score: {0:0.2f}".format(dt_accuracy))
            if mlp:
                st.write("Model ANN accuracy score: {0:0.2f}".format(mlp_accuracy))

       


# Implementasi
with implementation:
    with st.form("pendat_form"):
        st.subheader("Implementasi")
        Age = st.number_input('Masukkan Age (Umur) : ')
        RestingBP = st.number_input('Masukkan RestingBP (Sirkulasi darah) : ')
        Cholesterol = st.number_input('Masukkan Cholesterol (Kolesterol) : ')
        FastingBS = st.number_input('Masukkan FastingBS (Kadar gula darah) : ')
        MaxHR = st.number_input('Masukkan MaxHR (Max detak jantung) : ')
        Oldpeak = st.number_input('Masukkan Oldpeak (Tingkat stres) : ')
        model = st.selectbox('Pilihlah model yang akan anda gunakan untuk melakukan prediksi?',
                             ('Gaussian Naive Bayes', 'K-NN', 'Decision Tree'))

        submit_button = st.form_submit_button("Submit")
        if submit_button:
            inputs = np.array([
                Age,
                RestingBP,
                Cholesterol,
                FastingBS,
                MaxHR,
                Oldpeak
            ])

            input_norm = (inputs - df_min) / (df_max - df_min)
            input_norm = np.array(input_norm).reshape(1, -1)

            mod = None
           
            if model == 'Gaussian Naive Bayes':
                mod = gaussian
            elif model == 'K-NN':
                mod = knn
            elif model == 'Decision Tree':
                mod = dt

            if mod is not None:
                input_pred = mod.predict(input_norm)

                st.subheader('Hasil Prediksi')
                st.write('Menggunakan Pemodelan:', model)
                st.write('Hasil Diagnosa:', input_pred)
            else:
                st.write('Model belum dipilih')
