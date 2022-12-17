####################
##Görev 1: List Comprehension yapısı kullanarak car_crashes verisindeki numeric değişkenlerin isimlerini büyük harfe çeviriniz ve başına NUM ekleyiniz.
#Numeric olmayan değişkenlerin de isimleri büyümeli. Tek bir list comprehension yapısı kullanılmalı.

import pandas as pd
import seaborn as sns
df = sns.load_dataset("car_crashes")
df.head()
df.columns
df.dtypes

["NUM_" + col.upper() if df[col].dtype == "float64" else col.upper() for col in df.columns ]


####################
##Görev 2: List Comprehension yapısı kullanarak car_crashes verisinde isminde "no" barındırmayan değişkenlerin isimlerinin sonuna "FLAG" yazınız.
#Tüm değişkenlerin isimleri büyük harf olmalı. Tek bir list comprehension yapısı ile yapılmalı.

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

[col.upper() + "_FLAG" if "no" not in col else col.upper() for col in df.columns]


####################
##Görev 3: List Comprehension yapısı kullanarak aşağıda verilen değişken isimlerinden FARKLI olan değişkenlerin isimlerini seçiniz ve yeni bir dataframe oluşturunuz.
#Önce verilen listeye göre list comprehension kullanarak new_cols adında yeni liste oluşturunuz.
#Sonra df[new_cols] ile bu değişkenleri seçerek yeni bir df oluşturunuz ve adını new_df olarak isimlendiriniz.

og_list = ["abbrev", "no_previous"]

import seaborn as sns
df = sns.load_dataset("car_crashes")
df.columns

new_df = df[[col for col in df.columns if col not in og_list]]
new_df.head()

########################################################################################################################
# PANDAS ALIŞTIRMALAR
########################################################################################################################

import pandas as pd
import seaborn as sns
df = sns.load_dataset("titanic")

### 1.Kadın ve erkeklerin sayısını bulunuz.

df["sex"].value_counts()


### 2.Her bir sutuna ait unique değerlerin sayısını bulunuz.

df.nunique()


### 3.pclass değişkeninin unique değerlerinin sayısını bulunuz.

df["pclass"].nunique()


### 4.pclass ve parch değişkenlerinin unique değerlerinin sayısını bulunuz.

x = ["pclass", "parch"]
df[x].nunique()


### 5.embarked değişkeninin tipini kontrol ediniz. Tipini category olarak değiştiriniz ve tekrar kontrol ediniz.

df["embarked"].dtype
df["embarked"] = df["embarked"].astype("category")
df["embarked"].dtype


### 6.embarked değeri C olanların tüm bilgelerini gösteriniz.

df[df["embarked"] == "C"].head()


### 7.embarked değeri S olmayanların tüm bilgelerini gösteriniz.

df[df["embarked"] != "S"].head()


### 8.Yaşı 30'dan küçük ve kadın olan yolcuların tüm bilgilerini gösteriniz.

df.loc[(df["age"] < 30) & (df["sex" ] == "female")]


### 9.Fare'i 500'den büyük veya yaşı 70 den büyük yolcuların bilgilerini gösteriniz.

df.loc[(df["fare"] > 500) | (df["age"] > 70)]


### 10.Her bir değişkendeki boş değerlerin toplamını bulunuz.

df.isnull().sum()


### 11.who değişkenini dataframe’den çıkarınız.

df.drop("who", axis=1, inplace=True)


### 12.deck değişkenindeki boş değerleri deck değişkenin en çok tekrar eden değeri (mode) ile doldurunuz.

df["deck"].mode()[0]
df["deck"].fillna(df["deck"].mode()[0], inplace=True)
df["deck"].isnull().sum()


### 13.age değikenindeki boş değerleri age değişkenin medyanı ile doldurunuz.

df["age"].isnull().sum()
df["age"].fillna(df["age"].median(), inplace=True)


### 14.survived değişkeninin pclass ve cinsiyet değişkenleri kırılımınında sum, count, mean değerlerini bulunuz.

df.groupby(["pclass", "sex"]).agg({"survived" : ["sum", "count", "mean"]})


### 15.30 yaşın altında olanlar 1, 30'a eşit ve üstünde olanlara 0 vericek bir fonksiyon yazın.
#Yazdığınız fonksiyonu kullanarak titanik veri setinde age_flag adında bir değişken oluşturunuz. (apply ve lambda yapılarını kullanınız)

df["age_flag"] = df["age"].apply(lambda x : 1  if x <30 else 0)
df.head()


### 16.Seaborn kütüphanesi içerisinden Tips veri setini tanımlayınız.

import pandas as pd
import seaborn as sns
df = sns.load_dataset("tips")
df.head()


### 17.Time değişkeninin kategorilerine (Dinner, Lunch) göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.

df.groupby([ "time"]).agg({"total_bill" : ["sum", "min", "max", "mean"]})


### 18.Günlere ve time göre total_bill değerlerinin toplamını, min, max ve ortalamasını bulunuz.

df.groupby(["time", "day"]).agg({"total_bill" : ["sum", "min", "max", "mean"]})


### 20.Lunch zamanına ve kadın müşterilere ait total_bill ve tip değerlerinin day'e göre toplamını, min, max ve ortalamasını bulunuz.

df[(df["time"] == "Lunch") & (df["sex"] == "Female")].groupby("day").agg({"total_bill" : ["sum", "min", "max", "mean"],
                                                                          "tip" : ["sum", "min", "max", "mean"]})


### 21.size'i 3'ten küçük, total_bill'i 10'dan büyük olan siparişlerin ortalaması nedir? (loc kullanınız)

df.loc[(df["size"] < 3) & (df["total_bill"] > 10), "total_bill"].mean()


### 22.total_bill_tip_sum adında yeni bir değişken oluşturunuz. Her bir müşterinin ödediği total_bill ve tip'in toplamını versin.

df["total_bill_tip_sum"] = df["total_bill"] + df["tip"]
df.head()


### 23.total_bill_tip_sum değişkenine göre büyükten küçüğe sıralayınız ve ilk 30 kişiyi yeni bir dataframe'e atayınız.

new_df = df.sort_values("total_bill_tip_sum", ascending=False)[: 30]
new_df.shape











