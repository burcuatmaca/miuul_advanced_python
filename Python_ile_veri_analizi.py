######################################################################
# GELİŞMİŞ FONKSİYONEL KEŞİFÇİ VERİ ANALİZİ (ADVANCED FUNCTIONAL EDA)
######################################################################

#######################################################################
# 1. GENEL RESİM
#######################################################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

df.tail()
df.shape
df.info()
df.columns
df.index
df.describe().T
df.isnull().values.any()
df.isnull().sum()
########################################################################

def check_df(dataframe, head=5):
    print("############ Shape ############")
    print(dataframe.shape)
    print("############ dtypes ############")
    print(dataframe.dtypes)
    print("############ Head ############")
    print(dataframe.head(head))
    print("############ Tail ############")
    print(dataframe.tail(head))
    print("############ NA ############")
    print(dataframe.isnull().sum())
    print("############ Quantiles ############")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)

########################################################################

df = sns.load_dataset("tips")
check_df(df)


#######################################################################
# 2. KATEGORİK DEĞİŞKEN ANALİZİ (ANALYSİS OF CATEGORİCAL VARİABLES)
#######################################################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

df["embarked"].value_counts()
df["sex"].unique()  #eşsiz değerleri.
df["sex"].nunique() #toplam kaç tane eşsiz değeri var.

# Bütün olası kategorik değişkenleri seçmek istediğimde; (type bilgisine göre)

df.dtypes

# -Değişkenler içerisinde gezip olası kategorik değişkenleri seçmek istersek;
cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]

# -Type'i integer ya da float olup eşsiz sınıf sayısı (unique) belirli bir değerden küçük olanları seçmek istersek;
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]

# -type'i object ya da category olduğu halde unique değerleri çok yüksek olanları ayırt etmek istersek;
# (örneğin isim değişkeni, type'i category olsa da ölçülemez olduğundan category değişkeni değildir.)
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]


cat_cols = cat_cols + num_but_cat #bütün kategorik değişkenler artık elimizde.

cat_cols = [col for col in cat_cols if col not in cat_but_car]

df[cat_cols].nunique()

# Sayısal değişkenler ise;
[col for col in df.columns if col not in cat_cols]


# Kendisine girilen değerlerin kaç sınıfı olduğunu ve yüzdeliklerini gösteren fonksiyon;

df["survived"].value_counts()
100 * df["survived"].value_counts() / len(df)


def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name : dataframe[col_name].value_counts(),
                        "Ratio" : 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#########################################")

cat_summary(df,"sex")

for col in cat_cols:
    cat_summary(df, col)

# 2.1 : cat_summary fonksiyonuna grafik özelliğini de ekleyelim.

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name : dataframe[col_name].value_counts(),
                        "Ratio" : 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("#########################################")

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

cat_summary(df, "sex", plot=True)

# Yukarıdaki fonksiyonumuzu bütün değişkenleri görselleştirmek için kullanalım.

for col in cat_cols:
    cat_summary(df, col, plot=True)

# adult_male değişkenimizde hata verecek, çünkü type'i bool. Çözelim: (type'ı bool olunca "asdfghjklş" yaz sonra devam et dedik.

for col in cat_cols:
    if df[col].dtypes == "bool":
        print("asdfghjklş")
    cat_summary(df, col, plot=True)

# Başka nasıl bool type'ımızı değiştirebiliriz? ( type'ı bool ise type'ı düzeltip görselleştirdik.

for col in cat_cols:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)
        cat_summary(df, col, plot=True)
    else:
        cat_summary(df, col, plot=True)




#######################################################################
# 3. SAYISAL DEĞİŞKEN ANALİZİ
#######################################################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

df[["age","fare"]].describe().T

num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]



cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
num_but_cat = [col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in ["int", "float"]]
cat_but_car = [col for col in df.columns if df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
cat_cols = cat_cols + num_but_cat
cat_cols = [col for col in cat_cols if col not in cat_but_car]



num_cols = [col for col in num_cols if col not in cat_cols]


def num_summary(dataframe, numerical_col):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

num_summary(df, "age")


for col in num_cols:
    num_summary(df, col)


def num_summary(dataFrame, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataFrame[numerical_col].describe(quantiles).T)

    if plot:
        dataFrame[numerical_col].hist()
        plot.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

num_summary(df, "age", plot=True)

for col in num_cols:
    num_summary(df, col, plot=True)



#######################################################################
# 4. DEĞİŞKENLERİN YAKALANMASI VE İŞLEMLERİN GENELLEŞTİRİLMESİ
#######################################################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.wwidth", 500)
df = sns.load_dataset("titanic")
df.head()
df.info()

# Şu an yazacağımız fonksiyon yukarıda yaptığımız işlemlerin fonksiyonlaştırılmış halidir.
# Öyle bir işlem yapmalıyız ki veri setindeki kategorik ve numeric değişkenleri,kategorik ama kardinal değişken listesini versin.
# Docstring de yazıcaz. (yani fonksiyonumuzun açıklaması)
# docstringine ulaşmak için : help(grab_col_names)
# Soldaki + kutucuğuna basarak fonksiyonumuza ulaşabiliriz.

def grab_col_names(dataframe, cat_th= 10, car_th= 20):

    """
    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin ismini verir.

    Parameters
    ----------
    dataframe: dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th: int, float
        numerik fakat kategorik olan değişkenler için sınıf eşik değeridir.
    car_th: int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeridir.

    Returns
    -------
    cat_cols: list
        Kategorik değişken listesi
    num_cols: list
        Numerik değişken listesi
    cat_but_car: list
        Kategorik görünümlü kardinal değişken listesi

    Notes
    -----
    cat_cols + num_cols + cat_but_car = Toplam değişken sayısı
    num_but_cat, cat_cols'un içerisinde
    """

# cat_cols, cat_but_car (kategorik değişken analizi)

    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [ col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in["int", "float"]]
    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

# num_cols (numerik değişken analizi)

    num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

    print(f"observations: {dataframe.shape[0]}")
    print(f"variables: {dataframe.shape[1]}")
    print(f"cat_cols: {len(cat_cols)}")
    print(f"num_cols: {len(num_cols)}")
    print(f"cat_but_car: {len(cat_but_car)}")
    print(f"num_but_cat: {len(num_but_cat)}")

    return cat_cols, num_cols, cat_but_car

# Fonksiyonumu datasetimde uyguladım ve değişkenlerimi tuttum.

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Şimdi de yukarıda yazdığımız tüm fonksiyonları getirip bu fonksiyonumuzla birlikte çalıştıralım.
# ilk fonksiyonumuz:

def cat_summary(dataframe, col_name):
    print(pd.DataFrame({col_name : dataframe[col_name].value_counts(),
                        "Ratio" : 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("############################")

cat_summary(df,"sex")


for col in cat_cols:
    cat_summary(df,col)

# ikinci fonksiyonumuz:


def num_summary(dataFrame, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataFrame[numerical_col].describe(quantiles).T)

    if plot:
        dataFrame[numerical_col].hist()
        plot.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)


# BONUS: Amacımız bool type'teki değişkenleri bulup bunları integer'a çevirmek,
# daha sonra cat_summary fonksiyonunuz görsel özellikleri olarak kullanmaya çalışacağız.

df = sns.load_dataset("titanic")
df.info()

for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Yukarıdaki grafik özellikli cat_summary fonksiyonumu getiriyorum, bütün değişkenleri görselleştiriyorum.

def cat_summary(dataframe, col_name, plot= False):
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts()}))
        print("#########################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)

# cat_summary'i görselleştirdik:

for col in cat_cols:
    cat_summary(df, col, plot=True)

# num_summary'i görselleştirdik:

for col in num_cols:
    num_summary(df, col, plot=True)


#######################################################################
# 5. HEDEF DEĞİŞKEN ANALİZİ (ANALYSİS OF TARGET VARIABLE)
#######################################################################

import numpy as np
import pandas as pd
import seaborn as sns
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = sns.load_dataset("titanic")
df.head()

for col in df.columns:
    if df[col].dtypes == "bool":
        df[col] = df[col].astype(int)


def cat_summary(dataframe, col_name, plot= False):
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts()}))
        print("#########################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)

def grab_col_names(dataframe, cat_th= 10, car_th= 20):
    cat_cols = [col for col in df.columns if str(df[col].dtypes) in ["category", "object", "bool"]]
    num_but_cat = [ col for col in df.columns if df[col].nunique() < 10 and df[col].dtypes in["int", "float"]]
    cat_but_car = [col for col in df.columns if
                   df[col].nunique() > 20 and str(df[col].dtypes) in ["category", "object"]]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]


    num_cols = [col for col in df.columns if df[col].dtypes in ["int", "float"]]
    num_cols = [col for col in num_cols if col not in cat_cols]

cat_cols, num_cols, cat_but_car = grab_col_names(df)

df["survived"].value_counts()
cat_summary(df, "survived")


#####################################################
# Hedef Değişkenin Kategorik Değişkenler ile Analizi
#####################################################

# Problemimiz: survived değişkeni nelere bağlı? (diğer değişkenleri survived değişkenlerine göre inceliyoruz.)

df.groupby("sex")["survived"].mean()

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}),)


target_summary_with_cat(df, "survived", "pclass")

for col in cat_cols:
    target_summary_with_cat(df, "survived",col)


#####################################################
# Hedef Değişkenin Sayısal Değişkenler ile Analizi
#####################################################

df.groupby("survived")["age"].mean()

df.groupby("survived").agg({"age": "mean"})

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

target_summary_with_num(df, "survived", "age")

for col in num_cols:
    target_summary_with_num(df, "survived", col)

#######################################################################
# KORELASYON ANALİZİ (ANALYSİS OF CORRELATİON)
#######################################################################

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)
df = pd.read_csv("breast_cancer.csv")
df = df.iloc[:, 1:-1]  # Problemli değişkenleri dışarıda bıraktık. "id" ve "unnamed" istemediğimiz değişkenlerdi.
df.head()

# Numerik değişkenlerimizi seçiyoruz.
num_cols = [col for col in df.columns if df[col].dtype in [int,float]]

# Değişkenlerin birbirleriyle olan orelasyonlarını hesaplıyoruz.
corr = df[num_cols].corr()

# Isı haritasını oluşturalım.

sns.set(rc={"figure.figsize": (12,12)})
sns.heatmap(corr, cmap="RdBu")
plt.show()

#############################################
# Yüksek Korelasyonlu Değişkenlerin Silinmesi
#############################################

# Şu an için korelasyonun negatif veya pozitif olmasının bir anlamı olmadığı için mutlak değer fonksiyonundan  geçiriyoruz.
cor_matrix = df.corr().abs()

# Aynı değişkenler hem satırlarda hem sütunlarda gözüktüğü için onları temizliyoruz.
upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))

# Korelasyon değeri 0.90 olan değişkenlerin herhangi birini siliyoruz.
# Çünkü bazı analizler yaparken korelasyonun o kadar yüksek olması iki değişkeninde aynı etkiyi yarattığını gösterir.
# Bu yüzden bazı problemlerde o değişkenlerden birini silebiliriz.

upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
drop_list = [col for col in upper_triangle_matrix.cloumns if any(upper_triangle_matrix[col] > 0.90)]

cor_matrix[drop_list]
df.drop(drop_list, axis=1)


# Yukarıda yaptığımız işlemleri fonksiyonlaştırıyoruz.

def high_correlated_cols(dataframe, plot=False, corr_th=0.90):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={"figure.figsize": (15,15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

high_correlated_cols(df)
drop_list = high_correlated_cols(df, plot=True)
df.drop(drop_list, axis=1)
high_correlated_cols(df.drop(drop_list, axis=1), plot=True)










