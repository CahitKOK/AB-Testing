#--------------------------------------------------
#-- Temel İstatistik Kavramları
#--------------------------------------------------

import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# !pip install statsmodels
import statsmodels.stats.api as sms
from scipy.stats import ttest_1samp, shapiro, levene, ttest_ind, mannwhitneyu, \
    pearsonr, spearmanr, kendalltau, f_oneway, kruskal
from statsmodels.stats.proportion import proportions_ztest

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

#--------------------------------------------------
#-- Sampling (Örnekleme)
#--------------------------------------------------

populasyon = np.random.randint(0,80,10000)#0-80 arasinda 10000 tane sayi uret
##Bu populasyonu insanlarin yaslari olarak dusunebiliriz
populasyon.mean() #Ortalamasini aliyoruz. Out[6]: 39.1799

#Ornek kumesi bize sen buradaki 10000 kisinin istatiksel olarak temsil eden
##Belirli bir kumeyi al bu kume uzerinden genel icin cikarimlarda bulun

np.random.seed(115)

orneklem = np.random.choice(a=populasyon,size=100)
#Orneklem icin populasyon icinden 100 tane eleman sec anlamina gelmektedir.
orneklem.mean() # Ortalamasi Out[9]: 38.21
#Daha az veriyle genellemeler yapabilme imkani saglar

#Farz edelim 10 tane ornekleme cekmek istiyoruz.
orneklem1 = np.random.choice(a=populasyon, size=100)
orneklem2 = np.random.choice(a=populasyon, size=100)
orneklem3 = np.random.choice(a=populasyon, size=100)
orneklem4 = np.random.choice(a=populasyon, size=100)
orneklem5 = np.random.choice(a=populasyon, size=100)
orneklem6 = np.random.choice(a=populasyon, size=100)
orneklem7 = np.random.choice(a=populasyon, size=100)
orneklem8 = np.random.choice(a=populasyon, size=100)
orneklem9 = np.random.choice(a=populasyon, size=100)
orneklem10 = np.random.choice(a=populasyon, size=100)

(orneklem1.mean() + orneklem2.mean() + orneklem3.mean() + orneklem4.mean() + orneklem5.mean()
 + orneklem6.mean() + orneklem7.mean() + orneklem8.mean() + orneklem9.mean() + orneklem10.mean()) / 10
#Burada 10 ayri ornekleminde ortalamasina bakarsak
## Orneklemelerin artmasi bizim genel ortlamamiza yaklasmamizi saglar.

#--------------------------------------------------
#-- Descriptive Statistics (Betimsel İstatistikler)
#--------------------------------------------------

df=sns.load_dataset("tips")
df.describe().T

#Sayisal degiskenlere bakarak degiskenlerin dagilimi hakkinda cikarimlarda bulunabiliriz.
#Genelde bu cikarimlari ortalama ve çeyrekteki degerlere bakarak bu cikarimlari yapariz.
##Hangi degeri temsil ederken neleri esas almaliyiz
### Ortalamami yoksa standart sapmami daha güvenilir olur bunlari incelemeliyiz.
#Bu yapilan inceleme yorumlarin sonucundaki cikarimlarin hepsine betimsel istatikseklerden yapmaktayiz.


#--------------------------------------------------
#-- Confidence Intervals (Güven Aralıkları)
#--------------------------------------------------

#Güven araligi
#Ortalama +,- guven aralıgı (95,99)*standartsapma/(Kaç eleman)^1/2
##Guven araligi genel olarak 95 alinir ve tablodan 1,96 degerine karsilik gelir

df = sns.load_dataset("tips")
df.describe().T
df.head()
#Farzedelim tips verisi elinde olan bir restorantin işlerinin en kotu ayda ne kadar para kazanacagi
## gibi ongorulerde bulunmak ister. Bu ongoruler yorumlar dogrultusunda onlemler almak ister.
###İste burada guven araligini kullanabiliriz.

sms.DescrStatsW(df["total_bill"]).tconfint_mean()
#Out[8]: (18.66333170435847, 20.908553541543164)
#Burada yapacagimiz yorum restorana gelen musterilerin %95 guvenle
## islem basina birakacagi hesap 18.6 ile 20.9 arasindadir.

#Farz edelim elimizde yillik veride var. Buradan aylik islem sayisinin guven araligini bularak
## Aylik ne kadar para gelecebilecegini buradan hesaplayarak
### Is yerimiz ile ilgili onlemler alabiliriz.

sms.DescrStatsW(df["tip"]).tconfint_mean()#Bahsislerin guven araligi

df = sns.load_dataset("titanic")
sms.DescrStatsW(df["age"].dropna()).tconfint_mean()
#titanic degiskeninde yas araliginin guven araligina bakiyoruz.
##Bos degerlerin sorun cikarmamasi icin cikariyoruz

sms.DescrStatsW(df["fare"].dropna()).tconfint_mean()
df.describe().T

#--------------------------------------------------
#-- Correlation( Korelasyon)
#--------------------------------------------------

#1'e dogru gidildikçe mmükemmel pozitif kolerasyon
## -1'e dogru gidildikce mükemmel negatif kolerasyon

#Mükemmel pozitif kolerasyon ise X1,X2 degiskenlerinden birinin degeri artarken
## diger degiskenininde degerinin artmasidir.
### Kidem arttikca maasin artmasi ornek olarak verilebilir.

#Mükemmel negatif kolerasyon ise X1,X2 degiskenlerinden birinin degeri artarken
## diger degiskenininde degerinin azalmasidir.
### Bir bilgisayarin yıl geçtikçe perfonmasin düşmesi ornek olarak verilebilir.


# Bahşiş veri seti:
# total_bill: yemeğin toplam fiyatı (bahşiş ve vergi dahil)
# tip: bahşiş
# sex: ücreti ödeyen kişinin cinsiyeti (0=male, 1=female)
# smoker: grupta sigara içen var mı? (0=No, 1=Yes)
# day: gün (3=Thur, 4=Fri, 5=Sat, 6=Sun)
# time: ne zaman? (0=Day, 1=Night)
# size: grupta kaç kişi var?

df = sns.load_dataset('tips')
df.head()

df["total_bill"] = df["total_bill"] - df["tip"]

df.plot.scatter("tip","total_bill")
plt.show()
#bu grafikte gorecegimiz uzere pozitif bir iliski oldugu gozukmektedir.
##Bunun matematiksel olarak ifadesini ogrenmek icin

df["tip"].corr(df["total_bill"])
#:: Out[19]: 0.5766634471096382
#Görülecegi uzere 0.5 den yuksek ve pozitif bir korelasyon vardir.


#--------------------------------------------------
#-- AB Testing (Bağımsız İki Örneklem T Testi)
#--------------------------------------------------

# 1. Hipotezleri Kur
# 2. Varsayım Kontrolü
#   - 1. Normallik Varsayımı
#   - 2. Varyans Homojenliği
# 3. Hipotezin Uygulanması
#   - 1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
#   - 2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)
# 4. p-value değerine göre sonuçları yorumla
# Not:
# - Normallik sağlanmıyorsa direk 2 numara. Varyans homojenliği sağlanmıyorsa 1 numaraya arguman girilir.
# - Normallik incelemesi öncesi aykırı değer incelemesi ve düzeltmesi yapmak faydalı olabilir.


#--------------------------------------------------
# Uygulama 1: Sigara İçenler ile İçmeyenlerin Hesap Ortalamaları Arasında İstatistik olarak fark var mı?
#--------------------------------------------------

df= sns.load_dataset("tips")
df.head()
df.groupby("smoker").agg({"total_bill" : "mean"}) # Sigara içenleri gruplaştırarak
## toplam hesaplarin ortalmasini inceliyoruz.

#1.Hipotezi Kur
# H0 : M1=M2 #Musterilerin odeyecegi hesap arasinda fark yoktur hipotezi
# H1 : M1!=M2 # Musterilerin odeyecegi hesaplar arasinda fark vardir.

#2. Varsayım Kontrolu
# Normallik Varsayımı
# Varyans Homojenliği

#H0: Normal dağılım varsayımı sağlanmaktadır.
#H1: .... sağlanmamaktadır.


#----------------------------
#-- # Normallik Varsayımı
#----------------------------

test_stat,pvalue = shapiro(df.loc[df["smoker"]== "Yes","total_bill"])
print("Test Stat = %.4f, p-value = %.4f"%(test_stat,pvalue))
#Bu islemle normallik varsayımı testimizi shapiro ile kontrolunu yapiyoruz.
## Birinci bolumunde "df.loc[df["smoker"]== "Yes"" ilgili bölümü tanimliyoruz.
### İkinci bölümünde ise ilgili degiskeni tanimliyoruz. ""total_bill"" gibi

#p-value degeri < 0.05 ise H0 direkt olarak red edilir.
#p-value degeri < 0.05 değilse H0 rededilemez
#Yani buradan cikarcagimiz sonuç normal dağılım varsayımı sağlanmamaktadır olmalıdır.

test_stat,pvalue = shapiro(df.loc[df["smoker"]== "No","total_bill"])
print("Test Stat = %.4f, p-value = %.4f"%(test_stat,pvalue))
#Sigara içmeyenlerin hesap üzerindeki etkisini inceleyecek olursak
## p-value degeri Test Stat = 0.9045, p-value = 0.0000 olduğundan
### Normal dağılım varsayımı sağlanmamaktadır yorumunu yapabiliriz.

#Bu durumda varsayim sağlanmadığı için Non-parametrik bir test kullanmalıyız.
## Bu durumu simdi bir kere bakıp varsayim saglanmış gibi işleme devam edeceğiz.
#Bu normallik varsayimi bize olayi şansa bırakmadan istatiktik olarak yorum yapmamızı sağlar


#----------------------------
#-- Varyans Homojenligi
#----------------------------

# H0 : Varyanslar homojendir
# H1 : Varyanslar homojen değildir.

test_stat,pvalue = levene(df.loc[df["smoker"]== "Yes","total_bill"],
                          df.loc[df["smoker"]== "No","total_bill"])
print("Test Stat = %.4f, p-value = %.4f"%(test_stat,pvalue))
#Homojenliğe bakmak için ise leveneyi kullaniriz.
## levene bizden iki degeri birden ister.

#p-value degeri < 0.05 ise H0 direkt olarak red edilir.
#p-value degeri < 0.05 değilse H0 rededilemez
#p-value degeri 0.0452 olduğu içi H0 rededilir.
##Yani varyanslar homojen değildir.


#----------------------------
#-- 3. ve 4. Hipotezin Uygulanması
#----------------------------

#   1. Varsayımlar sağlanıyorsa bağımsız iki örneklem t testi (parametrik test)
#   2. Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)

#Normalde yukaridada ifade ettigimiz gibi varsayimlar saglanmadi fakat sağlıyormuş gibi işlem yapiyoruz.
#-- 1. Varsayimlar sağlandı bağımsız iki örneklem t testi

test_stat,pvalue = ttest_ind(df.loc[df["smoker"]== "Yes","total_bill"],
                             df.loc[df["smoker"]== "No","total_bill"],
                             equal_var=True)
#Normallik ve Varyans homojenliği sağlandığı durumlarda equal_var=True
##Normallik sağlandi ve Varyans homojenliği sağlanmadığı durumlarda equal_var=False
###olarak kullanılmalıdır.

print("Test Stat = %.4f, p-value = %.4f"%(test_stat,pvalue))
# Out Test Stat = 1.3384, p-value = 0.1820
## pvalue 0.05 den fazla oldugu için sigara içenler ile içmeyenlerin arasindaki hesap ödeme
###durumuna göre istatiksel olarak fark yoktur deriz.

#Onemli Not: Biz burada H0 durumunu inceleriz
## H0 red ettigimizde bu bize H1 kabul edildigi anlamina gelmez!!!

#---------------------------------------------
# 1.2 Varsayımlar sağlanmıyorsa mannwhitneyu testi (non-parametrik test)
#---------------------------------------------

test_stat, pvalue = mannwhitneyu(df.loc[df["smoker"] == "Yes", "total_bill"],
                                 df.loc[df["smoker"] == "No", "total_bill"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

#Hipotezi kurduğumuzda bu hipotezin yorumunu yanina yazmak faydamiza olacaktir.

############################
# Uygulama 2: Titanic Kadın ve Erkek Yolcuların Yaş Ortalamaları Arasında İstatistiksel Olarak Anl. Fark. var mıdır?
############################

df = sns.load_dataset("titanic")
df.head()

df.groupby("sex").agg({"age": "mean"})


# 1. Hipotezleri kur:
# H0: M1  = M2 (Kadın ve Erkek Yolcuların Yaş Ortalamaları Arasında İstatistiksel Olarak Anl. Fark. Yoktur)
# H1: M1! = M2 (... vardır)


# 2. Varsayımları İncele

# Normallik varsayımı
# H0: Normal dağılım varsayımı sağlanmaktadır.
# H1:..sağlanmamaktadır

test_stat,pvalue = shapiro(df.loc[df["sex"]=="female","age"].dropna())
print("Test stat = %.4f, p-value = %.4f" %(test_stat,pvalue))
#Buradan çıkan sonuç H0 Test stat = 0.9848, p-value = 0.0071 degeri 0.05'den kucuk oldugu için rededilir.

test_stat,pvalue = shapiro(df.loc[df["sex"]=="male","age"].dropna())
print("Test stat = %.4f, p-value = %.4f" %(test_stat,pvalue))
#Buradanda çıkan sonuç H0 Test stat = 0.9747, p-value = 0.0000 degeri 0.05'den kucuk oldugu için rededilir.

#Onemli not: Bu kisimda normal dağılım varsayımı sağlanmadıysa
## direkt olarak non-parametrik teste gidilmesi gerekir.
### biz daha iyi pekiştirmek adına varyans homojenliğinede bakıyoruz.

#Varyans Homojenliği
#H0: Varyanslar Homojendir
#H1 :değildir.

test_stat,pvalue = levene(df.loc[df["sex"]=="female","age"].dropna(),
                          df.loc[df["sex"]=="male","age"].dropna())
print("Test stat = %.4f, p-value = %.4f" %(test_stat,pvalue))
# Test stat = 0.0013, p-value = 0.9712 pvalue degeri 0.05 den buyuk oldugu icin
##Normal dagılımında rededilmedigini dusunursek
#test_stat,pvalue = ttest_ind(df.loc[df["smoker"]== "Yes","total_bill"],
#                            df.loc[df["smoker"]== "No","total_bill"],
#                            equal_var=True) islemini yapmamiz gerekirdi.
## unutulmamasi gereken nokta normal dagılım ve varyans homojenligi kabul edilirse
### equal_var=True normal dagılım var varyans homojenligi yoksa False olmalıdır.

#Biz normal dagılımın saglanmadıgı için bu yoldan devam edecek olursak

test_stat,pvalue = mannwhitneyu(df.loc[df["sex"]=="female","age"].dropna(),
                                df.loc[df["sex"]=="male","age"].dropna())
print("Test stat = %.4f, p-value = %.4f" %(test_stat,pvalue))

#Test stat = 53212.5000, p-value = 0.0261 H0 redederiz.


############################
# Uygulama 3: Diyabet Hastası Olan ve Olmayanların Yaşları Ort. Arasında İst. Ol. Anl. Fark var mıdır?
############################

df = pd.read_csv("datasets/diabetes.csv")
df.head()

df.groupby("Outcome").agg({"Age": "mean"})

# 1. Hipotezleri kur
# H0: M1 = M2
# Diyabet Hastası Olan ve Olmayanların Yaşları Ort. Arasında İst. Ol. Anl. Fark Yoktur
# H1: M1 != M2
# .... vardır.

# 2. Varsayımları İncele

# Normallik Varsayımı (H0: Normal dağılım varsayımı sağlanmaktadır.)
test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 1, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = shapiro(df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))



# Normallik varsayımı sağlanmadığı için nonparametrik.

# Hipotez (H0: M1 = M2)
test_stat, pvalue = mannwhitneyu(df.loc[df["Outcome"] == 1, "Age"].dropna(),
                                 df.loc[df["Outcome"] == 0, "Age"].dropna())
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


###################################################
# İş Problemi: Kursun Büyük Çoğunluğunu İzleyenler ile İzlemeyenlerin Puanları Birbirinden Farklı mı?
###################################################

# H0: M1 = M2 (... iki grup ortalamaları arasında ist ol.anl.fark yoktur.)
# H1: M1 != M2 (...vardır)

df = pd.read_csv("datasets/course_reviews.csv")
df.head()

df[(df["Progress"] > 75)]["Rating"].mean()

df[(df["Progress"] < 25)]["Rating"].mean()


test_stat, pvalue = shapiro(df[(df["Progress"] > 75)]["Rating"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


test_stat, pvalue = shapiro(df[(df["Progress"] < 25)]["Rating"])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

test_stat, pvalue = mannwhitneyu(df[(df["Progress"] > 75)]["Rating"],
                                 df[(df["Progress"] < 25)]["Rating"])

print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


######################################################
# AB Testing (İki Örneklem Oran Testi)
######################################################

# H0: p1 = p2
# Yeni Tasarımın Dönüşüm Oranı ile Eski Tasarımın Dönüşüm Oranı Arasında İst. Ol. Anlamlı Farklılık Yoktur.
# H1: p1 != p2
# ... vardır

basari_sayisi = np.array([300, 250])
gozlem_sayilari = np.array([1000, 1100])

proportions_ztest(count=basari_sayisi, nobs=gozlem_sayilari)


basari_sayisi / gozlem_sayilari


############################
# Uygulama: Kadın ve Erkeklerin Hayatta Kalma Oranları Arasında İst. Olarak An. Farklılık var mıdır?
############################

# H0: p1 = p2
# Kadın ve Erkeklerin Hayatta Kalma Oranları Arasında İst. Olarak An. Fark yoktur

# H1: p1 != p2
# .. vardır

df = sns.load_dataset("titanic")
df.head()

df.loc[df["sex"] == "female", "survived"].mean()

df.loc[df["sex"] == "male", "survived"].mean()

female_succ_count = df.loc[df["sex"] == "female", "survived"].sum()
male_succ_count = df.loc[df["sex"] == "male", "survived"].sum()

test_stat, pvalue = proportions_ztest(count=[female_succ_count, male_succ_count],
                                      nobs=[df.loc[df["sex"] == "female", "survived"].shape[0],
                                            df.loc[df["sex"] == "male", "survived"].shape[0]])
print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

