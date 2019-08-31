import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy import stats
import json
from sklearn.feature_selection import chi2
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import ExtraTreesClassifier
import datetime
import warnings
warnings.filterwarnings("ignore")

class FeatureSelection:

    def __init__(self, dataset, datatypes):
        with open(datatypes, 'r') as load_f:
            self.datatypes = json.load(load_f)

        self.ignore = {}
        self.ignore_list = []
        self.nominal = []
        self.real = []
        self.multi_plain = []
        self.date = []
        self.natural_language_text = []

        for key in self.datatypes.keys():
            if self.datatypes[key] == 'Nominal':
                self.nominal.append(key)

            elif self.datatypes[key] == 'Real':
                self.real.append(key)

            elif self.datatypes[key] == 'Multi_plain':
                self.multi_plain.append(key)
                self.ignore_list.append(key)
                self.ignore[key] = 'Multi_plain feature does not provide any information.'

            elif self.datatypes[key] == 'Date':
                self.date.append(key)

            elif self.datatypes[key] == 'Natural_language_text':
                self.natural_language_text.append(key)

            else:
                self.label = key

        try:
            self.df = pd.read_csv(dataset, low_memory=False)
            # self.df = pd.read_csv( dataset ,sep='\t', header = 0, low_memory=False)
        except:
            self.df = pd.read_excel(dataset)

        self.X_NOMINAL = self.df[self.nominal].copy()
        self.X_REAL = self.df[self.real].copy()
        self.X_DATE = self.df[self.date].copy()
        self.X_MULTI = self.df[self.multi_plain].copy()
        self.X_NATURAL = self.df[self.natural_language_text].copy()
        self.Y = self.df[self.label].copy()
        self.X = self.df.iloc[:, self.df.columns != self.label].copy()
        for multi in self.multi_plain:
            del self.X[multi]

        self.yvar = {}
        n = 0
        for i in self.Y.unique().tolist():
            self.yvar[i] = n
            n += 1
        self.Y = self.Y.replace(self.yvar)
        self.Y = self.Y.astype('int64')

        if len(self.date) == 0:
            pass

        elif len(self.date) == 1:
            if self.X_DATE[self.date[0]].dtype == 'int64':
                self.X_DATE[self.date[0]] = self.X_DATE[self.date[0]].apply(
                    lambda x: datetime.datetime.fromtimestamp(x / 1000))
                self.X_DATE[self.date[0]] = pd.to_datetime(self.X_DATE[self.date[0]])
                self.X_DATE[str(self.date[0]) + '_day_of_month'] = self.X_DATE[self.date[0]].apply(lambda x: x.day)
                self.X_DATE[str(self.date[0]) + '_day_of_week'] = self.X_DATE[self.date[0]].apply(lambda x: x.weekday())
                self.X_DATE[str(self.date[0]) + '_week_of_year'] = self.X_DATE[self.date[0]].dt.week
                del self.X_DATE[self.date[0]]
            elif self.X_DATE[self.date[0]].dtype == 'O':
                self.X_DATE[self.date[0]] = pd.to_datetime(self.X_DATE[self.date[0]])
                self.X_DATE[str(self.date[0]) + '_day_of_month'] = self.X_DATE[self.date[0]].apply(lambda x: x.day)
                self.X_DATE[str(self.date[0]) + '_day_of_week'] = self.X_DATE[self.date[0]].apply(lambda x: x.weekday())
                self.X_DATE[str(self.date[0]) + '_week_of_year'] = self.X_DATE[self.date[0]].dt.week
                self.X_DATE[self.date[0]]
                del self.X_DATE[self.date[0]]
            else:
                print('ERROR: DATE class should have a \'int64\' or \'object\' type.')
                del self.X_DATE[self.date[0]]
            print(
                'Your features with \'Date\' class have been transformed into three columns including : \'day_of_month\', \'day_of_week\' and \'week_of_year\'')

        elif len(self.date) > 1:
            for col in self.date:
                if self.X_DATE[col].dtype == 'int64':
                    self.X_DATE[col] = self.X_DATE[col].apply(lambda x: datetime.datetime.fromtimestamp(x / 1000))
                    self.X_DATE[col] = pd.to_datetime(self.X_DATE[col])
                    self.X_DATE[str(col) + '_day_of_month'] = self.X_DATE[col].apply(lambda x: x.day)
                    self.X_DATE[str(col) + '_day_of_week'] = self.X_DATE[col].apply(lambda x: x.weekday())
                    self.X_DATE[str(col) + '_week_of_year'] = self.X_DATE[col].dt.week
                    del self.X_DATE[col]
                elif self.X_DATE[col].dtype == 'O':
                    self.X_DATE[col] = pd.to_datetime(self.X_DATE[col])
                    self.X_DATE[str(col) + '_day_of_month'] = self.X_DATE[col].apply(lambda x: x.day)
                    self.X_DATE[str(col) + '_day_of_week'] = self.X_DATE[col].apply(lambda x: x.weekday())
                    self.X_DATE[str(col) + '_week_of_year'] = self.X_DATE[col].dt.week
                    del self.X_DATE[col]
                else:
                    del self.X_DATE[col]
                    print('ERROR: DATE class should have a \'int64\' or \'object\' type.')
            print(
                'Your features with \'Date\' class have been transformed into three columns including : \'day_of_month\', \'day_of_week\' and \'week_of_year\'')

        self.X_IG = pd.concat([self.X_REAL, self.X_NOMINAL, self.X_DATE, self.X_NATURAL, self.X_MULTI], axis=1).copy()

    def select_missing(self):
        for i in self.X:
            if self.X[i].isnull().sum() / self.X.shape[0] * 100 > 99:
                self.ignore[i + '[0]'] = 'This feature has more than 99% missing data.'
                self.ignore_list.append(i)
                if self.datatypes[i] == 'Nominal':
                    del self.X_NOMINAL[i]
                    del self.X[i]

                elif self.datatypes[i] == 'Real':
                    del self.X_REAL[i]
                    del self.X[i]

                elif self.datatypes[i] == 'Date':
                    del self.X[i]

                elif self.datatypes[i] == 'Natural_language_text':
                    del self.X_NATURAL[i]
                    del self.X[i]

        for i in self.X_DATE:
            if self.X_DATE[i].isnull().sum() / self.X.shape[0] * 100 > 99:
                self.ignore_list.append(i)
                self.ignore[i + '[0]'] = 'This feature has more than 99% missing data.'
                del self.X_DATE[i]

    def select_lowvariance(self):
        for i in self.X:
            if self.X[i].value_counts().max() / self.X.shape[0] > 0.95:
                self.ignore_list.append(i)
                self.ignore[i + '[1]'] = 'The VAR of this feature is very low.'
                if self.datatypes[i] == 'Nominal':
                    del self.X_NOMINAL[i]
                    del self.X[i]

                elif self.datatypes[i] == 'Real':
                    del self.X_REAL[i]
                    del self.X[i]

                elif self.datatypes[i] == 'Date':
                    del self.X[i]

                elif self.datatypes[i] == 'Natural_language_text':
                    del self.X_NATURAL[i]
                    del self.X[i]

        for i in self.X_DATE:
            if self.X_DATE[i].value_counts().max() / self.X.shape[0] > 0.95:
                self.ignore_list.append(i)
                self.ignore[i + '[1]'] = 'The VAR of this feature is very low.'
                del self.X_DATE[i]

    def select_pearson(self):
        pearsondict = {}
        pearsondict_p = {}
        X_PEA = self.X_REAL.copy()
        for col in self.X_DATE:
            X_PEA[col] = self.X_DATE[col]
        X_PEA = X_PEA.astype('float64')
        X_PEA_fillna = X_PEA.fillna(X_PEA.mean()[:])
        for i in range(0, X_PEA_fillna.shape[1]):
            if len(X_PEA_fillna.iloc[:, i].unique()) == 1:
                pearsondict[X_PEA_fillna.columns[i]] = 0
                pearsondict_p[X_PEA_fillna.columns[i]] = 1
            else:
                pearsondict[X_PEA_fillna.columns[i]] = abs(pearsonr(X_PEA_fillna.iloc[:, i], self.Y)[0])
                pearsondict_p[X_PEA_fillna.columns[i]] = pearsonr(X_PEA_fillna.iloc[:, i], self.Y)[1]

        # nlist = []
        # for key in pearsondict:
        #    nlist.append(abs(pearsondict[key]))
        # narray = abs(np.array(nlist))
        # sum1 = narray.sum()
        # mean = sum1/len(nlist)
        # std = np.std(narray)
        # interval = stats.t.interval(0.90,len(nlist)-1,mean,std)

        for key in pearsondict:
            if pearsondict_p[key] > 0.05:
                self.ignore_list.append(key)
                self.ignore[
                    key + '[2]'] = 'This feature is ignore according to the small Pearson correlation coefficient.'
            else:
                continue

        # return sorted(pearsondict.items(),key = lambda x:x[1],reverse = True),sorted(pearsondict_p.items(),key = lambda x:x[1],reverse = False)
        # print('%s : %f'%(self.X.columns[i],pearsonr(self.X.iloc[:,i],self.Y[self.target])[0]))
        # print('####################################################')

    def select_chi2(self):
        chi2dict_p = {}
        chi2dict_s = {}
        X_NOMINAL = self.X_NOMINAL.copy()
        X_NATURAL = self.X_NATURAL.copy()
        X_DATE = self.X_DATE.copy()
        for col in X_NATURAL:
            X_NATURAL[col] = X_NATURAL[col].replace({float('nan'): 'unknown'})
            vectorizer = HashingVectorizer(n_features=5, norm=None)
            # hash_vectorizer = abs(vectorizer.fit_transform(X_NATURAL[col]).toarray())
            hash_vectorizer = vectorizer.fit_transform(X_NATURAL[col]).toarray()
            hash_pd = pd.DataFrame(hash_vectorizer)
            for j in pd.DataFrame(hash_vectorizer):
                hash_pd[str(col) + '_hash' + str(j)] = hash_pd[j]
                hash_pd = hash_pd.drop(j, axis=1)
            X_NATURAL = pd.concat([X_NATURAL, hash_pd], axis=1)
            del X_NATURAL[col]

        for col in X_NOMINAL:
            actypes = {}
            n = 0
            for actype in X_NOMINAL[col].unique():
                actypes[actype] = n
                n += 1
            X_NOMINAL[col] = X_NOMINAL[col].replace(actypes)

        X_DATE = X_DATE.fillna(X_DATE.mean()[:])

        X_FINAL = pd.concat([X_NATURAL, X_NOMINAL, X_DATE], axis=1).copy()
        X_FINAL = X_FINAL.astype('float64')

        for i in X_FINAL:
            X_FINAL[i] = MinMaxScaler().fit_transform(X_FINAL[i].values.reshape(-1, 1))

        for i in X_FINAL:
            if len(X_FINAL[i].unique()) == 1:
                chi2dict_p[i] = 1
            else:
                chi2dict_p[i] = chi2(X_FINAL[i].values.reshape(-1, 1), self.Y)[1][0]

        for i in X_FINAL:
            if len(X_FINAL[i].unique()) == 1:
                chi2dict_s[i] = 0
            else:
                chi2dict_s[i] = chi2(X_FINAL[i].values.reshape(-1, 1), self.Y)[0][0]

        # nlist = []
        # for key in chi2dict_s:
        #    nlist.append(chi2dict_s[key])
        # narray = np.array(nlist)
        # sum1 = narray.sum()
        # mean = sum1/len(nlist)
        # std = np.std(narray)
        # interval = stats.t.interval(0.90,len(nlist)-1,mean,std)

        for key in X_NOMINAL:
            if (chi2dict_p[key] > 0.05):
                self.ignore_list.append(key)
                self.ignore[key + '[3]'] = 'This feature is ignore according to the chi-square test.'
            else:
                continue

        for key in X_DATE:
            if (chi2dict_p[key] > 0.05):
                self.ignore_list.append(key)
                self.ignore[key + '[3]'] = 'This feature is ignore according to the chi-square test.'
            else:
                continue

        for key in self.X_NATURAL:
            if (chi2dict_p[key + '_hash0'] > 0.05) and (chi2dict_p[key + '_hash1'] > 0.05) and (
                    chi2dict_p[key + '_hash2'] > 0.05) and (chi2dict_p[key + '_hash3'] > 0.05) and (
                    chi2dict_p[key + '_hash4'] > 0.05):
                self.ignore_list.append(key)
                self.ignore[key + '[3]'] = 'This feature is ignore according to the chi-square test.'
            else:
                continue

        # return sorted(chi2dict_s.items(),key = lambda x:x[1],reverse = True),sorted(chi2dict_p.items(),key = lambda x:x[1],reverse = False)

    def select_L1(self):
        lr = LogisticRegression(C=0.1, penalty="l1")
        X_NOMINAL = self.X_NOMINAL.copy()
        X_NATURAL = self.X_NATURAL.copy()
        X_DATE = self.X_DATE.copy()
        X_DATE = X_DATE.fillna(X_DATE.mean()[:])
        X_REAL = self.X_REAL.copy()
        X_REAL = X_REAL.fillna(X_REAL.mean()[:])
        dict_l1 = {}

        for col in X_NATURAL:
            X_NATURAL[col] = X_NATURAL[col].replace({float('nan'): 'unknown'})
            vectorizer = HashingVectorizer(n_features=5, norm=None)
            hash_vectorizer = abs(vectorizer.fit_transform(X_NATURAL[col]).toarray())
            hash_pd = pd.DataFrame(hash_vectorizer)
            for j in pd.DataFrame(hash_vectorizer):
                hash_pd[str(col) + '_hash' + str(j)] = hash_pd[j]
                hash_pd = hash_pd.drop(j, axis=1)
            X_NATURAL = pd.concat([X_NATURAL, hash_pd], axis=1)
            del X_NATURAL[col]

        for col in X_NOMINAL:
            actypes = {}
            n = 0
            x = 0
            for actype in X_NOMINAL[col].unique():
                actypes[actype] = n * pow(-1, x)
                x += 1
                if x % 2 == 1:
                    n = n + 1
                else:
                    n = n
            X_NOMINAL[col] = X_NOMINAL[col].replace(actypes)

        X_concat = pd.concat([X_REAL, X_NOMINAL, X_DATE, X_NATURAL], axis=1)

        X_norm = X_concat.copy()
        X_norm = X_norm.astype('float64')

        for i in X_norm:
            X_norm[i] = StandardScaler().fit_transform(X_norm[i].values.reshape(-1, 1))

        lr_l1 = lr.fit(X_norm, self.Y)

        for i in range(0, X_norm.shape[1]):
            dict_l1[X_norm.columns[i]] = abs(lr_l1.coef_[0][i])

        nlist = []
        for key in dict_l1:
            nlist.append(dict_l1[key])
        narray = np.array(nlist)
        sum1 = narray.sum()
        mean = sum1 / (len(nlist))
        std = np.std(narray)
        interval = stats.t.interval(0.90, len(nlist) - 1, mean, std)

        for key in X_NOMINAL:
            if (dict_l1[key] < 0.00001):
                self.ignore_list.append(key)
                self.ignore[key + '[4]'] = 'This feature is ignore according to the L1.'
            else:
                continue

        for key in X_REAL:
            if (dict_l1[key] < 0.00001):
                self.ignore_list.append(key)
                self.ignore[key + '[4]'] = 'This feature is ignore according to the L1.'
            else:
                continue

        for key in X_DATE:
            if (dict_l1[key] < 0.00001):
                self.ignore_list.append(key)
                self.ignore[key + '[4]'] = 'This feature is ignore according to the L1.'
            else:
                continue

        for key in self.X_NATURAL:
            if (dict_l1[key + '_hash0'] < 0.00001) and (dict_l1[key + '_hash1'] < 0.00001) and (
                    dict_l1[key + '_hash2'] < 0.00001) and (dict_l1[key + '_hash3'] < 0.00001) and (
                    dict_l1[key + '_hash4'] < 0.00001):
                self.ignore_list.append(key)
                self.ignore[key + '[4]'] = 'This feature is ignore according to the L1.'
            else:
                continue

        # return sorted(dict_l1.items(),key = lambda x:x[1],reverse = True)

    def select_tree(self):
        tr = ExtraTreesClassifier()
        X_NOMINAL = self.X_NOMINAL.copy()
        X_NATURAL = self.X_NATURAL.copy()
        X_DATE = self.X_DATE.copy()
        X_DATE = X_DATE.fillna(X_DATE.mean()[:])
        X_REAL = self.X_REAL.copy()
        X_REAL = X_REAL.fillna(X_REAL.mean()[:])
        dict_tr = {}

        for col in X_NATURAL:
            X_NATURAL[col] = X_NATURAL[col].replace({float('nan'): 'unknown'})
            vectorizer = HashingVectorizer(n_features=5, norm=None)
            hash_vectorizer = abs(vectorizer.fit_transform(X_NATURAL[col]).toarray())
            hash_pd = pd.DataFrame(hash_vectorizer)
            for j in pd.DataFrame(hash_vectorizer):
                hash_pd[str(col) + '_hash' + str(j)] = hash_pd[j]
                hash_pd = hash_pd.drop(j, axis=1)
            X_NATURAL = pd.concat([X_NATURAL, hash_pd], axis=1)
            del X_NATURAL[col]

        for col in X_NOMINAL:
            actypes = {}
            n = 0
            x = 0
            for actype in X_NOMINAL[col].unique():
                actypes[actype] = n * pow(-1, x)
                x += 1
                if x % 2 == 1:
                    n = n + 1
                else:
                    n = n
            X_NOMINAL[col] = X_NOMINAL[col].replace(actypes)

        X_concat = pd.concat([X_REAL, X_NOMINAL, X_DATE, X_NATURAL], axis=1)

        X_norm = X_concat.copy()
        X_norm = X_norm.astype('float64')

        for i in X_norm:
            X_norm[i] = StandardScaler().fit_transform(X_norm[i].values.reshape(-1, 1))

        tr = tr.fit(X_norm, self.Y)

        for i in range(0, X_norm.shape[1]):
            dict_tr[X_norm.columns[i]] = tr.feature_importances_[i]

        nlist = []
        for key in dict_tr:
            nlist.append(dict_tr[key])
        narray = np.array(nlist)
        sum1 = narray.sum()
        mean = sum1 / (len(nlist))
        std = np.std(narray)
        interval = stats.lognorm.interval(alpha=0.90, s=1, loc=mean, scale=std)
        interval_floor = interval[0] * 0.9

        for key in X_NOMINAL:
            if (dict_tr[key] < interval_floor):
                self.ignore_list.append(key)
                self.ignore[key + '[5]'] = 'This feature is ignore according to the Tree.'
            else:
                continue

        for key in X_REAL:
            if (dict_tr[key] < interval_floor):
                self.ignore_list.append(key)
                self.ignore[key + '[5]'] = 'This feature is ignore according to the Tree.'
            else:
                continue

        for key in X_DATE:
            if (dict_tr[key] < interval_floor):
                self.ignore_list.append(key)
                self.ignore[key + '[5]'] = 'This feature is ignore according to the Tree.'
            else:
                continue

        for key in self.X_NATURAL:
            if (dict_tr[key + '_hash0'] < interval_floor) and (dict_tr[key + '_hash1'] < interval_floor) and (
                    dict_tr[key + '_hash2'] < interval_floor) and (dict_tr[key + '_hash3'] < interval_floor) and (
                    dict_tr[key + '_hash4'] < interval_floor):
                self.ignore_list.append(key)
                self.ignore[key + '[5]'] = 'This feature is ignore according to the Tree.'
            else:
                continue

        # return sorted(dict_tr.items(),key = lambda x:x[1],reverse = True)

    def feature_selection(self):
        if len(self.yvar) == 2:
            self.select_missing()
            self.select_lowvariance()
            self.select_pearson()
            self.select_chi2()
            self.select_L1()
            self.select_tree()
            self.ignore_list_set = set(self.ignore_list)
        elif len(self.yvar) > 50:
            self.select_missing()
            self.select_lowvariance()
            self.select_pearson()
            self.select_chi2()
            self.select_tree()
            self.ignore_list_set = set(self.ignore_list)
        else:
            self.select_missing()
            self.select_lowvariance()
            self.select_pearson()
            self.select_chi2()

        return self.ignore, self.ignore_list_set

