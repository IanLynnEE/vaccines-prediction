import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer




def test():
    train_df = read_and_encode('data/training_set_features.csv')
    test_df = read_and_encode('data/training_set_features.csv')
    x, xt = impute_features(train_df, test_df)
    x.info()
    xt.info()
    # print(x['health_insurance'])
    return


def read_and_encode(filename: str) -> pd.DataFrame:
    """Read a file and encode the string objects to numbers. NaN will be retained.

    Args:
        filename (str): Path to the file.

    Returns:
        pd.DataFrame: DataFrame with values in int64 (or float64).
    """
    df = pd.read_csv(filename)

    age_map = {'18 - 34 Years': 0, '35 - 44 Years': 1, '45 - 54 Years': 2, '55 - 64 Years': 3, '65+ Years': 4}
    edu_map = {'< 12 Years': 0, '12 Years': 1, 'Some College': 2, 'College Graduate': 3}
    sex_map = {'Male': 0, 'Female': 1}
    income_map = {'Below Poverty': 0, '<= $75,000, Above Poverty': 1, '> $75,000': 2, 'NA': 0.8}
    income_map = {'Below Poverty': 0, '<= $75,000, Above Poverty': 1, '> $75,000': 2}
    marital_map = {'Not Married': 0, 'Married': 1}
    rent_map = {'Rent': 0, 'Own': 1}
    employ_map = {'Unemployed': 0, 'Not in Labor Force': 1, 'Employed': 2}
    census_map = {'Non-MSA': 0, 'MSA, Not Principle  City': 1, 'MSA, Principle City': 2}

    # TODO The following 4 features should be encoded more carefully.
    #race_map = {'Black': 2, 'Hispanic': 1, 'Other or Multiple': 0, 'White': 3}
    race_map = df.race.value_counts().to_dict()
    hhs_map = df.hhs_geo_region.value_counts().to_dict()
    for a, b in df.iterrows():
        if b['employment_status'] == 'Unemployed' and pd.isna(b['employment_industry']) and pd.isna(b['employment_occupation']):
            df.at[a, "employment_industry"] = 'NO_job'
            df.at[a, "employment_occupation"] = 'NO_job'
        if b['employment_status'] == 'Not in Labor Force' and pd.isna(b['employment_industry']) and pd.isna(b['employment_occupation']):
            df.at[a, "employment_industry"] = 'Not_in_Labor_Force'
            df.at[a, "employment_occupation"] = 'Not_in_Labor_Force'
    
    df['employment_industry'].fillna(value='Not Found', inplace=True)
    df['employment_occupation'].fillna(value='Not Found', inplace=True)
    print(df['employment_industry'].value_counts())
    df['health_insurance'].fillna(value='0.5', inplace=True)
    df['income_poverty'].fillna(value='NA', inplace=True)
    industry_map = df.employment_industry.value_counts().to_dict()
    occupation_map = df.employment_occupation.value_counts().to_dict()

    df.age_group = df.age_group.map(age_map)
    df.education = df.education.map(edu_map)
    df.race = df.race.map(race_map)
    df.sex = df.sex.map(sex_map)
    df.income_poverty = df.income_poverty.map(income_map)
    df.marital_status = df.marital_status.map(marital_map)
    df.rent_or_own = df.rent_or_own.map(rent_map)
    df.employment_status = df.employment_status.map(employ_map)
    df.hhs_geo_region = df.hhs_geo_region.map(hhs_map)
    df.census_msa = df.census_msa.map(census_map)
    df.employment_industry = df.employment_industry.map(industry_map)
    df.employment_occupation = df.employment_occupation.map(occupation_map)
    # df2 = pd.get_dummies(df['health_insurance'], prefix='health_insurance')
    # df = df.drop('health_insurance', 1)
    # df = pd.concat([df, df2],axis=1)
    # df2 = pd.get_dummies(df['hhs_geo_region'], prefix='hhs_geo_region')
    # df =df.drop('hhs_geo_region',1)
    # df = pd.concat([df, df2],axis=1)
    return df


def impute_features(x: pd.DataFrame, xt: pd.DataFrame,
                    *, strategy: str = 'mean', known_test: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Impute missing values based on other  data.

    Args:
        x (pd.DataFrame): Training set (features only).
        xt (pd.DataFrame): Test set (features only).
        strategy (str, optional): The imputation strategy. Defaults to 'mean'.
        known_test (bool, optional): Whether to use test set in calculation. Defaults to False.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Training set and test set.
    """
    # TODO For most of the features, it makes sense to use "mean", "median" or "most_frequent".
    # However, it might be better to directly assign values for some of the features.
    # Moreover, we should try to compute different means for different races/ages/education levels as well.
    #imp = SimpleImputer(strategy=strategy, copy=True)
    if (strategy == "KNN"):
        print("Use KNN imputing")
        imp = KNNImputer(n_neighbors=5)
        imp.fit(x)
        if known_test:
            data = pd.concat([x, xt], axis=0, ignore_index=True)
            imp.fit(data)
    elif (strategy == "Iterative"):
        print("Use Iterative imputing")
        imp = IterativeImputer(random_state=9, max_iter=10, initial_strategy='most_frequent')
        if known_test:
            data = pd.concat([x, xt], axis=0, ignore_index=True)
            imp.fit(data)
        else:
            imp.fit(x)
    else:
        imp = SimpleImputer(strategy=strategy, copy=True)
        imp.fit(x)
        if known_test:
            data = pd.concat([x, xt], axis=0, ignore_index=True)
            imp.fit(data)
    return pd.DataFrame(imp.transform(x), columns=x.columns), pd.DataFrame(imp.transform(xt), columns=xt.columns)


if __name__ == '__main__':
    test()
