import pandas as pd
from sklearn.experimental import enable_iterative_imputer   # noqa: F401
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer


def test():
    train_df = read_and_encode('data/training_set_features.csv')
    test_df = read_and_encode('data/training_set_features.csv')
    x, xt = impute_features(train_df, test_df)
    x.info()
    xt.info()
    return


def read_and_encode(filename: str) -> pd.DataFrame:
    """Read a file and encode the string objects to numbers. NaN will be retained for most of the features.
    Args:
        filename (str): Path to the file.

    Returns:
        pd.DataFrame: DataFrame with values in int64 (or float64).
    """
    df = pd.read_csv(filename)

    age_map = {'18 - 34 Years': 0, '35 - 44 Years': 1, '45 - 54 Years': 2, '55 - 64 Years': 3, '65+ Years': 4}
    edu_map = {'< 12 Years': 0, '12 Years': 1, 'Some College': 2, 'College Graduate': 3}
    sex_map = {'Male': 0, 'Female': 1}
    income_map = {'Below Poverty': 0, '<= $75,000, Above Poverty': 1, '> $75,000': 2}
    marital_map = {'Not Married': 0, 'Married': 1}
    rent_map = {'Rent': 0, 'Own': 1}
    employ_map = {'Unemployed': 0, 'Not in Labor Force': 1, 'Employed': 2}
    census_map = {'Non-MSA': 0, 'MSA, Not Principle  City': 1, 'MSA, Principle City': 2}

    # TODO The following 4 features should be encoded more carefully.
    # race_map = {'Black': 0, 'Hispanic': 1, 'Other or Multiple': 2, 'White': 3}
    race_map = df.race.value_counts().to_dict()
    hhs_map = df.hhs_geo_region.value_counts().to_dict()

    for i, row in df.iterrows():
        if pd.isna(row['employment_industry']) and pd.isna(row['employment_occupation']):
            if row['employment_status'] == 'Unemployed':
                df.at[i, 'employment_industry'] = 'NO_job'
                df.at[i, 'employment_occupation'] = 'NO_job'
            if row['employment_status'] == 'Not in Labor Force':
                df.at[i, 'employment_industry'] = 'Not_in_Labor_Force'
                df.at[i, 'employment_occupation'] = 'Not_in_Labor_Force'
    df['employment_industry'].fillna(value='Not Found', inplace=True)
    df['employment_occupation'].fillna(value='Not Found', inplace=True)

    industry_map = df.employment_industry.value_counts().to_dict()
    occupation_map = df.employment_occupation.value_counts().to_dict()
    # END of the comment above.

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
    # TESTING
    # x['health_insurance'].fillna(value=0.5, inplace=True)
    # xt['health_insurance'].fillna(value=0.5, inplace=True)

    # TESTING
    # x['income_poverty'].fillna(value=0.8, inplace=True)
    # xt['income_poverty'].fillna(value=0.8, inplace=True)

    if (strategy == 'KNN'):
        imp = KNNImputer(n_neighbors=5)
    elif (strategy == 'Iterative'):
        imp = IterativeImputer(random_state=9, max_iter=10, initial_strategy='most_frequent')
    else:
        imp = SimpleImputer(strategy=strategy, copy=True)

    if known_test:
        data = pd.concat([x, xt], axis=0, ignore_index=True)
        imp.fit(data)
    else:
        imp.fit(x)
    return pd.DataFrame(imp.transform(x), columns=x.columns), pd.DataFrame(imp.transform(xt), columns=xt.columns)


if __name__ == '__main__':
    test()
