'''

Description:
I've developed a recommendation system using a model-based approach, leveraging XGBoost as the primary model. To refine feature selection, I employed Forward Feature Selection, evaluating each feature's impact on the model's Root Mean Square Error (RMSE). For hyperparameter tuning, I utilized methods such as GridSearchCV and Optuna, with Grid Search proving most effective. Given the computational intensity, Grid Search was parallelized across multiple machines to reduce processing time. Grid search seemed to work better and faster for me when I used less paramters like (max_depth, nestimators and learning rate), after which I add more parameters, which helped me reduce my RMSE.

Total Nuber of features = >45

RMSE: 0.9774683724603355

Error Distribution
0 <= Error < 1: 102376  (72.07%)
1 <= Error < 2: 32720  (23.04%)
2 <= Error < 3: 6130  (4.32%)
3 <= Error < 4: 817  (0.58%)
Error >= 4: 1 (0.00%)



Execution Time:
772.3729314804077


'''

from collections import defaultdict
from pyspark import SparkConf, SparkContext
import json
import sys
import time
import numpy as np
import pandas as pd
from collections import defaultdict
import time
import os
from sklearn.model_selection import GridSearchCV
import math
from itertools import combinations
import random
import xgboost as xg
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV
import statistics
from collections import Counter





#python_path = 'C:\\Python368\\python.exe'

#os.environ['PYSPARK_PYTHON'] = python_path  
#os.environ['PYSPARK_DRIVER_PYTHON'] = python_path

sc= SparkContext('local[*]','Competition')
sc.setLogLevel("WARN")



dir_folder = sys.argv[1]
dir_val = sys.argv[2]
dir_final_op = sys.argv[3]




start = time.time()
param_grid = {
    'learning_rate': [0.01, 0.1, 0.5],             # Learning rate values to try
    'max_depth': [5, 6, 7],                        # Maximum depth of trees
    'min_child_weight': [1, 3, 5],                  # Minimum sum of instance weight needed in a child
    'gamma': [0, 0.1, 0.2],                         # Minimum loss reduction required to make a further partition on a leaf node
    'subsample': [0.8, 1.0],                        # Subsample ratio of the training instance
    'colsample_bytree': [0.8, 1.0],                 # Subsample ratio of columns when constructing each tree
    'reg_alpha': [0, 0.1, 0.5],                     # L1 regularization term on weights
    'reg_lambda': [0, 1, 2],                        # L2 regularization term on weights
    'n_estimators': [600, 700, 800, 900],                # Number of boosting rounds
    'random_state': [42]                            # Seed for random number generation
}

def grid_search_cv(df_train, target_df, param_grid, num_folds=5):
    # Instantiate an XGBRegressor
    xgb = xg.XGBRegressor()

    # Create a GridSearchCV object
    grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=num_folds, scoring='neg_mean_squared_error', n_jobs=-1)

    # Fit the GridSearchCV object
    grid_search.fit(df_train, target_df)

    # Get the best parameters
    best_params = grid_search.best_params_
    
    return best_params

zero = 0.0
def returnBool(value):
    return 1.0 if value.lower() == 'true' else zero

def askPriceRange(x):
    if x is not None:
        return float(x.get('RestaurantsPriceRange2', 2.0))
    else:
        return 2.0

def askAccpetedCards(x):
    if isinstance(x, dict):
        return returnBool(x.get('BusinessAcceptsCreditCards', 'false'))
    else:
        return zero

def askTakeout(x):
    if isinstance(x, dict):
        return returnBool(x.get('RestaurantsTakeOut', 'false'))
    else:
        return zero

def askReservations(x):
    if isinstance(x, dict):
        return returnBool(x.get('RestaurantsReservations', 'false'))
    else:
        return zero

def askDelivery(x):
    if isinstance(x, dict):
        return returnBool(x.get('RestaurantsDelivery', 'false'))
    else:
        return zero

def askBreakFast(x):
    if isinstance(x, dict) and 'GoodForMeal' in x and "'breakfast': True" in x['GoodForMeal']:
        return 1.0
    else:
        return zero

def askLunch(x):
    if isinstance(x, dict) and 'GoodForMeal' in x and "'lunch': True" in x['GoodForMeal']:
        return 1.0
    else:
        return zero

def askDinner(x):
    if isinstance(x, dict) and 'GoodForMeal' in x and "'dinner': True" in x['GoodForMeal']:
        return 1.0
    else:
        return zero

def askBrunch(x):
    if isinstance(x, dict) and 'GoodForMeal' in x and "'brunch': True" in x['GoodForMeal']:
        return 1.0
    else:
        return zero
def askWheelchairAccessible(x):
    if isinstance(x, dict):
        return returnBool(x.get('WheelchairAccessible', 'false'))
    else:
        return zero

def good_for_kidsF(x):
    if isinstance(x, dict):
        return returnBool(x.get('GoodForKids', 'false'))
    else:
        return zero
    
def good_for_groupsF(x):
    if isinstance(x, dict):
        return returnBool(x.get('RestaurantsGoodForGroups', 'false'))
    else:
        return zero

def wifiF(x):
    if isinstance(x, dict):
        if 'WiFi' in x and (x['WiFi'] == 'free' or x['WiFi'] == 'paid'):
            return 1.0
        else:
            return zero
    else:
        return zero
    
def askOutdoorSeating(x):
    if isinstance(x, dict):
        return returnBool(x.get('OutdoorSeating', 'false'))
    else:
        return zero

def askHasTV(x):
    if isinstance(x, dict):
        return returnBool(x.get('HasTV', 'false'))
    else:
        return zero

def alcoholF(x):
    if isinstance(x, dict):
        return x.get('Alcohol', 'false')
    else:
        return zero

def restaurantsdeliveryF(x):
    if isinstance(x,dict):
        return returnBool(x.get('RestaurantsDelivery', 'false'))
    else:
        return zero
    
def restaurantstakeoutF(x):
    if isinstance(x,dict):
        return returnBool(x.get('RestaurantsTakeOut', 'false'))
    else:
        return zero


def by_appointment_only(x):
    if isinstance(x,dict):
        return returnBool(x.get('ByAppointmentOnly', 'false'))
    else: 
        return zero

    
def create_tuple_list_from_json(json_file, fields):
    lines_r = sc.textFile(json_file)
    tuple_list = lines_r.map(json.loads).map(lambda line: tuple(line[field] for field in fields)).collect()
    return tuple_list


params = {
    'objective': 'regression',
    'metric': 'rmse',
    'num_leaves': 31,
    'learning_rate': 0.07,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0
}
from functools import reduce

def merge_dataframes(df1, df2, key, how='left'):
    return pd.merge(df1, df2, on=key, how=how)


def initialize_client_rating_data(df_train):
    client_rating_data = defaultdict(lambda: {'std_sum': 0, 'min': float('inf'), 'max': float('-inf'), 'no_of_reviews': 0})
    return client_rating_data

def update_client_rating_data(client_rating_data, client_id, stars, average_stars):
    client_rating_data[client_id]['std_sum'] += pow(stars - average_stars, 2)
    client_rating_data[client_id]['min'] = min(stars, client_rating_data[client_id]['min'])
    client_rating_data[client_id]['max'] = max(stars, client_rating_data[client_id]['max'])
    client_rating_data[client_id]['no_of_reviews'] += 1

def process_df_train(df_train):
    client_rating_data = initialize_client_rating_data(df_train)
    for _, row in df_train.iterrows():
        client_id = row['user_id']
        stars = row['stars']
        average_stars = row['average_stars']
        update_client_rating_data(client_rating_data, client_id, stars, average_stars)
    return client_rating_data

def create_dataframe_from_json(file_path, fields):
    data = create_tuple_list_from_json(file_path, fields)
    df = pd.DataFrame(data, columns=fields)
    return df

def write_predictions_to_file(dir_final_op, predicted_df):
    with open(dir_final_op, "w") as f:
        f.write("user_id,business_id,prediction\n")
        for index, row in predicted_df.iterrows():
            f.write(f"{row['user_id']},{row['business_id']},{row['prediction']}\n")


def get_latitude_or_zero(row):
    if 'latitude' in row:  # Check if the 'latitude' column exists in the row
        #print(row['latitude'])
        return row['latitude'] if not pd.isnull(row['latitude']) else zero
    else:
        return zero

def get_longitude_or_zero(row):
    if 'longitude' in row:  # Check if the 'latitude' column exists in the row
    
        return row['longitude'] if not pd.isnull(row['longitude']) else zero
    else:
        return zero




varh = 'hours'
vara = 'business_id'
vart = 'time'
checkinRDD = sc.textFile(dir_folder + "/checkin.json")
parsed_checkins = checkinRDD.map(lambda a: json.loads(a))
busin= parsed_checkins.map(lambda a: (a[vara], list(a[vart].values())))
checkins_businessD = busin.collectAsMap()

def ck_bs_days(business_id):
    if business_id in checkins_businessD:
        return len(checkins_businessD[business_id])
    else:
        return 0

# Read JSON file containing tips data
tips_rdd = sc.textFile(dir_folder +"/tip.json")
tips_data = tips_rdd.map(lambda x: json.loads(x))
business_tips_count = tips_data.map(lambda x: (x['business_id'], 1)).reduceByKey(lambda x, y: x + y)

tips_businessess = business_tips_count.collectAsMap()

def tips_business(business_id):
    if business_id in tips_businessess:
        x = tips_businessess[business_id]
        return x
    else:
        return 0

photo_rdd = sc.textFile(dir_folder +"/photo.json")
photo_data = photo_rdd.map(lambda x: json.loads(x))
business_photo_count = photo_data.map(lambda x: (x['business_id'], 1)).reduceByKey(lambda x, y: x + y)

photo_businessess = business_photo_count.collectAsMap()

def photo_business(business_id):
    if business_id in photo_businessess:
        x = photo_businessess[business_id]
        return x
    else:
        return 0
    
def rest_price_rangeF(x):
    if isinstance(x, dict):
        if 'RestaurantsPriceRange2' in x:
            f = x['RestaurantsPriceRange2']
            z = float(f)
            return z
    else:
        return zero

def adjust_prediction(val, par):
    decimal_part = val - math.floor(val)
    
    if decimal_part < 0.5:
        new_val = val - par
        threshold = math.floor(val)
        if new_val < threshold:
            new_val = threshold
    else:
        new_val = val + par
        threshold = math.ceil(val)
        if new_val > threshold:
            new_val = threshold
    return new_val


businessRDD = sc.textFile(dir_folder + "/business.json")
# based on opening time average
# Assuming businessRDD contains JSON data

# Step 1: Parse JSON data
parsed_data = businessRDD.map(lambda row: json.loads(row))
opening_times = parsed_data.map(lambda rowD: (
    rowD[vara], 
    [float(".".join(hours.split("-")[0].split(":"))) for hours in rowD[varh].values()]
) if (varh in rowD) and (rowD[varh]) else (rowD[vara], None))

average_opening_time = opening_times.map(lambda rowL: (
    rowL[0], 
    sum(rowL[1])/len(rowL[1]) if rowL[1] else None
))

opening_timeD = average_opening_time.collectAsMap()

sum_o = sum([num for num in opening_timeD.values() if num])

opening_timeD_avg = sum_o/len(opening_timeD)


def check_client_id_in_dict(user_id, user_dict):
    return user_id in user_dict and (user_dict[user_id] or user_dict[user_id] == 0)

def opening_timeF(business_id):
    if check_client_id_in_dict(business_id, opening_timeD):
        val = opening_timeD[business_id]
    else:
        val = opening_timeD_avg
    return val



def mulmimo(jh):
    try:
        return statistics.mode(jh)
    except statistics.StatisticsError:
        modes = find_modes(jh)
        return choose_mode(modes)

def find_modes(some_list):
    modes = []
    num_counts = Counter(some_list)
    max_mode_count = num_counts.most_common(1)[0][1]
    index = 0
    while index < len(some_list):
        num = some_list[index]
        if  (some_list.count(num) == max_mode_count) and (num not in modes):
            modes.append(num)
        index += 1
    return modes

def choose_mode(modes):
    if min(modes) > 6:
         return max(modes)
    else:
        return min(modes)
       


pds = businessRDD.map(lambda row: json.loads(row))
pda = pds.map(lambda rowD: ( rowD["business_id"], mulmimo([ float(".".join(hours.split("-")[1].split(":"))) for hours in rowD["hours"].values() ]) ) if ("hours" in rowD) and (rowD["hours"]) else (rowD["business_id"], None))
pdp = pda.collectAsMap()
closing_timeD_mode = mulmimo([num for num in pdp.values() if num])

def closing_timeF(business_id):
    if check_client_id_in_dict(business_id, pdp):
        x= pdp[business_id]
        return x
    else:
        return closing_timeD_mode





# Load JSON data from businessRDD
parsed_data = businessRDD.map(lambda row: json.loads(row))
filtered_data = parsed_data.filter(lambda rowD: ("hours" in rowD) and (rowD["hours"]))
mapped_data = filtered_data.map(lambda rowD: (rowD["business_id"], len(rowD["hours"])))
days_openD = mapped_data.collectAsMap()

sumx = sum([count for count in days_openD.values()])

days_openD_avg = sumx/len(days_openD)

def duv_hulF(business_id):
    if check_client_id_in_dict(business_id, days_openD):
        x = days_openD[business_id]
        return x
    else:
        return days_openD_avg
        # return None
def extract_business_category(line):
    line_parts = line.split('"categories": "')
    if len(line_parts) > 1:
        business_id = line_parts[1].split('", "')[0]
        return (business_id, 1)
    else:
        return (None, None)  # Indicates no categories
    


business_rdd = sc.textFile(dir_folder + "/business.json") \
    .map(lambda row: json.loads(row))

# Step 1: Filter out rows where "categories" is None or an empty string
filtered_business_rdd = business_rdd.filter(lambda rowD: rowD.get("categories"))

# Step 2: Extract all categories and count their occurrences
category_counts = filtered_business_rdd.flatMap(lambda rowD: rowD["categories"].split(",")) \
    .map(lambda category: (category.strip(), 1)) \
    .reduceByKey(lambda x, y: x + y)


top_N = 10
top_categories = category_counts.takeOrdered(top_N, key=lambda x: -x[1])
top_category_names = [category[0] for category in top_categories]

business_category_mapping = business_rdd.map(lambda rowD: (rowD["business_id"], rowD.get("categories", "").split(",") if rowD.get("categories") else [0 for _ in range(top_N)]))

def create_dummy_variables(categories_list, top_categories):
    dummy_vector = [1 if category.strip() in categories_list else 0 for category in top_categories]
    return dummy_vector

business_dummy_variables = business_category_mapping.map(lambda x: (x[0], create_dummy_variables(x[1], top_category_names)))
bus_dum = business_dummy_variables.collectAsMap()

def cat_get(business_id):
    if business_id in bus_dum:
        x =  bus_dum[business_id]
        return x
    else:
        return [0 for _ in range(top_N)]

def get_prediction():

    client_fields1 = ['user_id', 'average_stars'] 
    client_fields2 = ['review_count', 'useful','funny','cool', 'fans' ]
    client_fields3 = ['compliment_note', 'compliment_hot']
    client_fields = client_fields1 + client_fields2 + client_fields3
    enterprise_fields1 = ['business_id', 'stars', 'review_count']
    enterprise_fields2 = ['latitude','longitude']
    enterprise_fields3 = [ 'is_open', 'categories','attributes']
    enterprise_fields = enterprise_fields1 + enterprise_fields2 + enterprise_fields3
     
    client_df = create_dataframe_from_json(dir_folder + "/user.json", client_fields)
    enterprise_df = create_dataframe_from_json(dir_folder + "/business.json", enterprise_fields)

    attribute_functions = {
            'PriceRange': askPriceRange,
            'CardAccepted': askAccpetedCards,
            'Takeout': askTakeout,
            'Reservations': askReservations,
            'Delivery': askDelivery,
            'Breakfast': askBreakFast,
            'Lunch': askLunch,
            'Dinner': askDinner,
            'Brunch': askBrunch,
            'WheelchairAccessible': askWheelchairAccessible,
            'OutdoorSeating': askOutdoorSeating,
            'HasTV': askHasTV,
            'GoodForKidsF': good_for_kidsF,
            'GoodForGroupsF': good_for_groupsF,
            'WifiF': wifiF,
            'RestaurantsPriceRange' : rest_price_rangeF,
            'RestaurantsDelivery' : restaurantsdeliveryF,
            'RestaurantsTakeOut':restaurantstakeoutF
        }

    for attribute, function in attribute_functions.items():
        enterprise_df[attribute] = enterprise_df['attributes'].apply(function)



    enterprise_df['LatF'] = enterprise_df.apply(get_latitude_or_zero, axis = 1)
    enterprise_df['LongF'] = enterprise_df.apply(get_longitude_or_zero, axis = 1)
    enterprise_df['checkin_days'] = enterprise_df['business_id'].apply(ck_bs_days)
    enterprise_df['tips'] = enterprise_df['business_id'].apply(tips_business)
    enterprise_df['photos'] = enterprise_df['business_id'].apply(photo_business)
    enterprise_df['opening_timeF'] = enterprise_df['business_id'].apply(opening_timeF)
    enterprise_df['closing_timeF'] = enterprise_df['business_id'].apply(closing_timeF)
    enterprise_df['days_openF'] = enterprise_df['business_id'].apply(duv_hulF)

    enterprise_df['dum_categories'] = enterprise_df['business_id'].apply(cat_get)
    num_categories = len(enterprise_df['dum_categories'].iloc[0])
    name_categories = [] 
    for i in range(num_categories):
        enterprise_df[f'business_cat_{i+1}'] = enterprise_df['dum_categories'].apply(lambda x: x[i])
        name_categories.append(f'business_cat_{i+1}')
    enterprise_df.drop(columns=['dum_categories'], inplace=True)
    
    enterprise_df = enterprise_df[['business_id', 'stars', 'review_count', 'is_open', *attribute_functions.keys(), 'LatF', 'LongF','checkin_days', 'opening_timeF', 'closing_timeF', 'days_openF','tips','photos']+name_categories]

    client_df.columns = client_df.columns.str.replace('review_count', 'client_review_count')
    enterprise_df.columns = enterprise_df.columns.str.replace('stars', 'buniness_stars').str.replace('review_count', 'buniness_review_count')


    df_train = pd.read_csv(dir_folder+'/yelp_train.csv')
    df_train = merge_dataframes(df_train,client_df,'user_id','inner')
    df_train = merge_dataframes(df_train,enterprise_df,'business_id','inner')

    client_rating_data = defaultdict(lambda: {'std_sum': 0, 'min': float('inf'), 'max': float('-inf'), 'no_of_reviews': 0})

    client_rating_data = process_df_train(df_train)

    client_rating_data = {
    key: {'std': val['std_sum'] / val['no_of_reviews'], **val}
    for key, val in client_rating_data.items()}

    client_rating_df = pd.DataFrame.from_dict(client_rating_data,orient='index')
    client_rating_df = client_rating_df.rename_axis('user_id')


    df_train = merge_dataframes(df_train,client_rating_df,'user_id','inner')
    df_train = df_train.drop(['std_sum', 'no_of_reviews'], axis=1)

    features_imp = ['average_stars', 'client_review_count','useful','funny','cool', 'fans','buniness_stars', 'buniness_review_count','is_open','compliment_note','compliment_hot', 'PriceRange','CardAccepted', 'Takeout', 'Reservations', 'Delivery', 'Breakfast', 'Lunch', 'Dinner','OutdoorSeating','HasTV','GoodForKidsF','GoodForGroupsF','WifiF','LatF','LongF', 'RestaurantsPriceRange','checkin_days','opening_timeF', 'closing_timeF', 'days_openF','RestaurantsDelivery','RestaurantsTakeOut','tips','photos'] + name_categories
    df_train_features = df_train[features_imp]
    df_train_features.to_csv('train_features.csv', index=False)
    df_train_target = df_train[['stars']]
    df_train_target.to_csv('target_features.csv', index=False)

 
    #train_data = lgb.Dataset(df_train_features, label=df_train_target)
 
    #model = lgb.train(params, train_data,  num_boost_round=1000)
    print('training start')
    
    # BEST MODEL 9771 
    # model = xg.XGBRegressor(n_estimators=800,learning_rate=0.1,max_depth=5,reg_lambda = 2, objective= 'reg:squarederror',colsample_bytree= 0.9, min_child_weight = 2,gamma=0.05, random_state=1)

    
    # BEST MODEL 9771 Top N Category
    model = xg.XGBRegressor(n_estimators=800,learning_rate=0.1,max_depth=5,reg_lambda = 2,colsample_bytree= 0.9, min_child_weight = 2,gamma=0.05, random_state=1)
    #model = xg.XGBRegressor(**parameter)
    model.fit(df_train_features,df_train_target)
    importances = model.feature_importances_
# Get feature names
    feature_names = df_train_features.columns

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    '''
    print("Important features:")
    for i in range(len(importances)):
        print(f"{feature_names[indices[i]]}: {importances[indices[i]]}")
    '''
    df_test = pd.read_csv(dir_val)
    df_test1 = df_test.copy()
    lx = ['user_id','business_id']
    merged_df = merge_dataframes(df_test, client_df, lx[0])
    merged_df = merge_dataframes(merged_df, enterprise_df, lx[1])
    df_test = merge_dataframes(merged_df,client_rating_df,lx[0])
    drop_cols = ['std_sum', 'no_of_reviews']
    df_test = df_test.drop(drop_cols, axis=1)

    test_clients_id = df_test[lx[0]]
    test_enterprise_id = df_test[lx[1]]

    df_test = df_test.drop(lx,axis=1)
    df_test = df_test[features_imp]
    rating_prediction = model.predict(df_test)

    data = {
    'user_id': test_clients_id,
    'business_id': test_enterprise_id,
    'prediction': rating_prediction
    }

    predicted_df = pd.DataFrame(data)

    predicted_df.to_csv('predicted.csv', index=False)
    #par_value = 0.05
    #predicted_df['prediction'] = predicted_df['prediction'].apply(lambda x: adjust_prediction(x, par_value))
    squared_diff = (df_test1['stars'] - predicted_df['prediction']) ** 2
    # Step 2: Calculate mean of squared differences
    mse = np.mean(squared_diff)

    rmse = np.sqrt(mse)
    print("RMSE:", rmse)


    # Define the error ranges
    error_ranges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, float('inf'))]

    # Initialize counts for each error range
    error_counts = [0] * len(error_ranges)

    # Iterate through predictions and calculate error distribution
    for prediction, actual_value in zip(predicted_df['prediction'], df_test1['stars']):
        error = abs(prediction - actual_value)
        for i, (lower, upper) in enumerate(error_ranges):
            if lower <= error < upper:
                error_counts[i] += 1
                break

    # Calculate total predictions
    total_predictions = sum(error_counts)

    # Calculate proportion of predictions falling within each error range
    error_distribution = [count / total_predictions for count in error_counts]

    # Print error distribution
    for i, (lower, upper) in enumerate(error_ranges):
        if upper == float('inf'):
            print(f"Error >= {lower}: {error_counts[i]} ({error_distribution[i]*100:.2f}%)")
        else:
            print(f"{lower} <= Error < {upper}: {error_counts[i]}  ({error_distribution[i]*100:.2f}%)")
    return predicted_df



predicted_df = get_prediction()   
#print(predicted_df.head())
write_predictions_to_file(dir_final_op, predicted_df)
end = time.time()
duration = end - start         
print(duration)

#0.97715