import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from sklearn import preprocessing, cross_validation, neighbors, svm
import json
import datetime
from statistics import mean
from matplotlib import style
import matplotlib.pyplot as plt
style.use("ggplot")


'''


The below codes only belongs to question 4: BI open question



'''



# Utils

def handle_non_numerical_data(df):
	columns = df.columns.values
	num_str_dict = {}
	for column in columns:
		val_num_dict = {}
		num_val_dict = {}
		def convert_to_int(val):
			return val_num_dict[val]
		if df[column].dtype != np.int64 and df[column].dtype != np.float64:
			column_contents = df[column].values.tolist()
			unique_elements = set(column_contents)
			x = 0
			for unique in unique_elements:
				if unique not in val_num_dict:
					val_num_dict[unique] = x
					num_val_dict[x] = unique
					x += 1
			num_str_dict[str(column)] = num_val_dict
			df["%s_num"%str(column)] = list(map(convert_to_int, df[column]))
	num_str_dict['occupation'] = get_occupation_dict()
	num_str_dict['age'] = get_age_dict()
	return df, num_str_dict

def convert_to_time(df):
	timestamp_list = df['timestamp'].values.tolist();
	year_list = []
	for timestamp in timestamp_list:
		year = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m')
		year_list.append(year)
	# return datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
	return year_list

def handle_zipcode(df):
	zipcode_list = df['zip'].values.tolist()
	with open("zipcode_data.json", "r") as f:
		zipcode_database_dict = json.loads(f.read())
	state_list = []
	for i in range(len(zipcode_list)):
		if len(zipcode_list[i]) > 5:
			zipcode_list[i] = zipcode_list[i][:5]
	for zipcode in zipcode_list:
		try:
			state = zipcode_database_dict[str(zipcode)]
		except:
			pass
		state_list.append(state)
	return state_list

def clean_data(data):
	data['state'] = handle_zipcode(data)
	data['time'] = convert_to_time(data)
	df1 = data['zip']
	df2 = data[['title', 'gender']]
	df3 = data.drop(['zip', 'title'], 1)
	df, str_num_dict = handle_non_numerical_data(df3)
	df['zip'] = df1
	df['title'] = df2['title']
	df['gender'] = df2['gender']
	return df, str_num_dict

def is_genre_contained(target, genre):
	genre_list = genre.split("|")
	for genre in genre_list:
		if genre == target:
			return True
	return False

def generate_genres_set(df):
	genres_list_from_data = set(df['genres'].values.tolist())
	new_genres_set = set()
	for genres_combination in genres_list_from_data:
		each_genres_list = genres_combination.split("|")
		for each_genre in each_genres_list:
			new_genres_set.add(each_genre)
	return new_genres_set

def get_occupation_dict():
	return {0:"other", 1:"academic/educator", 2:"artist", 3:"clerical/admin", 4:"college/grad student",
			5:"customer service", 6:"doctor/health care", 7:"executive/managerial", 8:"farmer",
			9:"homemaker", 10:"K-12 student", 11:"lawyer", 12:"programmer", 13:"retired",
			14:"sales/marketing", 15:"scientist", 16:"self-employed", 17:"technician/engineer",
			18:"tradesman/craftsman", 19:"unemployed", 20:"writer"}

def get_age_dict():
	return {1:" Under 18",18:"18-24",25:"25-34",35:"35-44",45:"45-49",50:"50-55",56:"56+"}

def get_genres_set():
	return ['Horror', 'Animation', 'Romance', "Children's", 'Western', 'Crime', 
 				'Film-Noir', 'Comedy', 'Adventure', 'Musical', 'Thriller', 'War', 
 				'Documentary', 'Mystery', 'Action', 'Fantasy', 'Sci-Fi', 'Drama']

# Get Figure_4_1

def add_a_genre(df, genre):
	genres_original_list = df['genres'].values.tolist()
	for i in range(len(genres_original_list)):
		if contains(genre, genres_original_list[i]):
			genres_original_list[i] = genre
		else:
			genres_original_list[i] = np.nan
	df[genre] = genres_original_list
	return df

def create_split_genres(df, genres_set):
	df4 = df[['rating','genres']]
	for genre in genres_set:
		df4 = add_a_genre(df4, genre)
	return df4

def contains(str_short, str_long):
	str_long_list = str_long.split("|")
	if str_short in str_long_list:
		return True
	return False

def bar_plot_genres_info_4_1(df, genres_set):
	genres_mean_list = []
	for i in range(len(genres_set)):
		genres_mean = df.groupby(genres_set[i])['rating'].count().values
		genres_mean_list.append(list(genres_mean)[0])

	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.bar(range(len(genres_mean_list)), genres_mean_list, align='center', color = 'r', width=0.8)
	ax1.set_xticks(range(len(genres_set)))
	ax1.set_xticklabels(genres_set, rotation = 45, ha = 'right', size=12)
	plt.title("each genres's average score")
	plt.xlabel("genres")
	plt.ylabel('average score')
	plt.legend()
	plt.tight_layout()
	plt.show()

def plot_Figure_4_1(data):
	genres_set = get_genres_set()
	df = create_split_genres(data, genres_set)
	bar_plot_genres_info_4_1(df, genres_set)

# get_Figure_4_2

def get_male_female_rater_number(data):
	series_male_female = data.groupby('gender')['rating'].count()
	male_rater_number = series_male_female.ix['M']
	female_rater_number = series_male_female.ix['F']
	return male_rater_number, female_rater_number

def get_genres_count_list(data):
	genres_set = get_genres_set()
	df = create_split_genres2(data, genres_set)
	return relationship_genres_gender(df, genres_set)

def create_split_genres2(df, genres_set):
	df4 = df[['rating','gender','genres']]
	for genre in genres_set:
		df4 = add_a_genre(df4, genre)
	return df4

def relationship_genres_gender(df, genres_set, aggfunc='count'):
	genres_count_list = []
	for i in range(len(genres_set)):
		arg = []
		df1 = df.pivot_table('rating', index=genres_set[i], columns='gender', aggfunc=aggfunc)
		male_arg = df1['M'].values[0]
		arg.append(male_arg)
		female_arg = df1['F'].values[0]	
		arg.append(female_arg)
		genres_count_list.append(arg)
	return genres_count_list

def bar_plot_genres_info_4_2(x, y, title = '', xlabel = '', ylabel=''):
	y_male = [i[0] for i in y]
	y_female = [j[1] for j in y]
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	x_label = [i-0.4 for i in range(len(y))]
	ax1.bar(x_label, y_male, align='center', color = 'r', label = 'Male', width=0.4)
	ax1.bar(range(len(y)), y_female, align='center', color = 'c', label = 'Female', width=0.4)
	ax1.set_xticks(range(len(x)))
	ax1.set_xticklabels(x, rotation = 45, ha = 'right', size=12)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.legend()
	plt.tight_layout()
	plt.show()

def plot_Figure_4_2(data):
	genres_set = get_genres_set()
	male_rater_number, female_rater_number = get_male_female_rater_number(data)
	# genres_count_list = get_genres_count_list(data)		# this step is slow, so store result in next line
	genres_count_list = [[61751, 14635], [31072, 12221], [97226, 50297], [50869, 21317], [17206, 3477], [63099, 16442], [14059, 4202], [260309, 96271], [106621, 27332], [28028, 13505], [149372, 40308], [54434, 14093], [5970, 1940], [30202, 9976], [211807, 45650], [27583, 8718], [129894, 27400], [256376, 98153]]
	genres_count_list_ratio = [[int(i[0]*100/male_rater_number), int(i[1]*100/female_rater_number)] for i in genres_count_list]
	bar_plot_genres_info_4_2(genres_set, genres_count_list_ratio, title = 'ratio of gender for each genre', xlabel = 'genres', ylabel='percentage of gender (%)')

# get_Figure_4_3


def plot_scatter_linear_regression_4_3(rating_set, genres_set, title='', xlabel = '', ylabel=''):
	xs = np.array([i[0] for i in rating_set])
	ys = np.array([i[1] for i in rating_set])
	m,b = best_fit_slope_and_intercept(xs, ys)
	regression_line = [(m*x)+b for x in xs]
	
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.scatter(xs, ys, s=60)

	genres_man_women_mean_dict = {}
	for i in range(len(genres_set)):
		genres_man_women_mean_dict[str(rating_set[i])] = genres_set[i]
	best = max(rating_set)
	worst = min(rating_set)
	ax1.text(best[0], best[1]-0.5, 'Genre: \n\"%s\"'%genres_man_women_mean_dict[str(best)], fontsize=15, 
					horizontalalignment='center', verticalalignment='center')
	ax1.text(worst[0]+0.5, worst[1], 'Genre: \n\"%s\"'%genres_man_women_mean_dict[str(worst)], fontsize=15, 
					horizontalalignment='center', verticalalignment='center')
	ax1.annotate('', xy=best , xytext=(best[0], best[1]-0.5), 
					arrowprops=dict(facecolor='red', shrink=0.2))
	ax1.annotate('', xy=worst , xytext=(worst[0]+0.5, worst[1]), 
					arrowprops=dict(facecolor='red', shrink=0.2))
	ax1.plot(xs, regression_line)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.show()

def best_fit_slope_and_intercept(xs, ys):
	m = (((mean(xs)*mean(ys)) - mean(xs*ys)) / 
			((mean(xs))**2 - mean(xs**2)))
	b = mean(ys) - m*mean(xs)
	return m, b

def get_genres_mean_list(data):
	genres_set = get_genres_set()
	df = create_split_genres2(data, genres_set)
	return relationship_genres_gender(df, genres_set, aggfunc='mean')

def plot_Figure_4_3(data):
	genres_set = get_genres_set()
	# genres_mean_list = get_genres_mean_list(data)				# This line of code is slow, so store result in next line
	genres_mean_list = [[3.2178912082395428, 3.2028698325930987], [3.6613349639546859, 3.744701742901563], [3.5732622960936373, 3.673578941089926], [3.3589612534156363, 3.572547731857203], [3.6551203068696965, 3.5519125683060109], [3.7137197102965183, 3.6893321980294367], [4.0922540721246179, 4.0180866254164682], [3.5036667960001382, 3.5719375512875113], [3.468125416193808, 3.5128786770086347], [3.5963322391893819, 3.8091077378748612], [3.5696850815413868, 3.5733601270219313], [3.8933754638644964, 3.893138437522174], [3.9288107202680065, 3.9463917525773198], [3.662009138467651, 3.6865477145148358], [3.4913860259575933, 3.4902519167579409], [3.4266033426385816, 3.5130763936682725], [3.4699524227446994, 3.4502554744525549], [3.7665889162792148, 3.7656617729463187]]
	plot_scatter_linear_regression_4_3(genres_mean_list, genres_set, 
									title="Correlation between M/F's ratings in different genres", 
									xlabel = 'Male average rating', 
									ylabel=	'Female average rating')


# get Figure_4_4

'''
relations_occupation_genres_interest described how many votes on a specific genre among different occupations
This function focus on rating count
'''
def relations_occupation_genres_interest(df, genre):
	df3 = choose_a_genre(df, genre)
	df_occupation_genre = df3.groupby('occupation')[genre]
	occupation_focus_list = df_occupation_genre.count().tolist()
	occupation_dict = get_occupation_dict()
	occupation_list = [occupation_dict[i] for i in range(21)]	
	_plot_bar(occupation_list, occupation_focus_list, genre,
			title = 'relation of numbers of ratings on %s versus occupations'%genre, 
			xlabel = "occupations", ylabel='rating count')
'''
choose_a_genre function add a column named genre's name(eg. Action) to df, which values are either genre's game or NaN
'''
def choose_a_genre(df, genre):
	df_genre = df[['rating', 'genres', 'occupation']]
	genres_original_list = df_genre['genres'].values.tolist()
	for i in range(len(genres_original_list)):
		if contains(genre, genres_original_list[i]):
			genres_original_list[i] = genre
		else:
			genres_original_list[i] = np.nan
	df_genre[genre] = genres_original_list
	return df_genre

def _plot_bar(x, y, genre_name, title = "", xlabel = "", ylabel=""):
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.bar(range(len(y)), y, align='center', label=genre_name, color = 'c', width=0.8)
	ax1.set_xticks(range(len(x)))
	ax1.set_xticklabels(x, rotation = 45, ha = 'right', size=13)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.legend()
	plt.tight_layout()
	plt.show()

def plot_Figure_4_4(data, genre):
	relations_occupation_genres_interest(data, genre)


# get Figure_4_5

'''
The function below plot a ratio of one occupation who are interested in a specific genre
for example: for those who are writers, what is the percentage of them who are interested in Action genre?
'''
def relations_occupationRatio_genres_interest(df, genre):
	# Clean data and add a specific genre
	df3 = choose_a_genre(df, genre)

	# Obtain occupation_focus_list, which is [vote number to genre of occupation 0, vote number to genre of occupation 1, .....]
	df_occupation_genre = df3.groupby('occupation')[genre]
	occupation_focus_list = np.array(df_occupation_genre.count().tolist())

	# Obtain occupation_raters_list, which is [overall vote numbers of occupation 0, overall vote numbers of occupation 1, .....]
	# convert to nparray for calculation
	occupation_raters_list = df3.groupby('occupation')['rating'].count().tolist()
	occupation_raters_nparray = np.array(occupation_raters_list)

	
	# calcualte and obtain occupation_ratio_list, which is
	# [ratio of voting this genre of occupation 0, ratio of voting this genre of occupation 1, .....]
	occupation_ratio_list = list(occupation_focus_list*100/occupation_raters_nparray)
	
	# Obtain name of occupations
	occupation_dict = get_occupation_dict()
	occupation_list = [occupation_dict[i] for i in range(21)]

	# bar plot	
	_plot_bar(occupation_list, occupation_ratio_list, genre,
			title = 'occupations ratio who is interested in %s'%genre, 
			xlabel = "occupations", ylabel='ratio of occupation who is interested (%)')


def plot_Figure_4_5(data, genre):
	relations_occupationRatio_genres_interest(data, genre)



# get Figure_4_6 

'''
The function below plot the relationships of different occupation's rating on all genres
'''
def linear_regression_relations_genres_2occupationMeanRating(df, genres_set, ocp1, ocp2):

	source = []	# a 2D array: 1st layer is genre, 2nd is score list toward this genre

	for genre in genres_set:
		df3 = choose_a_genre(df, genre)
		df_occupation_genre = df3.pivot_table('rating', index='occupation', columns=genre)
		# Obtained a list of average score on a genra
		occupation_focus_list = df_occupation_genre[genre].values.tolist()
		source.append(occupation_focus_list)

	# Obtained list of all ocp1's scores with order of genres
	xs = np.array([genre[ocp1] for genre in source])
	# Obtained list of all ocp2's scores with order of genres	
	ys = np.array([genre[ocp2] for genre in source])

	# Calculating regression_line
	m,b = best_fit_slope_and_intercept(xs, ys)
	regression_line = [(m*x)+b for x in xs]
	

	# Plotting scatters and regression_line
	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.scatter(xs, ys, s=60)
	ax1.plot(xs, regression_line)

	# generate coefficient_of_determination
	coefficient_of_determination = generate_coefficient_of_determination(ys, regression_line)
	coefficient_of_determination = "%.3f"%coefficient_of_determination

	# codes below are for showing annotations of best and worst
	occupation_dict = get_occupation_dict()
	genres_occupations_mean_dict = {}
	zipped_xs_ys = list(zip(xs, ys))
	for i in range(len(genres_set)):
		genres_occupations_mean_dict[str(zipped_xs_ys[i])] = genres_set[i]
	best = max(zipped_xs_ys)
	worst = min(zipped_xs_ys)
	ax1.text(best[0], best[1]-0.5, 'Genre: \n\"%s\"'%genres_occupations_mean_dict[str(best)], fontsize=15, 
	 				horizontalalignment='center', verticalalignment='center')
	ax1.text(worst[0]+0.5, worst[1], 'Genre: \n\"%s\"'%genres_occupations_mean_dict[str(worst)], fontsize=15, 
	 				horizontalalignment='center', verticalalignment='center')

	ax1.annotate('', xy=best , xytext=(best[0], best[1]-0.5), 
					arrowprops=dict(facecolor='red', shrink=0.2))
	ax1.annotate('', xy=worst , xytext=(worst[0]+0.5, worst[1]), 
	 				arrowprops=dict(facecolor='red', shrink=0.2))

	plt.xlabel("Average scores of occupation: %s"%occupation_dict[ocp1])
	plt.ylabel("Average scores of occupation: %s"%occupation_dict[ocp2])
	plt.title('Correlation: scores of occupations %s and %s on genres\n Coefficient_of_Determination: %s'
						%(occupation_dict[ocp1], occupation_dict[ocp2], str(coefficient_of_determination)))
	plt.show()


def generate_coefficient_of_determination(ys_orig, ys_line):
	y_mean_line = [mean(ys_orig) for y in ys_orig]
	squared_error_regr = squared_error(ys_orig, ys_line)
	squared_error_y_mean = squared_error(ys_orig, y_mean_line)
	return 1 - (squared_error_regr/squared_error_y_mean)
def squared_error(ys_orig, ys_line):
	return sum((ys_line-ys_orig)**2)


def plot_Figure_4_6(data, inta, intb):
	genres_set = get_genres_set()
	linear_regression_relations_genres_2occupationMeanRating(data, genres_set, inta, intb)


# get Figure_4_7

def plot_Figure_4_7(data, inta, intb):
	genres_set = get_genres_set()
	linear_regression_relations_genres_2occupationMeanRating(data, genres_set, inta, intb)



# get Figure_4_8


'''
This function plot state rating count by analyzing zipcode
'''
def plot_state_rating_count(df):

	# shrink df to two columns, and obtain zipcode list
	df = df[['zip', 'rating']]
	zipcode_list = df['zip'].values.tolist()

	# some zipcode are 12345-4322, so remove second extending part
	for i in range(len(zipcode_list)):
		if len(zipcode_list[i]) > 5:
			zipcode_list[i] = zipcode_list[i][:5]

	# open zipcode_data libaray containg state information of each zipcode
	with open("zipcode_data.json", "r") as f:
		zipcode_database_dict = json.loads(f.read())
	
	# obtaining state list corresponding to zipcode with same sequence in df
	state_list = []
	for zipcode in zipcode_list:
		try:
			state = zipcode_database_dict[str(zipcode)]
		except:
			pass
		state_list.append(state)

	# add a new column in df with information of state
	df['state'] = state_list

	# group state and rating information to a serie for plotting
	state_rating_num_group = df.groupby('state')['rating']
	state_rating_num_serie = state_rating_num_group.count()

	# generate x and y for plotting
	x = state_rating_num_serie.index
	y = state_rating_num_serie.values

	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.bar(range(len(y)), y, align='center', color = 'b', width=1.2)
	ax1.set_xticks(range(len(x)))
	ax1.set_xticklabels(x, rotation = 45, ha = 'right', size=11)
	plt.title("average score based on states")
	plt.xlabel("state")
	plt.ylabel('voting count')
	plt.legend()
	plt.tight_layout()
	plt.show()


def plot_Figure_4_8(data):
	plot_state_rating_count(data)



# get Figure_4_9

'''
This function plot state rating count by analyzing zipcode
'''
def plot_state_rating_mean(df):

	# shrink df to two columns, and obtain zipcode list
	df = df[['zip', 'rating']]
	zipcode_list = df['zip'].values.tolist()

	# some zipcode are 12345-4322, so remove second extending part
	for i in range(len(zipcode_list)):
		if len(zipcode_list[i]) > 5:
			zipcode_list[i] = zipcode_list[i][:5]

	# open zipcode_data libaray containg state information of each zipcode
	with open("zipcode_data.json", "r") as f:
		zipcode_database_dict = json.loads(f.read())
	
	# obtaining state list corresponding to zipcode with same sequence in df
	state_list = []
	for zipcode in zipcode_list:
		try:
			state = zipcode_database_dict[str(zipcode)]
		except:
			pass
		state_list.append(state)

	# add a new column in df with information of state
	df['state'] = state_list

	# group state and rating information to a serie for plotting
	state_rating_num_group = df.groupby('state')['rating']
	state_rating_num_serie = state_rating_num_group.mean()

	# generate x and y for plotting
	x = state_rating_num_serie.index
	y = state_rating_num_serie.values

	fig = plt.figure()
	ax1 = fig.add_subplot(111)
	ax1.bar(range(len(y)), y, align='center', color = 'b', width=1.2)
	ax1.set_xticks(range(len(x)))
	ax1.set_xticklabels(x, rotation = 45, ha = 'right', size=11)
	plt.title("average score based on states")
	plt.xlabel("state")
	plt.ylabel('voting count')
	plt.legend()
	plt.tight_layout()
	plt.show()


def plot_Figure_4_9(data):
	plot_state_rating_mean(data)


'''
below are machine learning code

'''

# 1. generate features and saved as a new file

"""
for supervised ml, the columns contains:

# ['male_count' 'female_count' 'occupation_0' 'occupation_1' 'occupation_2'
#  'occupation_3' 'occupation_4' 'occupation_5' 'occupation_6' 'occupation_7'
#  'occupation_8' 'occupation_9' 'occupation_10' 'occupation_11'
#  'occupation_12' 'occupation_13' 'occupation_14' 'occupation_15'
#  'occupation_16' 'occupation_17' 'occupation_18' 'occupation_19'
#  'occupation_20' 'age_50-55' 'age_Under 18' 'age_18-24' 'age_35-44'
#  'age_56+' 'age_25-34' 'age_45-49' 'Horror' 'Animation' 'Romance'
#  "Children's" 'Western' 'Crime' 'Film-Noir' 'Comedy' 'Adventure' 'Musical'
#  'Thriller' 'War' 'Documentary' 'Mystery' 'Action' 'Fantasy' 'Sci-Fi'
#  'Drama' 'average score']

"""

def generate_ml_data(data, new_file_name):
	df_ml = DataFrame()

	# create two columns: male_count, female_count
	gender_info = data.pivot_table('rating', index='movie_id', columns='gender', aggfunc = 'count')
	df_ml['male_count'] = gender_info['M']
	df_ml['female_count'] = gender_info['F']

	df_ml.to_csv(new_file_name, index_col='movie_id')

	# create 21 columns: ocupation_1 ... 20 , each contains occupation counting number
	df_ml = pd.read_csv(new_file_name, index_col='movie_id')
	occupation_info = data.pivot_table('rating', index='movie_id', columns='occupation', aggfunc = 'count')
	for i in range(0, 21):
		df_ml['occupation_%d'%i] = occupation_info[i]

	# create 7 columns representing age count: age_1, age_18.....age_56
	age_info = data.pivot_table('rating', index='movie_id', columns='age', aggfunc = 'count')
	age_dict = {1:"Under 18",18:"18-24",25:"25-34",35:"35-44",45:"45-49",50:"50-55",56:"56+"}
	for k, v in age_dict.items():
		df_ml['age_%s'%v] = age_info[k]

	# create all genres columns: if a movie belongs to one genre, label 1, else: label 0
	genres_set = get_genres_set()
	def create_split_genres_for_ml(df, genres_set):
		df4 = df[['movie_id','genres']]
		for genre in genres_set:
			df4 = add_a_genre(df4, genre)
		return df4
	def add_a_genre(df, genre):
		genres_original_list = df['genres'].values.tolist()
		for i in range(len(genres_original_list)):
			if contains(genre, genres_original_list[i]):
				genres_original_list[i] = 1
			else:
				genres_original_list[i] = np.nan
		df[genre] = genres_original_list
		return df
	def contains(str_short, str_long):
		str_long_list = str_long.split("|")
		if str_short in str_long_list:
			return True
		return False

	df_only_genres = create_split_genres_for_ml(data, genres_set)
	genre_info = df_only_genres.pivot_table(genres_set, index='movie_id', aggfunc = 'mean').fillna(0.0)
	for genre in genres_set:
		df_ml[genre] = genre_info[genre]


	# add average score of each movie as addition column named 'average score'
	average_rating_info = data.groupby('movie_id')['rating'].mean()
	df_ml['average score'] = average_rating_info

	df_ml.to_csv(new_file_name, index_col='movie_id')


# 2. supervised machine learning

def predict_score_class(data, clf = 'K-NN', num_of_classes = 3, runtimes = 10):
	s_ml_data = pd.read_csv(data)
	s_ml_data.drop(['movie_id'], 1, inplace=True)
	s_ml_data.replace(np.nan, -99999, inplace=True)

	def class_average_score(nparray, num_of_classes):
		if num_of_classes > 5:
			print("num_of_classes can not be more than 5")
		def filter_3(x):		
			if 4 <= x:
				return 2
			elif 2 <= x < 4:
				return 1
			else:
				return 0
		if num_of_classes == 2:			# good movie: >4 / else: 1-4
			return {'1': 'rating over 4.0', '0': 'rating below 4.0'}, np.array(list(map(lambda x : 1 if x>=4 else 0, nparray)))
		elif num_of_classes == 3:		# bad: 1-2  / medium: 2-4 / great: 4-5
			return {'2': 'rating over 4.0', '1': 'rating between 2.0 ~ 4.0', '0': 'rating below 2.0'}, \
				np.array(list(map(lambda x : filter_3(x), nparray)))
		elif num_of_classes == 5:		# 0~1, 1~2, 2~3, 3~4, 4~5
			return {'5': 'rating over 4.0', 
					'4': 'rating between 3.0 ~ 4.0', 
					'3': 'rating between 2.0 ~ 3.0',
					'2': 'rating between 1.0 ~ 2.0',
					'0': 'rating below 1.0'}, \
					np.round(nparray)
			
	X = np.array(s_ml_data.drop(['average score'], 1))
	y = np.array(s_ml_data['average score'])
	class_dict, y = class_average_score(y, num_of_classes)
	if clf in ["K-NN", "KNN"]:
		clf = neighbors.KNeighborsClassifier()
	if clf == 'SVM':
		clf = svm.SVC()
	for i in range(runtimes):
		X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
		clf.fit(X_train, y_train)
		accuracy = clf.score(X_test, y_test)
		print("accuracy of classifier: ", accuracy)



'''
Done here
'''
