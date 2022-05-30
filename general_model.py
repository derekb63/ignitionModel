import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from tabulate import tabulate


def split_data(data, parameters, outcomes, test_proportion=0.25, state_num=42):
    # Split the data into the inputs (features) and outcomes to train for (labels)
    features = np.array(data[parameters])
    labels = data[outcomes]
    x_train, x_valid, y_train, y_valid = train_test_split(features,
                                                          labels,
                                                          test_size=1-test_proportion,
                                                          random_state=state_num)
    return x_train, x_valid, y_train, y_valid


def confusion_matrix_calc(model, test_data, actual_result, print_cm=True):
    cm = confusion_matrix(actual_result, model.predict(test_data))
    cm_list = cm.tolist()
    cm_list[0].insert(0, 'Real True')
    cm_list[1].insert(0, 'Real False')
    if print_cm is True:
        print(tabulate(cm_list, headers=['Real/Pred', 'Pred True', 'Pred False']))
    return cm


def split_and_score_data(model, test_data, parameter_list, outcome_col, cm_print=False):
    x_test = test_data[parameter_list].to_numpy()
    y_test = test_data[outcome_col].to_numpy()
    r2_score = model.score(x_test, y_test)
    cm = confusion_matrix_calc(model, x_test, y_test, print_cm=cm_print)
    return r2_score, cm


if __name__ == "__main__":
    # Load in the datasets
    data_from_experiments = pd.read_csv('data_from_experiments.csv', index_col=0)
    urban_data = pd.read_csv('urban_2018_data.csv', index_col=0)
    zak_data = pd.read_csv('zak_2014_data.csv', index_col=0)
    hadden_data = pd.read_csv('hadden_2011_data.csv', index_col=0)
    filkov_data = pd.read_csv('filkov_2016_data.csv', index_col=0)

    # Define the features that will be used in the models
    feature_list = ['mean_temperature', 'density', 'wind_speed', 'particle_size', 'heater_lead_angle']
    # Define the column that contains the ignition outcomes of the test defined by data
    outcome_column = 'ignition'

    # initialize the Random Forest Classifier model
    classifier_model = RandomForestClassifier(n_estimators=1000, criterion='gini', oob_score=True)

    # Train the classifier on the training portion of the data from the experments
    x_train_exp, x_test_exp, y_train_exp, y_test_exp = split_data(data_from_experiments, feature_list, outcome_column)
    classifier_model.fit(x_train_exp, y_train_exp.values.ravel())
    print('###############################################################')
    print('Initial model including only data from experimental efforts')
    print('###############################################################')
    # Test the new model on the data From experiments
    print()
    print('Initial Model Experimental Data Confusion Matrix')
    r2_exp, _ = split_and_score_data(classifier_model, data_from_experiments, feature_list, outcome_column, cm_print=True)
    print('R^2 Validation Score: {:.2f}'.format(r2_exp))

    # Test the new model on the datasets
    # Start with the Hadden data
    print()
    print('Initial Model Hadden et al. Confusion Matrix')
    r2_hadden, _ = split_and_score_data(classifier_model, hadden_data, feature_list, outcome_column, cm_print=True)
    print('R^2 Validation Score: {:.2f}'.format(r2_hadden))

    # Zak Data
    print()
    print('Initial Model Zak et al. Confusion Matrix')
    r2_zak, _ = split_and_score_data(classifier_model, zak_data, feature_list, outcome_column, cm_print=True)
    print('R^2 Validation Score: {:.2f}'.format(r2_zak))

    # Urban Data
    print()
    print('Initial Model Urban et al. Confusion Matrix')
    r2_urban, _ = split_and_score_data(classifier_model, urban_data, feature_list, outcome_column, cm_print=True)
    print('R^2 Validation Score: {:.2f}'.format(r2_urban))

    # Filkov Data
    print()
    print('Initial Model Filkov et al. Confusion Matrix')
    r2_filkov, _ = split_and_score_data(classifier_model, filkov_data, feature_list, outcome_column, cm_print=True)
    print('R^2 Validation Score: {:.2f}'.format(r2_filkov))
########################################################################################################################
    print('###############################################################')
    print('Expanded model including data from Filkov et al. and Zak et al.')
    print('###############################################################')
    # create a new model that includes some of the data from Zak and Folkov
    # combine the datasets
    expanded_data = pd.concat([data_from_experiments, zak_data, filkov_data], ignore_index=True)
    # initialize the Random Forest Classifier model
    expanded_classifier = RandomForestClassifier(n_estimators=1000, criterion='gini', oob_score=True)

    # Train the classifier on the training portion of the data from the experiments
    x_train_comb, x_test_comb, y_train_comb, y_test_comb = split_data(expanded_data, feature_list, outcome_column)
    expanded_classifier.fit(x_train_comb, y_train_comb.values.ravel())

    # Test the new model on the data From experiments
    print()
    print('Expanded Model Experimental Data Confusion Matrix')
    r2_exp, _ = split_and_score_data(expanded_classifier, data_from_experiments, feature_list, outcome_column, cm_print=True)
    print('R^2 Validation Score: {:.2f}'.format(r2_exp))

    # Test the new model on the datasets
    # Start with the Hadden data
    print()
    print('Expanded Model Hadden et al. Confusion Matrix')
    r2_hadden, _ = split_and_score_data(expanded_classifier, hadden_data, feature_list, outcome_column, cm_print=True)
    print('R^2 Validation Score: {:.2f}'.format(r2_hadden))

    # Zak Data
    print()
    print('Expanded Model Zak et al. Confusion Matrix')
    r2_zak, _ = split_and_score_data(expanded_classifier, zak_data, feature_list, outcome_column, cm_print=True)
    print('R^2 Validation Score: {:.2f}'.format(r2_zak))

    # Urban Data
    print()
    print('Expanded Model Urban et al. Confusion Matrix')
    r2_urban, _ = split_and_score_data(expanded_classifier, urban_data, feature_list, outcome_column, cm_print=True)
    print('R^2 Validation Score: {:.2f}'.format(r2_urban))

    # Filkov Data
    print()
    print('Expanded Model Filkov et al. Confusion Matrix')
    r2_filkov, _ = split_and_score_data(expanded_classifier, filkov_data, feature_list, outcome_column, cm_print=True)
    print('R^2 Validation Score: {:.2f}'.format(r2_filkov))









