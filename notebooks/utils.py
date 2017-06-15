from time import time
def desc_samples(data):
    
    n_samples = len(data.index)
    n_features = data.shape[1]-1

    # Print the results
    print "Total number of samples: {}".format(n_samples)
    print "Number of features: {}".format(n_features)

    feature_cols = list(data.columns[:-1])
    # Extract target column
    target_col = data.columns[-1] 
    # Show the list of columns
    print "Feature columns:\n{}".format(feature_cols)
    print "\nTarget column: {}".format(target_col)

    # Separate the data into feature data and target data (X_all and y_all, respectively)
    X_all = data[feature_cols]
    y_all = data[target_col]
    # Show the feature information by printing the first five rows
    print "\nFeature values:"
    print X_all.head()
    return X_all, y_all
    
    
def split_samples_train_test(percent, X_all, y_all):
    from sklearn.model_selection import train_test_split
    num_train = percent
    # Set the number of testing points
    num_test = X_all.shape[0] - num_train
    # TODO: Shuffle and split the dataset into the number of training and testing points above
    ## Taking the sugestion of the reviwer to use stratify (nice! Thanks!)
    X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, train_size=num_train, random_state=42)
    # Show the results of the split
    print "Training set has {} samples.".format(X_train.shape[0])
    print "Testing set has {} samples.".format(X_test.shape[0])
    return X_train, X_test, y_train, y_test


def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Print the results
    print "Trained model in {:.4f} seconds".format(end - start)

    
def predict_labels(clf, features, target):
    from sklearn.metrics import f1_score
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()
    
    # Print and return results
    print "Made predictions in {:.4f} seconds.".format(end - start)
    return f1_score(target.values, y_pred, pos_label='yes')


def train_predict(clf, X_train, y_train, X_test, y_test):
    from sklearn.metrics import f1_score
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    print "Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train))
    
    # Train the classifier
    train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    print "F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test))