import numpy as np
import matplotlib
import time
import tqdm
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage.transform import resize
from model import SVM

# The imports below are added for the various versions of the solution
# that we have implemented here. They should not be given to the students by
# default.
from numpy.linalg import norm
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cdist
from scipy.stats import mode

def get_tiny_images(image_paths, extra_credit=False):
    '''
    This feature is inspired by the simple tiny images used as features in
    80 million tiny images: a large dataset for non-parametric object and
    scene recognition. A. Torralba, R. Fergus, W. T. Freeman. IEEE
    Transactions on Pattern Analysis and Machine Intelligence, vol.30(11),
    pp. 1958-1970, 2008. http://groups.csail.mit.edu/vision/TinyImages/

    Inputs:
        image_paths: a 1-D Python list of strings. Each string is a complete
                     path to an image on the filesystem.
    Outputs:
        An n x d numpy array where n is the number of images and d is the
        length of the tiny image representation vector. e.g. if the images
        are resized to 16x16, then d is 16 * 16 = 256.

    To build a tiny image feature, resize the original image to a very small
    square resolution (e.g. 16x16). You can either resize the images to square
    while ignoring their aspect ratio, or you can crop the images into squares
    first and then resize evenly. Normalizing these tiny images will increase
    performance modestly.

    As you may recall from class, naively downsizing an image can cause
    aliasing artifacts that may throw off your comparisons. See the docs for
    skimage.transform.resize for details:
    http://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.resize

    Suggested functions: skimage.transform.resize, skimage.color.rgb2gray,
                         skimage.io.imread, np.reshape
    '''

    image_feats = np.zeros((len(image_paths), 256))
    dims = (1,256)

    for s, path in enumerate(image_paths):
        image = imread(path)

        if (len(image.shape) == 3) and (image.shape[2] == 3):
            image = rgb2gray(image)

        img = resize(image, (16, 16), anti_aliasing=True)
        image_feats[s, :] = np.reshape(img, dims, order="F")
    
    return image_feats

def build_vocabulary(image_paths, vocab_size, extra_credit=False):
    '''
    This function should sample HOG descriptors from the training images,
    cluster them with kmeans, and then return the cluster centers.

    Inputs:
        image_paths: a Python list of image path strings
         vocab_size: an integer indicating the number of words desired for the
                     bag of words vocab set

    Outputs:
        a vocab_size x (z*z*9) (see below) array which contains the cluster
        centers that result from the K Means clustering.

    You'll need to generate HOG features using the skimage.feature.hog() function.
    The documentation is available here:
    http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.hog

    However, the documentation is a bit confusing, so we will highlight some
    important arguments to consider:
        cells_per_block: The hog function breaks the image into evenly-sized
            blocks, which are further broken down into cells, each made of
            pixels_per_cell pixels (see below). Setting this parameter tells the
            function how many cells to include in each block. This is a tuple of
            width and height. Your SIFT implementation, which had a total of
            16 cells, was equivalent to setting this argument to (4,4).
        pixels_per_cell: This controls the width and height of each cell
            (in pixels). Like cells_per_block, it is a tuple. In your SIFT
            implementation, each cell was 4 pixels by 4 pixels, so (4,4).
        feature_vector: This argument is a boolean which tells the function
            what shape it should use for the return array. When set to True,
            it returns one long array. We recommend setting it to True and
            reshaping the result rather than working with the default value,
            as it is very confusing.

    It is up to you to choose your cells per block and pixels per cell. Choose
    values that generate reasonably-sized feature vectors and produce good
    classification results. For each cell, HOG produces a histogram (feature
    vector) of length 9. We want one feature vector per block. To do this we
    can append the histograms for each cell together. Let's say you set
    cells_per_block = (z,z). This means that the length of your feature vector
    for the block will be z*z*9.

    With feature_vector=True, hog() will return one long np array containing every
    cell histogram concatenated end to end. We want to break this up into a
    list of (z*z*9) block feature vectors. We can do this using a really nifty numpy
    function. When using np.reshape, you can set the length of one dimension to
    -1, which tells numpy to make this dimension as big as it needs to be to
    accomodate to reshape all of the data based on the other dimensions. So if
    we want to break our long np array (long_boi) into rows of z*z*9 feature
    vectors we can use small_bois = long_boi.reshape(-1, z*z*9).

    The number of feature vectors that come from this reshape is dependent on
    the size of the image you give to hog(). It will fit as many blocks as it
    can on the image. You can choose to resize (or crop) each image to a consistent size
    (therefore creating the same number of feature vectors per image), or you
    can find feature vectors in the original sized image.

    ONE MORE THING
    If we returned all the features we found as our vocabulary, we would have an
    absolutely massive vocabulary. That would make matching inefficient AND
    inaccurate! So we use K Means clustering to find a much smaller (vocab_size)
    number of representative points. We recommend using sklearn.cluster.KMeans
    to do this. Note that this can take a VERY LONG TIME to complete (upwards
    of ten minutes for large numbers of features and large max_iter), so set
    the max_iter argument to something low (we used 100) and be patient. You
    may also find success setting the "tol" argument (see documentation for
    details)
    '''

    # Allocate this so we can fill it later.
    # For non-variable sizing (100 features per image), uncomment this
    #features = np.zeros((len(image_paths) * 100, 36))

    # Create a base array that we can concatenate arbitrary numbers of
    # feature vectors on to for variable sizing
    features = np.zeros((1,36))
    
    num_imgs = len(image_paths)

    # Iterate over each path to load
    for i in tqdm(range(num_imgs), desc="Building Vocab"):
                         
        path = image_paths[i]
        # FOR NON-VARIABLE SIZING:
        # We resize every image to the same size, 176x176, when we read them in.
        # Specifically, it gives us 100 36-length vectors.
        #img = resize(rgb2gray(imread(path)), (176, 176), anti_aliasing=True)

        # Use this one for variable sizing (no resize = different number of features per img)
        img = imread(path)
        
        if (len(img.shape) == 3) and (img.shape[2] == 3):
            img = rgb2gray(img)

        # Extract HOG features for the image
        this_img_features = hog(img, pixels_per_cell=(16, 16), cells_per_block=(2,2), feature_vector=True)
        this_img_features = this_img_features.reshape(-1, 36) # see block comment

        #add to the big feature matrix

        # FOR NON-VARIABLE SIZING: 100 features, sliced right in all nice and neat
        #features[100 * i : 100 * (i + 1), :] = this_img_features

        # FOR VARIABLE SIZING: messy append
        features = np.append(features, this_img_features, axis=0)

    # FOR VARIABLE SIZING:
    # Remove the weird all-zero vector we needed to get append to work
    features = features[1:]

    # Cluster the extracted features using vocab_size clusters, then save the centroids
    # Note: we can add the argument random_state=[integer] to give this a constant seed for testing purposes
    # We use 50 max_iter for variable sizing and 100 for non-variable, just because with 100 and
    # all the extra features from variable sizing it takes FOREVERRR
    vocab = KMeans(n_clusters=vocab_size, max_iter=50).fit(features).cluster_centers_

    return vocab

def get_bags_of_words(image_paths, vocab, extra_credit=False):
    '''
    This function should take in a list of image paths and calculate a bag of
    words histogram for each image, then return those histograms in an array.

    Inputs:
        image_paths: A Python list of strings, where each string is a complete
                     path to one image on the disk.

    Outputs:
        An nxd numpy matrix, where n is the number of images in image_paths and
        d is size of the histogram built for each image.

    Use the same hog function to extract feature vectors as before (see
    build_vocabulary). It is important that you use the same hog settings for
    both build_vocabulary and get_bags_of_words! Otherwise, you will end up
    with different feature representations between your vocab and your test
    images, and you won't be able to match anything at all!

    After getting the feature vectors for an image, you will build up a
    histogram that represents what words are contained within the image.
    For each feature, find the closest vocab word, then add 1 to the histogram
    at the index of that word. For example, if the closest vector in the vocab
    is the 103rd word, then you should add 1 to the 103rd histogram bin. Your
    histogram should have as many bins as there are vocabulary words.

    Suggested functions: scipy.spatial.distance.cdist, np.argsort,
                         np.linalg.norm, skimage.feature.hog
    '''

    vocab_size = vocab.shape[0]

    image_feats = np.zeros((len(image_paths), vocab_size))

    for i,path in enumerate(image_paths):
        # FOR NON-VARIABLE SIZING:
        # In order to always get the same number of HOG features from the image,
        # we have to scale it to a uniform size.
        #img = resize(rgb2gray(imread(path)), (176, 176), anti_aliasing=True)

        # FOR VARIABLE SIZING:
        img = imread(path)
        
        if (len(img.shape) == 3) and (img.shape[2] == 3):
            img = rgb2gray(img)

        # Extract HOG features for the image
        features = hog(img, pixels_per_cell=(16,16), cells_per_block=(2,2), feature_vector=True)
        features = features.reshape(-1, 36)

        # Use cdist to find the distance to each word in the vocab from each
        # feature in the image
        distances = cdist(features, vocab, 'euclidean')

        # Build a histogram of word hits per image
        # That is, make a histogram the same length as the number of words in
        # the vocabulary and add 1 to the matching bin for whichever vocab
        # feature is closest to the current image feature.
        hist = np.zeros(vocab_size)
        for d in distances:
            indices = np.argsort(d)
            hist[indices[0]] += 1

        # Normalize the histogram
        hist = hist / norm(hist)
        image_feats[i,:] = hist

    return image_feats

def svm_classify(train_image_feats, train_labels, test_image_feats, extra_credit=False):
    '''
    This function will predict a category for every test image by training
    15 many-versus-one linear SVM classifiers on the training data, then
    using those learned classifiers on the testing data.

    Inputs:
        train_image_feats: An nxd numpy array, where n is the number of training
                           examples, and d is the image descriptor vector size.
        train_labels: An nx1 Python list containing the corresponding ground
                      truth labels for the training data.
        test_image_feats: An mxd numpy array, where m is the number of test
                          images and d is the image descriptor vector size.

    Outputs:
        An mx1 numpy array of strings, where each string is the predicted label
        for the corresponding image in test_image_feats

    Look at the model.py file to find the right arguments to train each classifier.
    Remember to go over your pseudocode from the written homework
    '''
    # Train the model on the training images
    
    categories = ['Kitchen', 'Store', 'Bedroom', 'LivingRoom', 'Office',
           'Industrial', 'Suburb', 'InsideCity', 'TallBuilding', 'Street',
           'Highway', 'OpenCountry', 'Coast', 'Mountain', 'Forest']

    predicted_scores = np.zeros([test_image_feats.shape[0], 15])
    model = SVM()

	# Train classification hyperplane
    for label in range(0, 15):
		# normalise training labels to 1 or -1 as we are dealing with a binary classifier
		# using one vs all approach. set one of the class to 1.
        new_train_labels = np.where(np.asarray(train_labels) == categories[label], 1, -1)
        weights, bias = model.train(train_image_feats, new_train_labels)
		# compute prediction score
        scores = [np.dot(weights, test_image_feats[i, :].T) + bias for i in range(test_image_feats.shape[0])]
        predicted_scores[:, label] = np.asarray(scores).T

	# returns index of max value i.e predicted class label
    preds = np.argmax(predicted_scores, axis=1)
    preds = [categories[pred] for pred in preds]
    
    return np.asarray(preds)

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats, extra_credit=False):
    '''
    This function will predict the category for every test image by finding
    the training image with most similar features. You will complete the given
    partial implementation of k-nearest-neighbors such that for any arbitrary
    k, your algorithm finds the closest k neighbors and then votes among them
    to find the most common category and returns that as its prediction.

    Inputs:
        train_image_feats: An nxd numpy array, where n is the number of training
                           examples, and d is the image descriptor vector size.
        train_labels: An nx1 Python list containing the corresponding ground
                      truth labels for the training data.
        test_image_feats: An mxd numpy array, where m is the number of test
                          images and d is the image descriptor vector size.

    Outputs:
        An mx1 numpy list of strings, where each string is the predicted label
        for the corresponding image in test_image_feats

    The simplest implementation of k-nearest-neighbors gives an even vote to
    all k neighbors found - that is, each neighbor in category A counts as one
    vote for category A, and the result returned is equivalent to finding the
    mode of the categories of the k nearest neighbors. A more advanced version
    uses weighted votes where closer matches matter more strongly than far ones.
    This is not required, but may increase performance.

    Be aware that increasing k does not always improve performance - even
    values of k may require tie-breaking which could cause the classifier to
    arbitrarily pick the wrong class in the case of an even split in votes.
    Additionally, past a certain threshold the classifier is considering so
    many neighbors that it may expand beyond the local area of logical matches
    and get so many garbage votes from a different category that it mislabels
    the data. Play around with a few values and see what changes.

    Useful functions:
        scipy.spatial.distance.cdist, np.argsort, scipy.stats.mode
    '''

    # This commented-out code runs the built-in K nearest neighbors classifier
    # Mostly it's here for comparison against our implementation

    # 1-nearest neighbor works really well actually
    # 5-nearest is slightly better than 1-nearest
    # everything else is TRASH
    # classifier = KNeighborsClassifier(n_neighbors = 5)
    # classifier.fit(train_image_feats, train_labels)
    #
    # return classifier.predict(test_image_feats)

    k = 5

    # Find the distance between each test image feature and train image feature
    distances = cdist(test_image_feats, train_image_feats, 'euclidean')

    # Sort the distance lists to put smallest up front
    # We use argsort because we don't care about the dists themselves, just the
    # feature points they correspond to
    sorted_indices = np.argsort(distances, axis=1)

    # Slice out the k nearest points from the index list
    k_nearest = sorted_indices[:,:k]

    # Convert train_labels to a numpy array so we can index into it with another array
    train_labels = np.array(train_labels)

    # Flatten our nearest points into one long list, index into labels with it
    # to get a list of labels, then reshape back to a series of k-length vectors
    labeled = train_labels[k_nearest.flatten()].reshape(-1, k)

    # Get the mode of every k-length list, then flatten to one array and return
    predictions = mode(labeled, keepdims=True, axis=1).mode.flatten()
    return predictions