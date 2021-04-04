"""
A Python implementation of the Viola-Jones ensemble classification method described in 
Viola, Paul, and Michael Jones. "Rapid object detection using a boosted cascade of simple features." Computer Vision and Pattern Recognition, 2001. CVPR 2001. Proceedings of the 2001 IEEE Computer Society Conference on. Vol. 1. IEEE, 2001.
Works in both Python2 and Python3
"""
from numba import cuda
import numba as nb
import numpy as np
import math
import pickle
from sklearn.feature_selection import SelectPercentile, f_classif

class ViolaJones:
    def __init__(self, T = 10):
        """
          Args:
            T: The number of weak classifiers which should be used
        """
        self.T = T
        self.alphas = []
        self.clfs = []

    def train(self, training, pos_num, neg_num, nbImage):
        """
        Trains the Viola Jones classifier on a set of images (numpy arrays of shape (m, n))
          Args:
            training: An array of tuples. The first element is the numpy array of shape (m, n) representing the image. The second element is its classification (1 or 0)
            pos_num: the number of positive samples
            neg_num: the number of negative samples
        """
        if nbImage > len(training):
            nb = len(training)
        else : nb = nbImage
        weights = np.zeros(nb)
        training_data = []
        print("Computing integral images")
        for x in range(nb):
            training_data.append((integral_image(training[x][0]), training[x][1]))
            if training[x][1] == 1:
                weights[x] = 1.0 / (2 * pos_num)
            else:
                weights[x] = 1.0 / (2 * neg_num)

        print("Building features")
        features = self.build_features(training_data[0][0].shape)
        print("Applying features to training examples")
        X, y = self.apply_features(features, training_data)
        print("Selecting best features")
        indices = SelectPercentile(f_classif, percentile=10).fit(X.T, y).get_support(indices=True)
        X = X[indices]
        features = features[indices]
        print("Selected %d potential features" % len(X))

        for t in range(self.T):
            weights = weights / np.linalg.norm(weights)
            weak_classifiers = self.train_weak(X, y, features, weights)
            clf, error, accuracy = self.select_best(weak_classifiers, weights, training_data)
            beta = error / (1.0 - error)
            for i in range(len(accuracy)):
                weights[i] = weights[i] * (beta ** (1 - accuracy[i]))
            alpha = math.log(1.0/beta)
            self.alphas.append(alpha)
            self.clfs.append(clf)
            print("Chose classifier: %s with accuracy: %f and alpha: %f" % (str(clf), len(accuracy) - sum(accuracy), alpha))

    def train_weak(self, X, y, features, weights):
        """
        Finds the optimal thresholds for each weak classifier given the current weights
          Args:
            X: A numpy array of shape (len(features), len(training_data)). Each row represents the value of a single feature for each training example
            y: A numpy array of shape len(training_data). The ith element is the classification of the ith training example
            features: an array of tuples. Each tuple's first element is an array of the rectangle regions which positively contribute to the feature. The second element is an array of rectangle regions negatively contributing to the feature
            weights: A numpy array of shape len(training_data). The ith element is the weight assigned to the ith training example
          Returns:
            An array of weak classifiers
        """
        total_pos, total_neg = 0, 0
        for w, label in zip(weights, y):
            if label == 1:
                total_pos += w
            else:
                total_neg += w

        classifiers = []
        total_features = X.shape[0]
        for index, feature in enumerate(X):
            if len(classifiers) % 1000 == 0 and len(classifiers) != 0:
                print("Trained %d classifiers out of %d" % (len(classifiers), total_features))

            applied_feature = sorted(zip(weights, feature, y), key=lambda x: x[1])

            pos_seen, neg_seen = 0, 0
            pos_weights, neg_weights = 0, 0
            min_error, best_feature, best_threshold, best_polarity = float('inf'), None, None, None
            for w, f, label in applied_feature:
                error = min(neg_weights + total_pos - pos_weights, pos_weights + total_neg - neg_weights)
                if error < min_error:
                    min_error = error
                    best_feature = features[index]
                    best_threshold = f
                    best_polarity = 1 if pos_seen > neg_seen else -1

                if label == 1:
                    pos_seen += 1
                    pos_weights += w
                else:
                    neg_seen += 1
                    neg_weights += w
            
            clf = WeakClassifier(best_feature[0], best_feature[1], best_threshold, best_polarity)
            classifiers.append(clf)
        return classifiers
                
    def build_features(self, image_shape):
        """
        Builds the possible features given an image shape
          Args:
            image_shape: a tuple of form (height, width)
          Returns:
            an array of tuples. Each tuple's first element is an array of the rectangle regions which positively contribute to the feature. The second element is an array of rectangle regions negatively contributing to the feature
        """
        height, width = image_shape
        features = []
        for w in range(1, width+1):
            for h in range(1, height+1):
                i = 0
                while i + w < width:
                    j = 0
                    while j + h < height:
                        #2 rectangle features
                        immediate = RectangleRegion(i, j, w, h)
                        right = RectangleRegion(i+w, j, w, h)
                        if i + 2 * w < width: #Horizontally Adjacent
                            features.append(([right], [immediate]))

                        bottom = RectangleRegion(i, j+h, w, h)
                        if j + 2 * h < height: #Vertically Adjacent
                            features.append(([immediate], [bottom]))
                        
                        right_2 = RectangleRegion(i+2*w, j, w, h)
                        #3 rectangle features
                        if i + 3 * w < width: #Horizontally Adjacent
                            features.append(([right], [right_2, immediate]))

                        bottom_2 = RectangleRegion(i, j+2*h, w, h)
                        if j + 3 * h < height: #Vertically Adjacent
                            features.append(([bottom], [bottom_2, immediate]))

                        #4 rectangle features
                        bottom_right = RectangleRegion(i+w, j+h, w, h)
                        if i + 2 * w < width and j + 2 * h < height:
                            features.append(([right, bottom], [immediate, bottom_right]))

                        j += 1
                    i += 1
        return np.array(features)

    def select_best(self, classifiers, weights, training_data):
        """
        Selects the best weak classifier for the given weights
          Args:
            classifiers: An array of weak classifiers
            weights: An array of weights corresponding to each training example
            training_data: An array of tuples. The first element is the numpy array of shape (m, n) representing the integral image. The second element is its classification (1 or 0)
          Returns:
            A tuple containing the best classifier, its error, and an array of its accuracy
        """
        best_clf, best_error, best_accuracy = None, float('inf'), None
        for clf in classifiers:
            error, accuracy = 0, []
            for data, w in zip(training_data, weights):
                correctness = abs(clf.classify(data[0]) - data[1])
                accuracy.append(correctness)
                error += w * correctness
            error = error / len(training_data)
            if error < best_error:
                best_clf, best_error, best_accuracy = clf, error, accuracy
        return best_clf, best_error, best_accuracy
    
    def apply_features(self, features, training_data):
        """
        Maps features onto the training dataset
          Args:
            features: an array of tuples. Each tuple's first element is an array of the rectangle regions which positively contribute to the feature. The second element is an array of rectangle regions negatively contributing to the feature
            training_data: An array of tuples. The first element is the numpy array of shape (m, n) representing the integral image. The second element is its classification (1 or 0)
          Returns:
            X: A numpy array of shape (len(features), len(training_data)). Each row represents the value of a single feature for each training example
            y: A numpy array of shape len(training_data). The ith element is the classification of the ith training example
        """
        X = np.zeros((len(features), len(training_data)))
        y = np.array(list(map(lambda data: data[1], training_data)))
        i = 0
        for positive_regions, negative_regions in features:
            feature = lambda ii: sum([pos.compute_feature(ii) for pos in positive_regions]) - sum([neg.compute_feature(ii) for neg in negative_regions])
            X[i] = list(map(lambda data: feature(data[0]), training_data))
            i += 1
        return X, y

    def classify(self, image):
        """
        Classifies an image
          Args:
            image: A numpy 2D array of shape (m, n) representing the image
          Returns:
            1 if the image is positively classified and 0 otherwise
        """
        total = 0
        ii = integral_image(image)
        for alpha, clf in zip(self.alphas, self.clfs):
            total += alpha * clf.classify(ii)
        return 1 if total >= 0.5 * sum(self.alphas) else 0

    def save(self, filename):
        """
        Saves the classifier to a pickle
          Args:
            filename: The name of the file (no file extension necessary)
        """
        with open(filename+".pkl", 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        A static method which loads the classifier from a pickle
          Args:
            filename: The name of the file (no file extension necessary)
        """
        with open(filename+".pkl", 'rb') as f:
            return pickle.load(f)

class WeakClassifier:
    def __init__(self, positive_regions, negative_regions, threshold, polarity):
        """
          Args:
            positive_regions: An array of RectangleRegions which positively contribute to a feature
            negative_regions: An array of RectangleRegions which negatively contribute to a feature
            threshold: The threshold for the weak classifier
            polarity: The polarity of the weak classifier
        """
        self.positive_regions = positive_regions
        self.negative_regions = negative_regions
        self.threshold = threshold
        self.polarity = polarity
    
    def classify(self, x):
        """
        Classifies an integral image based on a feature f and the classifiers threshold and polarity
          Args:
            x: A 2D numpy array of shape (m, n) representing the integral image
          Returns:
            1 if polarity * feature(x) < polarity * threshold
            0 otherwise
        """
        feature = lambda ii: sum([pos.compute_feature(ii) for pos in self.positive_regions]) - sum([neg.compute_feature(ii) for neg in self.negative_regions])
        return 1 if self.polarity * feature(x) < self.polarity * self.threshold else 0
    
    def __str__(self):
        return "Weak Clf (threshold=%d, polarity=%d, %s, %s" % (self.threshold, self.polarity, str(self.positive_regions), str(self.negative_regions))

class RectangleRegion:
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    def compute_feature(self, ii):
        """
        Computes the value of the Rectangle Region given the integral image
        Args:
            integral image : numpy array, shape (m, n)
            x: x coordinate of the upper left corner of the rectangle
            y: y coordinate of the upper left corner of the rectangle
            width: width of the rectangle
            height: height of the rectangle
        """
        return ii[self.y+self.height][self.x+self.width] + ii[self.y][self.x] - (ii[self.y+self.height][self.x]+ii[self.y][self.x+self.width])

    def __str__(self):
        return "(x= %d, y= %d, width= %d, height= %d)" % (self.x, self.y, self.width, self.height)
    def __repr__(self):
        return "RectangleRegion(%d, %d, %d, %d)" % (self.x, self.y, self.width, self.height)


@cuda.jit(device=True)
def index(index):
    #pour ma cg : GTX 1050 2GB
    nb_banks = 16
    nb_core = 640
    index_div = index // nb_banks
    index_mod = index % nb_banks
    offset_index = ((nb_core * index_mod) + index_div)
    return offset_index
    
@cuda.jit
def prescan(input, output, n):
    
    ## alloue mémoire partagée
    temp = cuda.shared.array(12288, dtype=float32)
    tdx = cuda.threadIdx.x
    threadblocks = cuda.blockIdx.x*cuda.blockDim.x
    offset = 1

    ai = tdx
    bi = tdx + (n//2)

    ## on crée des shifted index afin d'éviter les banks conflicts
    shifted_ai = index(ai)
    shifted_bi = index(bi)

     ##on charge l'input dans la mémoire partagée
    temp[shifted_ai] = input[ai + threadblocks]
    temp[shifted_bi] = input[bi + threadblocks]

    #phase montante
    d = n//2
    while d > 0:
        cuda.syncthreads()
        if tdx < d:
            ai = offset*(2*tdx+1)-1
            bi = offset*(2*tdx+2)-1
            shifted_ai = index(ai)
            shifted_bi = index(bi)
            temp[shifted_bi] += temp[shifted_ai]
        offset *= 2
        #on shift d sur la droite
        d >>= 1
    cuda.syncthreads()

    ## on clear le dernier element
    if tdx == 0:
         temp[index(n-1)] = 0
    
    
    #phase descendante 
    d = 1
    while d < n:
        offset >>=1
        cuda.syncthreads()

        if tdx < d:
            ai = offset*(2*tdx+1)-1
            bi = offset*(2*tdx+2)-1
            shifted_ai = index(ai)
            shifted_bi = index(bi)
            t = temp[shifted_ai]
            temp[shifted_ai] = temp[shifted_bi]
            temp[shifted_bi] += t

        d *=2
    cuda.syncthreads()


    #on écrit les resultats sur la mémoire
    output[ai + threadblocks] = temp[shifted_ai]
    output[bi + threadblocks] = temp[shifted_bi]

    cuda.syncthreads()
    
@cuda.jit
def transpose(input, output, width, height):
    TPB = 16
    temp = cuda.shared.array(shape=(TPB, TPB+1), dtype=float32)

    xIndex = cuda.blockIdx.x*TPB + cuda.threadIdx.x
    yIndex = cuda.blockIdx.y*TPB + cuda.threadIdx.y

    if xIndex < width and yIndex < height:
        id_input = yIndex * width + xIndex
        temp[cuda.threadIdx.y][cuda.threadIdx.x] = input[id_input]

    cuda.syncthreads()

    xIndex = cuda.blockIdx.x*TPB + cuda.threadIdx.x
    yIndex = cuda.blockIdx.y*TPB + cuda.threadIdx.y
    if xIndex * height and yIndex * width:
        id_output = yIndex * height + xIndex
        output[id_output] = temp[cuda.threadIdx.x][cuda.threadIdx.y]



@cuda.jit
def integral_image(image):
    input = image
    output = np.zeros(image.shape)
    n = len(image)
    prescan(input,output,n)
    output_transpose = np.zeros(image.shape)
    transpose(output,output_transpose,np.size(image,0),np.size(image,1))
    output = np.zeros(image.shape)
    prescan(output_transpose,output,len(output_transpose))
    return output



""" @cuda.jit      
def integral_image(image):
   
    ii = np.zeros(image.shape)
    s = np.zeros(image.shape)
    for y in range(len(image)):
        for x in range(len(image[y])):
            s[y][x] = s[y-1][x] + image[y][x] if y-1 >= 0 else image[y][x]
            ii[y][x] = ii[y][x-1]+s[y][x] if x-1 >= 0 else s[y][x]
    return ii
 """

