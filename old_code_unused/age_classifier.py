import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import os 
from age_predictor import Age_Model
BAR_WIDTH = 0.35
NUM_BRACKETS = 8
NUM_ROIS = 360

age_labels = np.load(os.path.join("data", "cam_can_multiple", "age_labels_592_sbj_filtered.npy"))

SUBJECT_INDICES_TO_AGE_BRACKETS = dict()
for subject_index, age in enumerate(age_labels):
    if 10 <= age < 20: SUBJECT_INDICES_TO_AGE_BRACKETS[subject_index] = 0
    if 20 <= age < 30: SUBJECT_INDICES_TO_AGE_BRACKETS[subject_index] = 1
    if 30 <= age < 40: SUBJECT_INDICES_TO_AGE_BRACKETS[subject_index] = 2
    if 40 <= age < 50: SUBJECT_INDICES_TO_AGE_BRACKETS[subject_index] = 3
    if 50 <= age < 60: SUBJECT_INDICES_TO_AGE_BRACKETS[subject_index] = 4
    if 60 <= age < 70: SUBJECT_INDICES_TO_AGE_BRACKETS[subject_index] = 5
    if 70 <= age < 80: SUBJECT_INDICES_TO_AGE_BRACKETS[subject_index] = 6
    if 80 <= age < 90: SUBJECT_INDICES_TO_AGE_BRACKETS[subject_index] = 7
class Age_Classifier(Age_Model):
    def __init__(self, 
                date: str, 
                log_num: str, 
                type_of_classifier: str, 
                projection_type: str="SQ-R", 
                architecture: str="FHNN", 
                dataset: str="Cam-CAN", 
                alpha=100):
        """
        1. Data MEG MRI fMRI
            Access PLV Subject Matrices 
        
        2. Get Adjacency Matrix
            Define Threshold
            Binarize Matrix
        
        3. Create Brain Graph
            Training Set    
            Validation Set 
            Test Set
        
        4. Create HGCNN Embeddings
            Visualize Embeddings by plotting in Poincare Disk --> Drew Wilimitis Code
        
        5. SVM Classification
        
        6. Evaluate Classification Model: Accuracy, F1 Score, Precision, Recall
        
        7. Visualize Predicted Age vs. Actual Age
        """
        # def __init__(self, date : str, log_num : str, projection_type: str ="HR", dataset: str ="Cam-CAN"):

        super().__init__(date, log_num, projection_type, dataset)
        type_of_classifier = type_of_classifier.lower()
        projection_type = projection_type.replace("-", "").upper()
        self.architecture = architecture
        self.dataset = dataset
        if type_of_classifier == "linear":
            self.classifier = SVC(kernel = "linear")
        elif type_of_classifier == "rbf":
            self.classifier = SVC(kernel = "rbf")
        elif type_of_classifier == "polynomial":
            raise AssertionError("Polynomial Classifier not implemented yet!")
            poly = PolynomialFeatures(degree=2)
            embeddings_poly = poly.fit_transform(embeddings)
            self.classifier = LinearRegression()
        elif type_of_classifier == "hyperbolic":
            raise AssertionError("Hyperbolic Classifier not implemented yet!")
            self.classifier = HyperbolicCentroidRegression()
        elif type_of_classifier == "random_forest":
            # self.classifier = RandomForestClassifier(n_estimators=200, random_state=21)
            self.classifier = RandomForestRegressor(n_estimators = 100, random_state=21)
        else:
            raise AssertionError(f"Invalid Classifier type : {type_of_classifier}!")
        self.train_age_brackets = [SUBJECT_INDICES_TO_AGE_BRACKETS[train_index] for train_index in self.train_indices]
        self.val_age_brackets = [SUBJECT_INDICES_TO_AGE_BRACKETS[val_index] for val_index in self.val_indices]
        self.test_age_brackets = [SUBJECT_INDICES_TO_AGE_BRACKETS[test_index] for test_index in self.test_indices]
        
        self.model_str = "SVC_Linear" if type_of_classifier == "linear" else "SVC_RBF" if type_of_classifier != "random_forest" else "Random_Forest"
        # self.projection_type = projection_type
        # date = "2023_6_17"
        # log_num = '15'
        log_path = os.path.join("logs", "lp", date, log_num)
        # print(f"Using Log Path : {log_path}")
        self.embeddings_dir = os.path.join(log_path, 'embeddings')
    def classify(self) -> float:
        self.train()
        predicted_age_brackets, accuracy_score = self.test()
        print("Predicted Age Brackets : ", predicted_age_brackets)  
        if self.model_str == "SVC_Linear": self.visualize_model_parameters(use_jet = False)
        self.plot_accuracy_per_age_bracket(predicted_age_brackets)
        self.plot_age_brackets_vs_predicted_ages_brackets(predicted_age_brackets)
        self.plot_age_brackets_vs_predicted_ages_brackets_curves(predicted_age_brackets)
        # TODO: Include F1 Score
        print(f"{self.model_str} Model with Projection {self.projection_type} Accuracy Score:", accuracy_score)
        return accuracy_score
    
    def train(self):
        """
        1. Get embeddings
        2. Get age labels
        3. Perform regression
        4. Return predicted ages with age labels

        """
        train_embeddings_list = []
        val_embeddings_list = []
        for train_index in self.train_indices:
            train_embeddings = np.load(os.path.join(self.embeddings_dir, f'embeddings_train_{train_index}.npy'))
            train_embeddings_list.append(train_embeddings)
        for val_index in self.val_indices:
            val_embeddings = np.load(os.path.join(self.embeddings_dir, f'embeddings_val_{val_index}.npy'))
            val_embeddings_list.append(val_embeddings)
        
        train_embeddings_list += val_embeddings_list
        
        if type(self.classifier) == SVC or type(self.classifier) == RandomForestClassifier or type(self.classifier) == RandomForestRegressor:
            # Projection Mapping from 3D to 1D
            print("Projecting Train Embeddings :")
            projected_embeddings = self.project_embeddings(train_embeddings_list)
            if self.projection_type == "SQR": 
                scaler = StandardScaler()
                projected_embeddings = scaler.fit_transform(projected_embeddings)
            # self.regressor_model.fit(projected_embeddings, self.train_age_labels)
            self.classifier.fit(projected_embeddings, self.train_age_brackets + self.val_age_brackets)
        
    def test(self):
        test_embeddings_list = []
        print(f"THESE ARE THE TEST INDICES: {self.test_indices}")
        print(f"THESE ARE THE TEST BRACKETS: {self.test_age_brackets}")
        for test_index in self.test_indices:
            test_embeddings = np.load(os.path.join(self.embeddings_dir, f'embeddings_test_{test_index}.npy'))
            test_embeddings_list.append(test_embeddings)
        projected_embeddings = self.project_embeddings(test_embeddings_list)
        predicted_age_brackets = self.classifier.predict(projected_embeddings)
        print(f"PREDICTED AGE BRACKETS!!! : {predicted_age_brackets}")
        predicted_age_brackets = [round(predicted_age_bracket) for predicted_age_bracket in predicted_age_brackets]
        # Encode categorical labels to numerical values
        # label_encoder = LabelEncoder()
        # y_encoded = label_encoder.fit_transform(y)

        # # Split data into training and testing sets
        # X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

        # # Create and train a classifier
        # classifier = SVC(kernel='linear', decision_function_shape='ovr')
        # classifier.fit(X_train, y_train)

        # # Make predictions
        # y_pred = classifier.predict(X_test)

        # # Decode predicted labels
        # y_pred_decoded = label_encoder.inverse_transform(y_pred)


        # Calculate accuracy
        accuracy = accuracy_score(self.test_age_brackets, predicted_age_brackets)
        print("Accuracy:", accuracy)
        return predicted_age_brackets, accuracy
    def visualize_model_parameters(self, use_jet: bool=False):
        # plt.figure(figsize = (10, 10))
        plt.figure()
        plt.title(f"{self.model_str} Model with Projection {self.projection_type} Trained Parameters")
        plt.ylabel('Parameter Value')
        plt.xlabel('Region Of Interest (ROI) Index')
        x = np.arange(NUM_ROIS)
        if use_jet: 
            cmap = plt.cm.jet
            # plt.bar(x, self.classifier.coef_, color=cmap(x / len(x)))
        else:
            # plt.bar(x, self.classifier.coef_)
            print(f"SVM CLassifier Weights Dimensions : {self.classifier.coef_.shape}")
            print(f"SVM CLassifier Weights: {self.classifier.coef_}")
    def plot_accuracy_per_age_bracket(self, predicted_age_brackets):
        accuracies_per_age_bracket = dict()
        for age_bracket in range(NUM_BRACKETS):
            age_bracket_indices = [index for index, test_age_bracket in enumerate(self.test_age_brackets) if test_age_bracket == age_bracket]
            age_bracket_specific_predictions = [predicted_age_brackets[age_bracket_index] for age_bracket_index in age_bracket_indices]
            accuracies_per_age_bracket[age_bracket] = accuracy_score([age_bracket] * len(age_bracket_indices), age_bracket_specific_predictions)
        plt.figure()
        # for index, (age_bracket, label_len) in enumerate(zip(age_brackets, label_lens)):
        # plt.bar(age_bracket, label_len, color = cmap(index / len(age_brackets)))
    
        cmap = plt.cm.jet
        plt.title(f"{self.model_str} Model with Projection {self.projection_type} Accuracy Per Age Bracket")
        for index in range(NUM_BRACKETS):
            accuracies = list(accuracies_per_age_bracket.values())
            plt.bar(index, accuracies[index], color = cmap(index / len(accuracies)))
        plt.ylabel("Accuracy")
        plt.xlabel("Age Bracket")

    def plot_age_brackets_vs_predicted_ages_brackets(self, predicted_age_brackets):
        plt.figure()
        plt.title(f"{self.model_str} Model with Projection {self.projection_type} Age Brackets vs Predicted Age Brackets")
        plt.ylabel('Age Bracket')
        plt.xlabel('Subject Index')

        # Plotting the barplots
        x = np.arange(len(self.test_indices))
        plt.bar(x - BAR_WIDTH/2, self.test_age_brackets, BAR_WIDTH, label="Ground Truth Age Bracket")
        plt.bar(x + BAR_WIDTH/2, predicted_age_brackets, BAR_WIDTH, label='Predicted Age Bracket')
        plt.legend()

    def plot_age_brackets_vs_predicted_ages_brackets_curves(self, predicted_age_brackets):
        plt.figure()
        plt.title(f"{self.model_str} Model with Projection {self.projection_type} Age Brackets vs Predicted Age Brackets")
        plt.ylabel('Age Bracket')
        plt.xlabel('Subject Index')

        # Plotting the barplots
        x = np.arange(len(self.test_indices))
        plt.plot(self.test_age_brackets, color="blue", label="Ground Truth Age Bracket", marker="o", markersize=5)
        plt.plot(predicted_age_brackets, color="orange", label="Predicted Age Bracket", marker="v", markersize=5)
        plt.legend()

