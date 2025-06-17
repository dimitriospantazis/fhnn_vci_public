from sklearn.linear_model import HuberRegressor
from sklearn.ensemble import IsolationForest
from nn_regressor import Neural_Network_Regressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD, PCA, NMF
from sklearn.manifold import TSNE, Isomap
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from umap import UMAP
import matplotlib.pyplot as plt
from visualization import to_poincare
import torch
from hyperbolic_clustering.utils.utils import poincare_dist
import numpy as np 
import networkx as nx
from typing import List
from typing import Tuple    
import time
from tqdm import tqdm
import numpy as np
import os 
import pickle
from utils.constants_utils import NUM_ROIS, NUM_SBJS, NUM_MEG_COLE_ROIS
BAR_WIDTH = 0.35
age_labels = np.load(os.path.join("data", "cam_can_multiple", "age_labels_592_sbj_filtered.npy"))
# meg_age_labels = np.load(os.path.join("data", "meg", "meg_age_labels_180.npy"))

class Age_Model:
    def __init__(self, date : str, log_num : str, projection_type: str ="HR", dataset: str ="Cam-CAN"):
        log_path = os.path.join("logs", "lp", date, log_num)
        print(f"Using Log Path : {log_path}")
        self.embeddings_dir = os.path.join(log_path, 'embeddings')
        projection_type = projection_type.replace("-", "").upper()
        self.projection_type = projection_type
        self.val_indices = self.get_split_indices("val")
        self.test_indices = self.get_split_indices("test")
        # self.train_indices = self.get_split_indices("train")
        self.train_indices = [i for i in range(NUM_SBJS) if i not in self.val_indices and i not in self.test_indices]
        print(self.train_indices, "THESE ARE THE TRAIN INDICES")
        self.dataset = dataset
        if dataset not in ["Cam-CAN", "MEG"]:
            raise ValueError("Dataset must be either Cam-CAN or MEG")
        self.num_rois = NUM_ROIS if self.dataset == "Cam-CAN" else NUM_MEG_COLE_ROIS

    def project_embeddings(self, embeddings_list) -> np.ndarray:
        projection_function = self.get_projection_function()
        # scaled_embeddings = self.scale_embeddings(embeddings_list)
        # projected_embeddings = [projection_function(embeddings) for embeddings in tqdm(scaled_embeddings)]
        projected_embeddings = [projection_function(embeddings) for embeddings in tqdm(embeddings_list)]
        np_projected_embeddings = np.array(projected_embeddings)
        if self.projection_type != "HR": 
            projected_embeddings = np_projected_embeddings.reshape((len(np_projected_embeddings), self.num_rois))
        return projected_embeddings
    
    def detect_outliers(self, projected_embeddings) -> List[int]:
        """
        Detect Outliers, returns a list of 1's and -1's, 1 if inlier, -1 if outlier of size len(projected_embeddings)
        """
        detector_model = IsolationForest(contamination=0.05)  # Adjust the contamination parameter

        is_inlier_or_outlier_list = detector_model.fit_predict(projected_embeddings)
        num_outliers = np.count_nonzero(is_inlier_or_outlier_list == -1)
        print(f"Detected {num_outliers} Outliers")
        return is_inlier_or_outlier_list

    # TODO: Figure out if should scale before or after projection
    def scale_embeddings(self, embeddings_list) -> np.ndarray:
        print("Scaling Embeddings :")
        scaler = StandardScaler()
        scaled_embeddings = [scaler.fit_transform(embeddings) for embeddings in tqdm(embeddings_list)]
        np_scaled_embeddings = np.array(scaled_embeddings)
        scaled_embeddings = np_scaled_embeddings.reshape((len(np_scaled_embeddings), self.num_rois))
        return scaled_embeddings
    def get_projection_function(self):
        projection_function = lambda x : x
        if self.projection_type == "HR": 
            
            def inner_product(u, v):
                return -u[0]*v[0] + np.dot(u[1:], v[1:]) 
            def get_hyperbolic_radius(embeddings):
                origin = np.array([1, 0, 0]) # .to(self.args.device)
                return [np.arccosh(-1 * inner_product(origin, coord)) for coord in embeddings]
            
            def get_poincare_radius(embeddings):
                poincare_origin = torch.Tensor([0, 0]) 
                torch_embeddings = torch.from_numpy(embeddings)
                c = 1.0
                poincare_embeddings = [to_poincare(torch_embedding, c) for torch_embedding in torch_embeddings]
                return [poincare_dist(poincare_embedding, poincare_origin) for poincare_embedding in poincare_embeddings]
            # def get_squared_radius(embeddings):
            #     return [coord[0] ** 2 + coord[1] ** 2 + coord[2] ** 2 for coord in embeddings]
            # projection_function = get_poincare_radius
            projection_function = get_hyperbolic_radius

        elif self.projection_type == "TSNE": projection_function = TSNE(n_components=1, init='random', perplexity=3).fit_transform
        elif self.projection_type == "SVD": projection_function = TruncatedSVD(n_components=1).fit_transform
        elif self.projection_type == "PCA": projection_function = PCA(n_components=1).fit_transform
        elif self.projection_type == "ISOMAP": projection_function = Isomap(n_components=1).fit_transform
        elif self.projection_type == "UMAP": projection_function = UMAP(n_components=1).fit_transform
        else: raise AssertionError(f"Invalid Projection Type : {self.projection_type}!")
        # Other possibilities: MDS, LLE, Laplacian Eigenmaps, etc.
        return projection_function
            
    def get_split_indices(self, split_str):
        split_indices = []
        for split_embeddings_dir in os.listdir(self.embeddings_dir):
            if split_str not in split_embeddings_dir: continue
            _, _, split_index_str = split_embeddings_dir.split("_")
            split_index, _ = split_index_str.split(".")
            split_index = int(split_index)
            split_indices.append(split_index)
        return sorted(split_indices)
        # Sorting the indices should not affect age prediction results... must investigate further...
        # return split_indices

class Age_Model_for_Single_Graphs:
    def __init__(self, projection_type: str ="HR", dataset: str ="Cam-CAN"):
        self.train_indices, self.val_indices, self.test_indices = self.get_split_indices()        
        projection_type = projection_type.replace("-", "").upper()
        self.projection_type = projection_type
        print(self.train_indices, "THESE ARE THE TRAIN INDICES")
        self.dataset = dataset
        if dataset not in ["Cam-CAN", "MEG"]:
            raise ValueError("Dataset must be either Cam-CAN or MEG")
        self.num_rois = NUM_ROIS if self.dataset == "Cam-CAN" else NUM_MEG_COLE_ROIS

    def project_embeddings(self, embeddings_list) -> np.ndarray:
        projection_function = self.get_projection_function()
        # scaled_embeddings = self.scale_embeddings(embeddings_list)
        # projected_embeddings = [projection_function(embeddings) for embeddings in tqdm(scaled_embeddings)]
        projected_embeddings = [projection_function(embeddings) for embeddings in tqdm(embeddings_list)]
        np_projected_embeddings = np.array(projected_embeddings)
        if self.projection_type != "HR": 
            projected_embeddings = np_projected_embeddings.reshape((len(np_projected_embeddings), self.num_rois))
        return projected_embeddings
    
    def detect_outliers(self, projected_embeddings) -> List[int]:
        """
        Detect Outliers, returns a list of 1's and -1's, 1 if inlier, -1 if outlier of size len(projected_embeddings)
        """
        detector_model = IsolationForest(contamination=0.05)  # Adjust the contamination parameter

        is_inlier_or_outlier_list = detector_model.fit_predict(projected_embeddings)
        num_outliers = np.count_nonzero(is_inlier_or_outlier_list == -1)
        print(f"Detected {num_outliers} Outliers")
        return is_inlier_or_outlier_list

    # TODO: Figure out if should scale before or after projection
    def scale_embeddings(self, embeddings_list) -> np.ndarray:
        print("Scaling Embeddings :")
        scaler = StandardScaler()
        scaled_embeddings = [scaler.fit_transform(embeddings) for embeddings in tqdm(embeddings_list)]
        np_scaled_embeddings = np.array(scaled_embeddings)
        scaled_embeddings = np_scaled_embeddings.reshape((len(np_scaled_embeddings), self.num_rois))
        return scaled_embeddings
    def get_projection_function(self):
        projection_function = lambda x : x
        if self.projection_type == "HR": 
            
            def inner_product(u, v):
                return -u[0]*v[0] + np.dot(u[1:], v[1:]) 
            def get_hyperbolic_radius(embeddings):
                origin = np.array([1, 0, 0]) # .to(self.args.device)
                return [np.arccosh(-1 * inner_product(origin, coord)) for coord in embeddings]
            
            def get_poincare_radius(embeddings):
                poincare_origin = torch.Tensor([0, 0]) 
                torch_embeddings = torch.from_numpy(embeddings)
                c = 1.0
                poincare_embeddings = [to_poincare(torch_embedding, c) for torch_embedding in torch_embeddings]
                return [poincare_dist(poincare_embedding, poincare_origin) for poincare_embedding in poincare_embeddings]
            # def get_squared_radius(embeddings):
            #     return [coord[0] ** 2 + coord[1] ** 2 + coord[2] ** 2 for coord in embeddings]
            # projection_function = get_poincare_radius
            projection_function = get_hyperbolic_radius

        elif self.projection_type == "TSNE": projection_function = TSNE(n_components=1, init='random', perplexity=3).fit_transform
        elif self.projection_type == "SVD": projection_function = TruncatedSVD(n_components=1).fit_transform
        elif self.projection_type == "PCA": projection_function = PCA(n_components=1).fit_transform
        elif self.projection_type == "ISOMAP": projection_function = Isomap(n_components=1).fit_transform
        elif self.projection_type == "UMAP": projection_function = UMAP(n_components=1).fit_transform
        else: raise AssertionError(f"Invalid Projection Type : {self.projection_type}!")
        # Other possibilities: MDS, LLE, Laplacian Eigenmaps, etc.
        return projection_function
            
    def get_split_indices(self) -> List[List[int]]:
        train_split_indices, val_split_indices, test_split_indices = [], [], []
        nth_index = 0
        num_sbjs = NUM_SBJS
        TRAIN_SPLIT = 0.7
        VAL_SPLIT = 0.2
        TEST_SPLIT = 0.1
        train_num, val_num = int(num_sbjs * TRAIN_SPLIT), int(num_sbjs * VAL_SPLIT)
        test_num = num_sbjs - train_num - val_num 
        seen = set()
        lower_bound = num_sbjs * nth_index
        upper_bound = lower_bound + num_sbjs
        for split_indices, split_num in zip([train_split_indices, val_split_indices, test_split_indices],
                                        [train_num, val_num, test_num]):
            num_indices = 0
            while num_indices < split_num:
                # index = np.random.randint(0, num_sbjs) # Do not forget filtering! (subtract 5)        
                index = np.random.randint(lower_bound, upper_bound) # Do not forget filtering! (subtract 5)        
                if index in seen: continue
                split_indices.append(index)
                num_indices += 1
                seen.add(index)
        assert not set(train_split_indices).intersection(val_split_indices), "Train and Val Sets should not overlap"
        assert not set(train_split_indices).intersection(test_split_indices), "Train and Test Sets should not overlap"
        assert not set(val_split_indices).intersection(test_split_indices), "Val and Test Sets should not overlap"
            
        return sorted(train_split_indices), sorted(val_split_indices), sorted(test_split_indices)
    

class Age_Predictor(Age_Model):
    def __init__(self, date : str, log_num : str, type_of_regression: str, projection_type: str="HR", architecture: str="FHNN", dataset: str="Cam-CAN", alpha=100):
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
        
        5. Ridge Regression
        
        6. Evaluate Regression Model: MSE
        
        7. Visualize Predicted Age vs. Actual Age
        """
        super().__init__(date, log_num, projection_type, dataset)
        type_of_regression = type_of_regression.lower()
        self.architecture = architecture
        self.dataset = dataset
        if type_of_regression == "linear":
            self.regressor_model = LinearRegression()
        elif type_of_regression == "ridge":
            print("Alpha Parameter :", alpha)
            self.regressor_model = Ridge(alpha = alpha)

        elif type_of_regression == "polynomial":
            raise AssertionError("Polynomial Regression not implemented yet!")
            poly = PolynomialFeatures(degree=2)
            embeddings_poly = poly.fit_transform(embeddings)
            self.regressor_model = LinearRegression()
        elif type_of_regression == "hyperbolic":
            raise AssertionError("Hyperbolic Regression not implemented yet!")
            self.regressor_model = HyperbolicCentroidRegression()
        elif type_of_regression == "random_forest":
            # self.regressor_model = RandomForestRegressor(n_estimators=50, random_state=42)
            self.regressor_model = RandomForestRegressor(n_estimators=100, random_state=42)
            # self.regressor_model = RandomForestRegressor(n_estimators=500, random_state=42)
        
        elif type_of_regression == "neural_network":
            input_size = NUM_ROIS
            hidden_size = NUM_ROIS
            # hidden_size = 200
            output_size = 1
            lr = 0.001
            # lr = 0.0005
            self.regressor_model = Neural_Network_Regressor(input_size, hidden_size, output_size, learning_rate=lr)
        elif type_of_regression == "huber":
            # epsilon = 1.35
            epsilon = 1.0
            max_iter = 100
            self.regressor_model = HuberRegressor(epsilon=epsilon, max_iter=max_iter)
        else:
            raise AssertionError(f"Invalid Regression type : {type_of_regression}!")
        if dataset == "Cam-CAN":
            self.train_age_labels = [age_labels[train_index] for train_index in self.train_indices] 
            self.val_age_labels = [age_labels[val_index] for val_index in self.val_indices]
            self.test_age_labels = [age_labels[test_index] for test_index in self.test_indices]
        elif dataset == "MEG":
            self.train_age_labels = [meg_age_labels[train_index] for train_index in self.train_indices] 
            self.val_age_labels = [meg_age_labels[val_index] for val_index in self.val_indices]
            self.test_age_labels = [meg_age_labels[test_index] for test_index in self.test_indices]
        else:
            raise AssertionError(f"Invalid Dataset : {dataset}!")
        self.model_str = "Linear" if type(self.regressor_model) == LinearRegression \
            else "Ridge" if type(self.regressor_model) == Ridge \
            else "Random Forest" if type(self.regressor_model) == RandomForestRegressor \
            else "Feed-Forward NN" if type(self.regressor_model) == Neural_Network_Regressor \
            else "Huber Regressor" if type(self.regressor_model) == HuberRegressor \
            else "Unknown"
        
    def regression(self) -> float:
        """
        Train, Test, and Plot Regression Model
        """
        self.train()
        predicted_ages, mae_score, mse_score, correlation, r2 = self.test()
        # self.plot_age_labels_vs_predicted_ages(predicted_ages)
        self.visualize_model_parameters(use_jet = False)
        self.plot_age_labels_vs_predicted_ages_curves(predicted_ages)
        self.plot_age_labels_directly_to_predicted_ages_curves(predicted_ages)
        print(f"{self.model_str} Model with Projection {self.projection_type} Mean Absolute Error (MAE):", mae_score)
        print(f"{self.model_str} Model with Projection {self.projection_type} Mean Squared Error (MSE):", mse_score)
        print(f"{self.model_str} Model with Projection {self.projection_type} Correlation:", correlation)
        print(f"{self.model_str} Model with Projection {self.projection_type} R^2 Value:", r2)
        return mae_score, mse_score, correlation, r2
    
    def visualize_model_parameters(self, use_jet=False):
        # plt.figure(figsize = (10, 10))
        plt.figure()
        plt.title(f"{self.model_str} Model with Projection {self.projection_type} Trained Parameters")
        plt.ylabel('Parameter Value')
        plt.xlabel('Region Of Interest (ROI) Index')
        x = np.arange(self.num_rois)
        cmap = plt.cm.jet        
        
        if type(self.regressor_model) == RandomForestRegressor:
            plt.bar(x, self.regressor_model.feature_importances_, color=cmap(x / len(x)))
            return 
        elif type(self.regressor_model) == Neural_Network_Regressor:
            print(f"Model Parameters : {[tensor.shape for tensor in self.regressor_model.parameters()]}")
            row_sums = [torch.sum(tensor, dim=0).detach().numpy() for tensor in self.regressor_model.parameters()]
            print("ROW SUMS : ", row_sums)
            x = np.arange(len(row_sums[0]))
            plt.bar(x, row_sums[0], color=cmap(x / len(x)))
            return
        
            # plt.imshow(self.regressor_model.parameters())
        
        if use_jet:
            plt.bar(x, self.regressor_model.coef_, color=cmap(x / len(x)))
        else:
            plt.bar(x, self.regressor_model.coef_)

    def plot_age_labels_vs_predicted_ages(self, predicted_ages):
        # Generate x-axis values
        plt.figure()
        x = np.arange(len(self.test_indices))
        # Set width of each bar

        # Plotting the barplots
        
        plt.bar(x - BAR_WIDTH/2, self.test_age_labels, BAR_WIDTH, label='Age Label')
        plt.bar(x + BAR_WIDTH/2, predicted_ages, BAR_WIDTH, label='Predicted Age')

        # Set labels, title, and legend
        plt.xticks(x, self.test_indices, rotation=90, fontsize=6)
        plt.xlabel('Subject Index')
        plt.ylabel('Age')
        plt.title(f'{self.model_str} Model with Projection {self.projection_type}')
        plt.legend()
        
        # Show the plot
        plt.show()
    
    def plot_age_labels_vs_predicted_ages_curves(self, predicted_ages):
        
        # Generate x-axis values
        plt.figure()
        x = np.arange(len(self.test_indices))
        # x = np.arange(len(self.test_indices + self.val_indices))
        plt.plot(x, predicted_ages, linestyle='-', marker='v', color='orange', label='Predicted Age', markersize=5)
        # plt.plot(x, self.test_age_labels + self.val_age_labels, linestyle='-', marker='o', color='blue', label='Age Label', markersize=5)
        plt.plot(x, self.test_age_labels, linestyle='-', marker='o', color='blue', label='Age Label', markersize=5)
        # Set labels, title, and legend
        plt.xticks(x, self.test_indices, rotation=90, fontsize=6)
        # plt.xticks(x, self.test_indices + self.val_indices, rotation=90, fontsize=6)
        plt.xlabel('Subject Index')
        plt.ylabel('Age')
        plt.title(f'{self.model_str} Model with Projection {self.projection_type} Predicted Ages')
        plt.ylim(0, 100)
        plt.legend()

        plt.show()
    
    def plot_age_labels_directly_to_predicted_ages_curves(self, predicted_ages):
        
        # Generate x-axis values
        plt.figure()
        x = np.arange(len(self.test_age_labels))
        y = np.arange(len(predicted_ages))
        # x = np.arange(len(self.test_indices + self.val_indices))
        # plt.plot(x, predicted_ages, linestyle='-', marker='v', color='orange', label='Predicted Age', markersize=5)
        # plt.plot(x, self.test_age_labels + self.val_age_labels, linestyle='-', marker='o', color='blue', label='Age Label', markersize=5)
        # plt.plot(x, self.test_age_labels, linestyle='-', marker='o', color='blue', label='Age Label', markersize=5)
        plt.scatter(self.test_age_labels, predicted_ages, c='blue', marker='o', label='Actual vs. Predicted')
        
        # plt.xticks(x, self.test_indices + self.val_indices, rotation=90, fontsize=6)
        plt.xlabel('Subject Age')
        plt.ylabel('Predicted Age')
        plt.title(f'{self.model_str} Model with Projection {self.projection_type} Predicted Ages')
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        
        # Add a diagonal line for reference (perfect prediction)
        plt.plot([min(self.test_age_labels), max(self.test_age_labels)], [min(self.test_age_labels), max(self.test_age_labels)], linestyle='--', color='gray', label='Perfect Prediction')

        plt.grid()
        plt.legend(loc='upper left')
        plt.show()        

    def plot_difference_between_predicted_and_labels(self, predicted_ages):
        # Generate x-axis values
        plt.figure()
        x = np.arange(len(self.test_indices))
        
        plt.bar(x - BAR_WIDTH/2, predicted_ages - self.test_age_labels, BAR_WIDTH, label='Predicted Age - Age Label')
        plt.xticks(x, self.test_indices, rotation=90, fontsize=6)
        # ax.set_xticklabels(self.test_indices, rotation=90)
        plt.xlabel('Subject Index')
        plt.ylabel('Age')
        plt.title(f'{self.model_str} Model with Projection {self.projection_type} Difference Plot')
        plt.legend()

        plt.show()

    
    def test(self) -> float:
        """
        Must make sure training has been done beforehand
        Test Predicted Ages from embeddings
        Returns Predicted Ages, MAE, MSE, Correlation Coefficient, and R^2 between Predicted Ages and Actual Ages
        """
        
        test_embeddings_list = []
        val_embeddings_list = []
        for test_index in self.test_indices:
            test_embeddings = np.load(os.path.join(self.embeddings_dir, f'embeddings_test_{test_index}.npy'))
            test_embeddings_list.append(test_embeddings)
        for val_index in self.val_indices:
            val_embeddings = np.load(os.path.join(self.embeddings_dir, f'embeddings_val_{val_index}.npy'))
            val_embeddings_list.append(val_embeddings)
        # TODO: Change back to only test_embeddings_list
        # test_embeddings_list += val_embeddings_list
        print("Projecting Test Embeddings :")
        projected_embeddings = self.project_embeddings(test_embeddings_list)
        
        print("Scaling Projected Test Embeddings :")
        scaler = StandardScaler()
        projected_embeddings = scaler.fit_transform(projected_embeddings)
        
        is_inlier_or_outlier_list = self.detect_outliers(projected_embeddings)
        
        # test_embeddings_list = [test_embeddings for index, test_embeddings in enumerate(test_embeddings_list) \
        #                             if is_inlier_or_outlier_list[index] == 1]
        projected_embeddings = [projected_embedding for index, projected_embedding in enumerate(projected_embeddings) \
                                    if is_inlier_or_outlier_list[index] == 1]
        self.test_age_labels = [test_age_label for index, test_age_label in enumerate(self.test_age_labels) \
                                    if is_inlier_or_outlier_list[index] == 1]
        self.test_indices = [test_index for index, test_index in enumerate(self.test_indices) \
                                    if is_inlier_or_outlier_list[index] == 1]
        
        if type(self.regressor_model) == LinearRegression \
            or type(self.regressor_model) == Ridge \
            or type(self.regressor_model) == RandomForestRegressor \
            or type(self.regressor_model) == HuberRegressor:
            predicted_ages = self.regressor_model.predict(projected_embeddings)
        if type(self.regressor_model) == Neural_Network_Regressor:
            projected_embeddings_tensor = torch.from_numpy(np.array(projected_embeddings)).clone().detach().to(dtype=torch.float32).squeeze()
            predicted_ages = self.regressor_model.predict(projected_embeddings_tensor)
            predicted_ages = predicted_ages.squeeze() # Reduce extra dimension from [num_test, 1] to [num_test]
            predicted_ages = predicted_ages.detach().numpy()
        print("Age Labels: ", self.test_age_labels)
        print("Predicted Ages: ", predicted_ages)
        
        return predicted_ages, \
                mean_absolute_error(predicted_ages, self.test_age_labels), \
                mean_squared_error(predicted_ages, self.test_age_labels), \
                np.corrcoef(predicted_ages, self.test_age_labels)[0, 1], \
                r2_score(predicted_ages, self.test_age_labels)

    def get_embeddings_to_labels(self):
        embeddings = []
        embeddings_directory = 'embeddings'
        embeddings_to_labels = dict()
        for embeddings_filename in os.listdir(embeddings_directory):
            if os.path.isfile(os.path.join(embeddings_directory, embeddings_filename)):
                _, _, train_index = embeddings_filename.split()
                train_index = int(train_index)
                
                age_label = age_labels[train_index]
                # meg_age_label = meg_age_labels[train_index]

                embeddings = np.load(os.path.join(embeddings_directory, embeddings_filename))
                embeddings_to_labels[tuple(embeddings)] = age_label
        return embeddings_to_labels
    
    def train(self) -> List[Tuple[float]]:
        """
        1. Get embeddings
        2. Get age labels
        3. Perform regression
        4. Return predicted ages with age labels

        """
        
        train_embeddings_list = []
        val_embeddings_list = []
        test_embeddings_list = []
        for train_index in self.train_indices:
            train_embeddings = np.load(os.path.join(self.embeddings_dir, f'embeddings_train_{train_index}.npy'))
            train_embeddings_list.append(train_embeddings)
        for val_index in self.val_indices:
            val_embeddings = np.load(os.path.join(self.embeddings_dir, f'embeddings_val_{val_index}.npy'))
            val_embeddings_list.append(val_embeddings)
        for test_index in self.test_indices:
            test_embeddings = np.load(os.path.join(self.embeddings_dir, f'embeddings_test_{test_index}.npy'))
            test_embeddings_list.append(test_embeddings)
        
        print("Projecting Train Embeddings :")
        projected_embeddings = self.project_embeddings(train_embeddings_list)
        print("Scaling Projected Train Embeddings :")
        scaler = StandardScaler()
        projected_embeddings = scaler.fit_transform(projected_embeddings)
        is_inlier_or_outlier_list = self.detect_outliers(projected_embeddings)
        
        projected_embeddings = [projected_embedding for index, projected_embedding in enumerate(projected_embeddings) \
                                    if is_inlier_or_outlier_list[index] == 1]
        self.train_age_labels = [train_age_label for index, train_age_label in enumerate(self.train_age_labels) \
                                    if is_inlier_or_outlier_list[index] == 1]
        self.train_indices = [train_index for index, train_index in enumerate(self.train_indices) \
                                    if is_inlier_or_outlier_list[index] == 1]
        
        if type(self.regressor_model) == LinearRegression \
            or type(self.regressor_model) == Ridge \
            or type(self.regressor_model) == RandomForestRegressor \
            or type(self.regressor_model) == HuberRegressor:
            # Projection Mapping from 3D to 1D
            self.regressor_model.fit(projected_embeddings, self.train_age_labels)
        elif type(self.regressor_model == Neural_Network_Regressor):

            projected_embeddings_tensor = torch.from_numpy(np.array(projected_embeddings)) \
                .clone() \
                .detach() \
                .to(dtype=torch.float32) \
                .squeeze()
            
            age_labels_tensor = torch.from_numpy(np.array(self.train_age_labels)) \
                .clone() \
                .detach() \
                .to(dtype=torch.float32) \
                .squeeze()
            
            self.regressor_model.train(projected_embeddings_tensor, age_labels_tensor)

        else: raise AssertionError("Invalid Regression Model!")

    def predict_age(self, embeddings) -> float:
        """
        Predict age from embeddings, no label difference calculation
        """
        if type(self.regressor_model) == LinearRegression:
            radii_sqs = [coord[0] ** 2 + coord[1] ** 2 + coord[2] ** 2 for coord in embeddings]
            predicted_age = self.regressor_model.predict(radii_sqs)
        return predicted_age

    def mse_loss(self, predicted_ages_with_age_labels) -> float:    
        """
        Return mean-squared errors between predicted and actual ages
        
        """
        # mse_loss = sum((predicted_age - age_label) ** 2 for predicted_age, age_label 
        #     in predicted_ages_with_age_labels) / len(predicted_ages_with_age_labels)
        predicted_ages = [pred_label[0] for pred_label in predicted_ages_with_age_labels]
        respective_age_labels = [pred_label[1] for pred_label in predicted_ages_with_age_labels]
        mse_loss = mean_squared_error(predicted_ages, respective_age_labels)
        return mse_loss


class Age_Predictor_for_Node_Level_Measures(Age_Predictor):
    def __init__(self, 
                 measure_str : str,
                 date : str, 
                 log_num : str, 
                 type_of_regression: str, 
                 projection_type: str="HR", 
                 architecture: str="FHNN", 
                 dataset: str="Cam-CAN", 
                 alpha=100):

        super().__init__(date, 
                    log_num, 
                    type_of_regression, 
                    projection_type, 
                    architecture, 
                    dataset, 
                    alpha)
        self.train_indices, self.val_indices, self.test_indices = self.get_dataset_split_indices()
        # self.train_indices, self.val_indices, self.test_indices = sorted(self.train_indices), sorted(self.val_indices), sorted(self.test_indices)
        self.train_indices.sort()
        self.val_indices.sort()
        self.test_indices.sort()
        self.train_age_labels = [age_labels[train_index] for train_index in self.train_indices] 
        self.val_age_labels = [age_labels[val_index] for val_index in self.val_indices]
        self.test_age_labels = [age_labels[test_index] for test_index in self.test_indices]
        self.measure_str = measure_str
    
    def regression(self) -> float:
        """
        Train, Test, and Plot Regression Model
        """
        self.train()
        predicted_ages, mae_score, mse_score, correlation, r2 = self.test()
        # self.plot_age_labels_vs_predicted_ages(predicted_ages)
        self.visualize_model_parameters(use_jet = False)
        # self.plot_difference_between_predicted_and_labels(predicted_ages)
        self.plot_age_labels_vs_predicted_ages_curves(predicted_ages)
        self.plot_age_labels_directly_to_predicted_ages_curves(predicted_ages)
        print(f"{self.model_str} Model with Projection {self.projection_type} Mean Absolute Error (MAE):", mae_score)
        print(f"{self.model_str} Model with Projection {self.projection_type} Mean Squared Error (MSE):", mse_score)
        print(f"{self.model_str} Model with Projection {self.projection_type} Correlation:", correlation)
        print(f"{self.model_str} Model with Projection {self.projection_type} R^2 Value:", r2)
        return mae_score, mse_score, correlation, r2
    
    def test(self) -> float:
        """
        Must make sure training has been done beforehand
        Test Predicted Ages from embeddings
        Returns Predicted Ages, MAE, MSE, Correlation, R^2 Coefficient between Predicted Ages and Actual Ages
        """
        
        train_embeddings_list = []
        val_embeddings_list = []
        test_embeddings_list = []
        # with open('node_level_measures_dict.pkl', 'rb') as file:
        #     node_level_measures_dict = pickle.load(file)
        with open('node_level_and_hyper_measures.pkl', 'rb') as file:
            node_level_measures_dict = pickle.load(file)
        measures = node_level_measures_dict[self.measure_str]
        
        # train_embeddings_list = [measures[train_index] for train_index in self.train_indices]
        # val_embeddings_list = [measures[val_index] for val_index in self.val_indices]
        test_embeddings_list = [measures[test_index] for test_index in self.test_indices]
        

        projected_embeddings = test_embeddings_list
            
        print("Scaling Projected Test Embeddings :")
        scaler = StandardScaler()
        projected_embeddings = scaler.fit_transform(projected_embeddings)
        
        # Detect and prune out outliers from regression
        is_inlier_or_outlier_list = self.detect_outliers(projected_embeddings)

        projected_embeddings = [projected_embedding for index, projected_embedding in enumerate(projected_embeddings) \
                                 if is_inlier_or_outlier_list[index] == 1]
        self.test_age_labels = [test_age_label for index, test_age_label in enumerate(self.test_age_labels) \
                                    if is_inlier_or_outlier_list[index] == 1]
        self.test_indices = [test_index for index, test_index in enumerate(self.test_indices) \
                                    if is_inlier_or_outlier_list[index] == 1]
        
        if type(self.regressor_model) == LinearRegression \
            or type(self.regressor_model) == Ridge \
            or type(self.regressor_model) == RandomForestRegressor \
            or type(self.regressor_model) == HuberRegressor:
            
            predicted_ages = self.regressor_model.predict(projected_embeddings)
        if type(self.regressor_model) == Neural_Network_Regressor:

            projected_embeddings_tensor = torch.from_numpy(projected_embeddings).clone().detach().to(dtype=torch.float32).squeeze()
            predicted_ages = self.regressor_model.predict(projected_embeddings_tensor)

            predicted_ages = predicted_ages.squeeze() # Reduce extra dimension from [num_test, 1] to [num_test]
            predicted_ages = predicted_ages.detach().numpy()
        print("Age Labels: ", self.test_age_labels)
        print("Predicted Ages: ", predicted_ages)
        
        return predicted_ages, \
                mean_absolute_error(predicted_ages, self.test_age_labels), \
                mean_squared_error(predicted_ages, self.test_age_labels), \
                np.corrcoef(predicted_ages, self.test_age_labels)[0, 1], \
                r2_score(predicted_ages, self.test_age_labels)

    def train(self) -> List[Tuple[float]]:
        """
        1. Get embeddings
        2. Get age labels
        3. Perform regression
        4. Return predicted ages with age labels

        """
        
        train_embeddings_list = []
        with open('node_level_and_hyper_measures.pkl', 'rb') as file:
            node_level_measures_dict = pickle.load(file)
        measures = node_level_measures_dict[self.measure_str]
        
        train_embeddings_list = [measures[train_index] for train_index in self.train_indices]
        
        projected_embeddings = train_embeddings_list
        print("Scaling Projected Train Embeddings :")
        scaler = StandardScaler()
        projected_embeddings = scaler.fit_transform(projected_embeddings)

        is_inlier_or_outlier_list = self.detect_outliers(projected_embeddings)
        projected_embeddings = [projected_embedding for index, projected_embedding in enumerate(projected_embeddings) \
                                    if is_inlier_or_outlier_list[index] == 1]
        self.train_age_labels = [train_age_label for index, train_age_label in enumerate(self.train_age_labels) \
                                    if is_inlier_or_outlier_list[index] == 1]
        self.train_indices = [train_index for index, train_index in enumerate(self.train_indices) \
                                    if is_inlier_or_outlier_list[index] == 1]

        if type(self.regressor_model) == LinearRegression \
            or type(self.regressor_model) == Ridge \
            or type(self.regressor_model) == RandomForestRegressor \
            or type(self.regressor_model) == HuberRegressor:

            self.regressor_model.fit(projected_embeddings, self.train_age_labels)
            
        elif type(self.regressor_model == Neural_Network_Regressor):
            
            projected_embeddings_tensor = torch.from_numpy(np.array(projected_embeddings)) \
                .clone() \
                .detach() \
                .to(dtype=torch.float32) \
                .squeeze()
            
            age_labels_tensor = torch.from_numpy(np.array(self.train_age_labels)) \
                .clone() \
                .detach() \
                .to(dtype=torch.float32) \
                .squeeze()
            
            self.regressor_model.train(projected_embeddings_tensor, age_labels_tensor)

        else: raise AssertionError("Invalid Regression Model!")

    def plot_age_labels_vs_predicted_ages_curves(self, predicted_ages):
        
        # Generate x-axis values
        plt.figure()
        x = np.arange(len(self.test_indices))
        # x = np.arange(len(self.test_indices + self.val_indices))
        plt.plot(x, predicted_ages, linestyle='-', marker='v', color='orange', label='Predicted Age', markersize=5)
        # plt.plot(x, self.test_age_labels + self.val_age_labels, linestyle='-', marker='o', color='blue', label='Age Label', markersize=5)
        plt.plot(x, self.test_age_labels, linestyle='-', marker='o', color='blue', label='Age Label', markersize=5)
        # Set labels, title, and legend
        plt.xticks(x, self.test_indices, rotation=90, fontsize=6)
        # plt.xticks(x, self.test_indices + self.val_indices, rotation=90, fontsize=6)
        plt.xlabel('Subject Index')
        plt.ylabel('Age')
        plt.title(f'{self.model_str} Model using {self.measure_str} Predicted Ages')
        plt.ylim(0, 100)
        plt.legend()

        plt.show()
    
    def plot_age_labels_directly_to_predicted_ages_curves(self, predicted_ages):
        
        # Generate x-axis values
        plt.figure()
        x = np.arange(len(self.test_age_labels))
        y = np.arange(len(predicted_ages))
        
        plt.scatter(self.test_age_labels, predicted_ages, c='blue', marker='o', label='Actual vs. Predicted')
        
        # plt.xticks(x, self.test_indices + self.val_indices, rotation=90, fontsize=6)
        plt.xlabel('Subject Age')
        plt.ylabel('Predicted Age')
        plt.title(f'{self.model_str} Model using {self.measure_str} Predicted Ages')
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        
        # Add a diagonal line for reference (perfect prediction)
        plt.plot([min(self.test_age_labels), max(self.test_age_labels)], [min(self.test_age_labels), max(self.test_age_labels)], linestyle='--', color='gray', label='Perfect Prediction')

        plt.grid()
        plt.legend(loc='upper left')
        plt.show()        


    # @staticmethod    
    def get_dataset_split_indices(self) -> List[List[int]]:
        
        train_split_indices, val_split_indices, test_split_indices = [], [], []
        import logging
        nth_index = 0
        nth_str = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th"][nth_index]
        num_sbjs = NUM_SBJS
        TRAIN_SPLIT = 0.7
        VAL_SPLIT = 0.2
        TEST_SPLIT = 0.1
        logging.info(f"Using Only {nth_str} {num_sbjs} Subjects")
        train_num, val_num = int(num_sbjs * TRAIN_SPLIT), int(num_sbjs * VAL_SPLIT)
        test_num = num_sbjs - train_num - val_num 

        seen = set()
        lower_bound = num_sbjs * nth_index
        upper_bound = lower_bound + num_sbjs
        # upper_bound = 587 # [490, 587) -> 97
        logging.info("Using 100% of subject indices as training set")
        for split_indices, split_num in zip([train_split_indices, val_split_indices, test_split_indices],
                                        [train_num, val_num, test_num]):
            num_indices = 0
            while num_indices < split_num:
                # index = np.random.randint(0, num_sbjs) # Do not forget filtering! (subtract 5)        
                index = np.random.randint(lower_bound, upper_bound) # Do not forget filtering! (subtract 5)        
                if index in seen: continue
                split_indices.append(index)
                num_indices += 1
                seen.add(index)
        assert not set(train_split_indices).intersection(val_split_indices), "Train and Val Sets should not overlap"
        assert not set(train_split_indices).intersection(test_split_indices), "Train and Test Sets should not overlap"
        assert not set(val_split_indices).intersection(test_split_indices), "Val and Test Sets should not overlap"
            
        return train_split_indices, val_split_indices, test_split_indices
    
class Age_Predictor_for_Single_Graphs(Age_Model_for_Single_Graphs):
    def __init__(self, type_of_regression: str, projection_type: str="HR", architecture: str="FHNN", dataset: str="Cam-CAN", alpha=100):
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
        
        5. Ridge Regression
        
        6. Evaluate Regression Model: MSE
        
        7. Visualize Predicted Age vs. Actual Age
        """
        super().__init__(projection_type, dataset)
        type_of_regression = type_of_regression.lower()
        self.architecture = architecture
        self.dataset = dataset
        if type_of_regression == "linear":
            self.regressor_model = LinearRegression()
        elif type_of_regression == "ridge":
            print("Alpha Parameter :", alpha)
            self.regressor_model = Ridge(alpha = alpha)

        elif type_of_regression == "polynomial":
            raise AssertionError("Polynomial Regression not implemented yet!")
            poly = PolynomialFeatures(degree=2)
            embeddings_poly = poly.fit_transform(embeddings)
            self.regressor_model = LinearRegression()
        elif type_of_regression == "hyperbolic":
            raise AssertionError("Hyperbolic Regression not implemented yet!")
            self.regressor_model = HyperbolicCentroidRegression()
        elif type_of_regression == "random_forest":
            # self.regressor_model = RandomForestRegressor(n_estimators=50, random_state=42)
            self.regressor_model = RandomForestRegressor(n_estimators=100, random_state=42)
            # self.regressor_model = RandomForestRegressor(n_estimators=500, random_state=42)
        
        elif type_of_regression == "neural_network":
            input_size = NUM_ROIS
            hidden_size = NUM_ROIS
            # hidden_size = 200
            output_size = 1
            lr = 0.001
            # lr = 0.0005
            self.regressor_model = Neural_Network_Regressor(input_size, hidden_size, output_size, learning_rate=lr)
        elif type_of_regression == "huber":
            # epsilon = 1.35
            epsilon = 1.0
            max_iter = 100
            self.regressor_model = HuberRegressor(epsilon=epsilon, max_iter=max_iter)
        else:
            raise AssertionError(f"Invalid Regression type : {type_of_regression}!")
        if dataset == "Cam-CAN":
            self.train_age_labels = [age_labels[train_index] for train_index in self.train_indices] 
            self.val_age_labels = [age_labels[val_index] for val_index in self.val_indices]
            self.test_age_labels = [age_labels[test_index] for test_index in self.test_indices]
        elif dataset == "MEG":
            self.train_age_labels = [meg_age_labels[train_index] for train_index in self.train_indices] 
            self.val_age_labels = [meg_age_labels[val_index] for val_index in self.val_indices]
            self.test_age_labels = [meg_age_labels[test_index] for test_index in self.test_indices]
        else:
            raise AssertionError(f"Invalid Dataset : {dataset}!")
        self.model_str = "Linear" if type(self.regressor_model) == LinearRegression \
            else "Ridge" if type(self.regressor_model) == Ridge \
            else "Random Forest" if type(self.regressor_model) == RandomForestRegressor \
            else "Feed-Forward NN" if type(self.regressor_model) == Neural_Network_Regressor \
            else "Huber Regressor" if type(self.regressor_model) == HuberRegressor \
            else "Unknown"
        
    def regression(self) -> float:
        """
        Train, Test, and Plot Regression Model
        """
        self.train()
        predicted_ages, mae_score, mse_score, correlation, r2 = self.test()
        # self.plot_age_labels_vs_predicted_ages(predicted_ages)
        self.visualize_model_parameters(use_jet = False)
        # TODO: Uncomment this
        # self.plot_difference_between_predicted_and_labels(predicted_ages)
        self.plot_age_labels_vs_predicted_ages_curves(predicted_ages)
        self.plot_age_labels_directly_to_predicted_ages_curves(predicted_ages)
        print(f"{self.model_str} Model with Projection {self.projection_type} Mean Absolute Error (MAE):", mae_score)
        print(f"{self.model_str} Model with Projection {self.projection_type} Mean Squared Error (MSE):", mse_score)
        print(f"{self.model_str} Model with Projection {self.projection_type} Correlation:", correlation)
        print(f"{self.model_str} Model with Projection {self.projection_type} R^2 Value:", r2)
        return mae_score, mse_score, correlation, r2
    
    def visualize_model_parameters(self, use_jet=False):
        plt.figure()
        plt.title(f"{self.model_str} Model with Projection {self.projection_type} Trained Parameters")
        plt.ylabel('Parameter Value')
        plt.xlabel('Region Of Interest (ROI) Index')
        x = np.arange(self.num_rois)
        cmap = plt.cm.jet        
        
        if type(self.regressor_model) == RandomForestRegressor:
            plt.bar(x, self.regressor_model.feature_importances_, color=cmap(x / len(x)))
            return 
        elif type(self.regressor_model) == Neural_Network_Regressor:
            print(f"Model Parameters : {[tensor.shape for tensor in self.regressor_model.parameters()]}")
            row_sums = [torch.sum(tensor, dim=0).detach().numpy() for tensor in self.regressor_model.parameters()]
            print("ROW SUMS : ", row_sums)
            x = np.arange(len(row_sums[0]))
            plt.bar(x, row_sums[0], color=cmap(x / len(x)))
            return
        
            # plt.imshow(self.regressor_model.parameters())
        
        if use_jet:
            plt.bar(x, self.regressor_model.coef_, color=cmap(x / len(x)))
        else:
            plt.bar(x, self.regressor_model.coef_)
    
    def test(self) -> float:
        """
        Must make sure training has been done beforehand
        Test Predicted Ages from embeddings
        Returns Predicted Ages, MAE, MSE, Correlation Coefficient, and R^2 between Predicted Ages and Actual Ages
        """
        
        test_embeddings_list = []
        val_embeddings_list = []
        for test_index in self.test_indices:
            log_path = os.path.join("logs", "lp", "592_single_graph_fhnn_runs", str(test_index))
            test_embeddings = np.load(os.path.join(log_path, 'embeddings.npy'))
            test_embeddings_list.append(test_embeddings)
        for val_index in self.val_indices:
            log_path = os.path.join("logs", "lp", "592_single_graph_fhnn_runs", str(val_index))
            val_embeddings = np.load(os.path.join(log_path, 'embeddings.npy'))
            val_embeddings_list.append(val_embeddings)
        
        print("Projecting Test Embeddings :")
        projected_embeddings = self.project_embeddings(test_embeddings_list)
        
        print("Scaling Projected Test Embeddings :")
        scaler = StandardScaler()
        projected_embeddings = scaler.fit_transform(projected_embeddings)
        
        is_inlier_or_outlier_list = self.detect_outliers(projected_embeddings)
        
        projected_embeddings = [projected_embedding for index, projected_embedding in enumerate(projected_embeddings) \
                                    if is_inlier_or_outlier_list[index] == 1]
        self.test_age_labels = [test_age_label for index, test_age_label in enumerate(self.test_age_labels) \
                                    if is_inlier_or_outlier_list[index] == 1]
        self.test_indices = [test_index for index, test_index in enumerate(self.test_indices) \
                                    if is_inlier_or_outlier_list[index] == 1]
        
        if type(self.regressor_model) == LinearRegression \
            or type(self.regressor_model) == Ridge \
            or type(self.regressor_model) == RandomForestRegressor \
            or type(self.regressor_model) == HuberRegressor:
            predicted_ages = self.regressor_model.predict(projected_embeddings)
        if type(self.regressor_model) == Neural_Network_Regressor:
            projected_embeddings_tensor = torch.from_numpy(np.array(projected_embeddings)).clone().detach().to(dtype=torch.float32).squeeze()
            predicted_ages = self.regressor_model.predict(projected_embeddings_tensor)
            predicted_ages = predicted_ages.squeeze() # Reduce extra dimension from [num_test, 1] to [num_test]
            predicted_ages = predicted_ages.detach().numpy()
        print("Age Labels: ", self.test_age_labels)
        print("Predicted Ages: ", predicted_ages)
        
        return predicted_ages, \
                mean_absolute_error(predicted_ages, self.test_age_labels), \
                mean_squared_error(predicted_ages, self.test_age_labels), \
                np.corrcoef(predicted_ages, self.test_age_labels)[0, 1], \
                r2_score(predicted_ages, self.test_age_labels)

    def get_embeddings_to_labels(self):
        embeddings = []
        embeddings_directory = 'embeddings'
        embeddings_to_labels = dict()
        for embeddings_filename in os.listdir(embeddings_directory):
            if os.path.isfile(os.path.join(embeddings_directory, embeddings_filename)):
                _, _, train_index = embeddings_filename.split()
                train_index = int(train_index)
                
                age_label = age_labels[train_index]
                # meg_age_label = meg_age_labels[train_index]

                embeddings = np.load(os.path.join(embeddings_directory, embeddings_filename))
                embeddings_to_labels[tuple(embeddings)] = age_label
        return embeddings_to_labels
    
    def train(self) -> List[Tuple[float]]:
        """
        1. Get embeddings
        2. Get age labels
        3. Perform regression
        4. Return predicted ages with age labels

        """
        
        train_embeddings_list = []
        val_embeddings_list = []
        test_embeddings_list = []
        for train_index in self.train_indices:
            log_path = os.path.join("logs", "lp", "592_single_graph_fhnn_runs", str(train_index))
            train_embeddings = np.load(os.path.join(log_path, 'embeddings.npy'))
            train_embeddings_list.append(train_embeddings)
        for val_index in self.val_indices:
            log_path = os.path.join("logs", "lp", "592_single_graph_fhnn_runs", str(val_index))
            val_embeddings = np.load(os.path.join(log_path, 'embeddings.npy'))
            val_embeddings_list.append(val_embeddings)
        for test_index in self.test_indices:
            log_path = os.path.join("logs", "lp", "592_single_graph_fhnn_runs", str(test_index))
            test_embeddings = np.load(os.path.join(log_path, 'embeddings.npy'))
            test_embeddings_list.append(test_embeddings)
        
        print("Projecting Train Embeddings :")
        projected_embeddings = self.project_embeddings(train_embeddings_list)
        print("Scaling Projected Train Embeddings :")
        scaler = StandardScaler()
        projected_embeddings = scaler.fit_transform(projected_embeddings)
        is_inlier_or_outlier_list = self.detect_outliers(projected_embeddings)
        
        projected_embeddings = [projected_embedding for index, projected_embedding in enumerate(projected_embeddings) \
                                 if is_inlier_or_outlier_list[index] == 1]
        self.train_age_labels = [train_age_label for index, train_age_label in enumerate(self.train_age_labels) \
                                    if is_inlier_or_outlier_list[index] == 1]
        self.train_indices = [train_index for index, train_index in enumerate(self.train_indices) \
                                    if is_inlier_or_outlier_list[index] == 1]
        
        if type(self.regressor_model) == LinearRegression \
            or type(self.regressor_model) == Ridge \
            or type(self.regressor_model) == RandomForestRegressor \
            or type(self.regressor_model) == HuberRegressor:
            # Projection Mapping from 3D to 1D
            self.regressor_model.fit(projected_embeddings, self.train_age_labels)
        elif type(self.regressor_model == Neural_Network_Regressor):

            projected_embeddings_tensor = torch.from_numpy(np.array(projected_embeddings)) \
                .clone() \
                .detach() \
                .to(dtype=torch.float32) \
                .squeeze()
            age_labels_tensor = torch.from_numpy(np.array(self.train_age_labels)) \
                .clone() \
                .detach() \
                .to(dtype=torch.float32) \
                .squeeze()
            
            self.regressor_model.train(projected_embeddings_tensor, age_labels_tensor)

        else: raise AssertionError("Invalid Regression Model!")

    def predict_age(self, embeddings) -> float:
        """
        Predict age from embeddings, no label difference calculation
        """
        if type(self.regressor_model) == LinearRegression:
            radii_sqs = [coord[0] ** 2 + coord[1] ** 2 + coord[2] ** 2 for coord in embeddings]
            predicted_age = self.regressor_model.predict(radii_sqs)
        return predicted_age

    def mse_loss(self, predicted_ages_with_age_labels) -> float:    
        """
        Return mean-squared errors between predicted and actual ages
        
        """
        # mse_loss = sum((predicted_age - age_label) ** 2 for predicted_age, age_label 
        #     in predicted_ages_with_age_labels) / len(predicted_ages_with_age_labels)
        predicted_ages = [pred_label[0] for pred_label in predicted_ages_with_age_labels]
        respective_age_labels = [pred_label[1] for pred_label in predicted_ages_with_age_labels]
        mse_loss = mean_squared_error(predicted_ages, respective_age_labels)
        return mse_loss
    
    def plot_age_labels_directly_to_predicted_ages_curves(self, predicted_ages):
        # Generate x-axis values
        plt.figure()
        x = np.arange(len(self.test_age_labels))
        y = np.arange(len(predicted_ages))
        
        plt.scatter(self.test_age_labels, predicted_ages, c='blue', marker='o', label='Actual vs. Predicted')
        
        # plt.xticks(x, self.test_indices + self.val_indices, rotation=90, fontsize=6)
        plt.xlabel('Subject Age')
        plt.ylabel('Predicted Age')
        plt.title(f'{self.model_str} Model with Projection {self.projection_type} Predicted Ages')
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        
        # Add a diagonal line for reference (perfect prediction)
        plt.plot([min(self.test_age_labels), max(self.test_age_labels)], [min(self.test_age_labels), max(self.test_age_labels)], linestyle='--', color='gray', label='Perfect Prediction')

        plt.grid()
        plt.legend(loc='upper left')
        plt.show()        

    def plot_difference_between_predicted_and_labels(self, predicted_ages):
        # Generate x-axis values
        plt.figure()
        x = np.arange(len(self.test_indices))
        
        plt.bar(x - BAR_WIDTH/2, predicted_ages - self.test_age_labels, BAR_WIDTH, label='Predicted Age - Age Label')
        plt.xticks(x, self.test_indices, rotation=90, fontsize=6)
        # ax.set_xticklabels(self.test_indices, rotation=90)
        plt.xlabel('Subject Index')
        plt.ylabel('Age')
        plt.title(f'{self.model_str} Model with Projection {self.projection_type} Difference Plot')
        plt.legend()

        plt.show()
    
    def plot_age_labels_vs_predicted_ages_curves(self, predicted_ages):
        # Generate x-axis values
        plt.figure()
        x = np.arange(len(self.test_indices))
        # x = np.arange(len(self.test_indices + self.val_indices))
        plt.plot(x, predicted_ages, linestyle='-', marker='v', color='orange', label='Predicted Age', markersize=5)
        # plt.plot(x, self.test_age_labels + self.val_age_labels, linestyle='-', marker='o', color='blue', label='Age Label', markersize=5)
        plt.plot(x, self.test_age_labels, linestyle='-', marker='o', color='blue', label='Age Label', markersize=5)
        # Set labels, title, and legend
        plt.xticks(x, self.test_indices, rotation=90, fontsize=6)
        # plt.xticks(x, self.test_indices + self.val_indices, rotation=90, fontsize=6)
        plt.xlabel('Subject Index')
        plt.ylabel('Age')
        plt.title(f'{self.model_str} Model with Projection {self.projection_type} Predicted Ages')
        plt.ylim(0, 100)
        plt.legend()

        plt.show()